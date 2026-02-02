
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.task_reader import RouterTaskReader
import re
import os
import threading
import time
import requests
from loguru import logger
from textwrap import dedent
from ajet.copilot.job import AgentJetJob
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet import WorkflowOutput
from ajet.task_reader import RouterTaskReader
from ajet.utils.retry import retry_with_backoff
from ajet.schema.task import Task
from concurrent.futures import ThreadPoolExecutor
from beast_logger import print_listofdict
import asyncio

# Import reward computation from t-agent-reward.py
from t_agent_reward import TranslationQualityGrader, build_translation_quality_messages



LOCAL_DATASET_PATH = "/mnt/data_cpfs/qingxu.fu/agentjet/agentjet/tmp/arxiv_papers/train.parquet"


# Handshake with tinkerscript remote, then send training param to tinkerscript remote (such as model to be trained, algorithm, etc)
dataset = RouterTaskReader(
    reader_type = "huggingface_dat_repo",
    reader_config = AjetTaskReader(
        huggingface_dat_repo = HuggingfaceDatRepo(
            dataset_path = LOCAL_DATASET_PATH
        )
    )
)


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    # Prepare base_url, api_key
    base_url, api_key = (api_baseurl_key.base_url, api_baseurl_key.api_key)
    # Read dataset item
    title = task.metadata['title']
    authors = task.metadata['authors']
    abstract = task.metadata['abstract']
    # Prepare messages
    messages, rough_translate = rough_translate_agent(base_url, api_key, abstract)

    messages, fix_nouns = detect_hard_proper_nouns(base_url, api_key, abstract, rough_translate)
    print_listofdict(messages, header="detect_hard_proper_nouns", mod="c")

    messages, final_translation = produce_final_translation(base_url, api_key, abstract, rough_translate, fix_nouns)
    print_listofdict(messages, header="final_translation", mod="c")

    # Compute reward
    time.sleep(1)
    # Use the translation quality grader from t-agent-reward.py
    from openjudge.models import OpenAIChatModel
    # from openjudge.models.openai_model import OpenAIModel
    grader = TranslationQualityGrader(
        model=OpenAIChatModel(base_url=base_url, api_key=api_key, model="qwen-max")
    )
    grader_score = asyncio.run(grader.aevaluate(original_text=abstract, translation=final_translation))
    raw_reward = grader_score.score / 3.0  # Normalize to 0-1 range (score is 0-3)
    # Return
    return WorkflowOutput(reward=raw_reward, metadata={
        "rough_translate": rough_translate,
        "fix_nouns": fix_nouns,
        "final_translation": final_translation
    })


def detect_hard_proper_nouns(base_url, api_key, abstract, rough_translate):
    messages = [
        {
            "role": "system",
            "content": "You are responsible for detecting translation errors of discipline-specific proper nouns. "
                       "Use json to list all errors found in the translation result and provide correction. "
                       "Json format: [{\"original_word\": \"xxx\", \"wrong_translation\": \"xxx\", \"wrong_reason\": \"xxx\", \"correct_translation\": \"xxx\"}, ...]. "
                       "If no errors are found, return an empty list []."
        },
        {
            "role": "user",
            "content": abstract
        },
        {
            "role": "assistant",
            "content": rough_translate
        },
        {
            "role": "user",
            "content": "Please list all translation errors of discipline-specific proper nouns found in the translation result according to the requirements."
        },
    ]

    # Use raw http requests (non-streaming) to get response
    response = requests.post( f"{base_url}/chat/completions", json = { "model": "qwen-max", "messages": messages, },
                               headers = { "Authorization": f"Bearer {api_key}" } )
    fix_nouns = response.json()['choices'][0]['message']['content']
    messages += [
        {
            "role": "assistant",
            "content": fix_nouns
        }
    ]
    return messages, fix_nouns


def produce_final_translation(base_url, api_key, abstract, rough_translate, fix_nouns):
    """
    Third agent: Apply the corrections from fix_nouns to produce the final polished translation.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional academic translator responsible for producing the final, polished Chinese translation. "
                       "You will receive: 1) the original English abstract, 2) an initial translation, and 3) a list of corrections for proper nouns. "
                       "Your task is to apply all the corrections to produce a final translation that is accurate, fluent, and meets Chinese academic writing standards. "
                       "Ensure that all discipline-specific proper nouns are translated correctly according to the provided corrections. "
                       "Maintain the academic tone and ensure the translation is concise, rigorous, and natural in Chinese."
        },
        {
            "role": "user",
            "content": f"Original English Abstract:\n{abstract}"
        },
        {
            "role": "user",
            "content": f"Initial Translation:\n{rough_translate}"
        },
        {
            "role": "user",
            "content": f"Corrections for Proper Nouns:\n{fix_nouns}"
        },
        {
            "role": "user",
            "content": "Please produce the final, corrected Chinese translation by applying all the corrections listed above. "
                       "Output only the final translation without any explanations or additional text."
        },
    ]

    # Use raw http requests (non-streaming) to get response
    response = requests.post( f"{base_url}/chat/completions", json = { "model": "qwen-max", "messages": messages, },
                               headers = { "Authorization": f"Bearer {api_key}" } )
    final_translation = response.json()['choices'][0]['message']['content']

    messages += [
        {
            "role": "assistant",
            "content": final_translation
        }
    ]

    return messages, final_translation


def rough_translate_agent(base_url, api_key, abstract):
    messages = [
        {
            "role": "system",
            "content":
                "You are a professional language translator. "
                "Translate the given Academic English text into Chinese accurately. "
                "During the translation process, it is necessary to meet the linguistic norms of Chinese academic papers "
                "such as conforming to the logic of the Chinese language, being simple, rigorous, and concise, "
                "and avoiding the use of first-person pronouns when passive voice is appropriate. "
                "Ensure that specialized terms are translated correctly according to academic standards. "
                "Replace 我们 with 本研究 or 本文. "
                "If an abbreviation is short in Chinese, use Chinese. "
                "If an abbreviation is long in Chinese, use abbreviation. "
        },
        {
            "role": "user",
            "content": abstract
        }
    ]

    examples = [
        {
            "original": "We find that the EMBB is dominated by GW bursts from stellar mass black holes",
            "hint": "1. 我们->本研究/本文（删除第一人称） 2. GWs->引力波（有简洁的中文表达），但EMBB保留（没有简洁的中文表达） 3. 调换语序，这句话中的重点是“恒星级黑洞发出的引力波”，所以调换语序突出重点。",
            "bad": "我们发现，EMBB主要由恒星级黑洞发出的GWs爆发主导",
            "good": "本研究发现恒星级黑洞发出的引力波爆发在EMBB中占主导地位",
        },
        {
            "original": "In a previous paper (Gayon &amp; Bois 2008a), we have shown the general efficiency of retrograde resonances for stabilizing compact planetary systems.",
            "bad": "在先前的一篇论文（Gayon & Bois 2008a）中，本文展示了逆向共振在稳定紧凑行星系统中的普遍效率。",
            "hint": "修复主语，删除冗余的逗号，替换“效率”为“有效性”更符合学术表达。",
            "good": "先前的一篇论文（Gayon & Bois 2008a）阐释了逆向共振在稳定紧凑行星系统中的普遍有效性。",
        },
    ]

    # add examples to system prompt
    for ex in examples:
        messages[0]['content'] += f"\n\nExample:\n\tOriginal: {ex['original']}\n\tHint: {ex['hint']}\n\tBad Translation: {ex['bad']}\n\tGood Translation: {ex['good']}"

    # Use raw http requests (non-streaming) to get response
    response = requests.post( f"{base_url}/chat/completions", json = { "model": "qwen-max", "messages": messages, },
                               headers = { "Authorization": f"Bearer {api_key}" } )
    rough_translate = response.json()['choices'][0]['message']['content']
    # print(rough_translate)

    messages += [
        {
            "role": "assistant",
            "content": rough_translate
        }
    ]

    return messages, rough_translate




for i, task in enumerate(dataset.generate_training_tasks()):
    if i >= 2:
        execute_agent(
            task,
            OpenaiBaseUrlAndApiKey(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.environ.get("DASHSCOPE_API_KEY", "")
            )
        )
