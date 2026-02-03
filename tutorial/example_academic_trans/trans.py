
import re
import os
import time
import asyncio
import requests
import threading
from loguru import logger
from textwrap import dedent

from ajet import WorkflowOutput
from ajet.schema.task import Task
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.retry import retry_with_backoff
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from beast_logger import print_listofdict

# Import reward computation from trans_reward.py
from openjudge.models import OpenAIChatModel
from .trans_reward import TranslationQualityGrader, build_translation_quality_messages, examples


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

@retry_with_backoff(max_retry=3)
def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    # Prepare base_url, api_key
    base_url, api_key = (api_baseurl_key.base_url, api_baseurl_key.api_key)
    grader_base_url, grader_api_key = ("https://dashscope.aliyuncs.com/compatible-mode/v1", os.environ.get("DASHSCOPE_API_KEY", ""))
    # Read dataset item
    title = task.metadata['title']
    authors = task.metadata['authors']
    abstract = task.metadata['abstract']

    messages, rough_translate = rough_translate_agent(base_url, api_key, abstract)
    # print_listofdict(messages, header="rough_translate_agent", mod="c")

    messages, fix_nouns = detect_hard_proper_nouns(messages, base_url, api_key, abstract, rough_translate)
    # print_listofdict(messages, header="detect_hard_proper_nouns", mod="c")

    messages, final_translation = produce_final_translation(messages, base_url, api_key, abstract, rough_translate, fix_nouns)
    print_listofdict(messages, header="final_translation", mod="c")

    grader = TranslationQualityGrader(
        model=OpenAIChatModel(base_url=grader_base_url, api_key=grader_api_key, model="qwen-max")
    )
    grader_score = asyncio.run(grader.aevaluate(original_text=abstract, translation=final_translation))
    raw_reward = grader_score.score  # Normalize to 0-1 range (score is 0-3)
    return WorkflowOutput(reward=raw_reward, metadata={
        "rough_translate": rough_translate,
        "fix_nouns": fix_nouns,
        "final_translation": final_translation
    })


def detect_hard_proper_nouns(messages, base_url, api_key, abstract, rough_translate):
    messages = messages + [

        {
            "role": "user",
            "content":  "You new job is to detect translation errors of discipline-specific proper nouns. "
                        "Use json to list all errors found in the translation result and provide correction. "
                        "Json format: [{\"original_word\": \"xxx\", \"wrong_translation\": \"xxx\", \"wrong_reason\": \"xxx\", \"correct_translation\": \"xxx\"}, ...]. "
                        "If no errors are found, return an empty list []."
                        "Please list all translation errors of discipline-specific proper nouns found in the translation result according to the requirements."
        },
    ]

    response = requests.post( f"{base_url}/chat/completions", json = { "model": "qwen-turbo", "messages": messages, }, headers = { "Authorization": f"Bearer {api_key}" } )
    fix_nouns = response.json()['choices'][0]['message']['content']
    messages += [
        {
            "role": "assistant",
            "content": fix_nouns
        }
    ]
    return messages, fix_nouns


def produce_final_translation(messages, base_url, api_key, abstract, rough_translate, fix_nouns):
    messages = messages + [
        {
            "role": "user",
            "content": "Please produce the final, corrected Chinese translation by applying all the corrections listed above. "
                       "Output only the final translation without any explanations or additional text."
        },
    ]

    response = requests.post( f"{base_url}/chat/completions", json = { "model": "qwen-turbo", "messages": messages, }, headers = { "Authorization": f"Bearer {api_key}" } )
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

    for ex in examples:
        messages[0]['content'] += f"\n\nExample:\n\tOriginal: {ex['original']}\n\tBad Translation: {ex['bad']}\n\tHint: {ex['hint']}\n\tGood Translation: {ex['good']}"
    response = requests.post( f"{base_url}/chat/completions", json = { "model": "qwen-turbo", "messages": messages, }, headers = { "Authorization": f"Bearer {api_key}" } )
    rough_translate = response.json()['choices'][0]['message']['content']
    messages += [
        {
            "role": "assistant",
            "content": rough_translate
        }
    ]

    return messages, rough_translate



if __name__ == "__main__":

    for i, task in enumerate(dataset.generate_training_tasks()):
        execute_agent(
            task,
            OpenaiBaseUrlAndApiKey(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.environ.get("DASHSCOPE_API_KEY", "")
            )
        )


