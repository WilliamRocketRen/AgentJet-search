from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
import re
from typing import List


def get_translation_quality_system_prompt() -> str:
    """Get the translation quality system prompt."""
    return """
You are an objective translation quality evaluator for academic paper translations from English to Chinese. Your task is to identify ONLY the specific types of errors demonstrated in the provided examples - not general translation quality issues.

Focus (but do not limit to) on issues below (as shown in the examples):

1. **First-person pronoun issues** - Using "我们" instead of "本研究" or "本文" in academic contexts
2. **Abbreviation translation errors** - Using abbreviations when concise Chinese exists (e.g., "GWs" instead of "引力波"), or translating abbreviations that should remain in English (like "EMBB")
3. **Word order problems** - Not adjusting sentence structure to emphasize key points in Chinese academic style
4. **Subject-verb inconsistencies** - Mismatched subjects due to improper sentence structure (e.g., "在...中，本文展示..." where the subject is confused)
5. **Inappropriate word choices** - Using colloquial or incorrect terms instead of proper academic expressions (e.g., "效率" vs "有效性" in certain contexts)
6. **Redundant punctuation** - Unnecessary commas or other punctuation that disrupts Chinese reading flow

**Examples of these errors:**

Example 1:
- Original: "We find that the EMBB is dominated by GW bursts from stellar mass black holes"
- Bad Translation: "我们发现，EMBB主要由恒星级黑洞发出的GWs爆发主导"
- Issues: 1) "我们" should be "本研究/本文", 2) "GWs" should be "引力波" (has concise Chinese), 3) Word order doesn't emphasize the key point
- Good Translation: "本研究发现恒星级黑洞发出的引力波爆发在EMBB中占主导地位"

Example 2:
- Original: "In a previous paper (Gayon & Bois 2008a), we have shown the general efficiency of retrograde resonances for stabilizing compact planetary systems."
- Bad Translation: "在先前的一篇论文（Gayon & Bois 2008a）中，本文展示了逆向共振在稳定紧凑行星系统中的普遍效率。"
- Issues: 1) Subject confusion (在...中，本文...), 2) Redundant comma, 3) "效率" should be "有效性" for better academic expression
- Good Translation: "先前的一篇论文（Gayon & Bois 2008a）阐释了逆向共振在稳定紧凑行星系统中的普遍有效性。"

Example 3:
- Original: "To improve the transferability of ViT, we introduce a novel and effective module, named Domain Transferable-guided Attention Block (DTAB)."
- Bad Translation: "为了提高ViT的迁移能力，本文引入了一个新颖且有效的模块，称为域可迁移引导注意力块（DTAB）"
- Issues: 1) 语言顺序和表达不符合中文习惯。2) 没有在首次出现自定义缩写时，给出英文全称
- Good Translation: "为提高ViT的迁移能力，本文引入了名为“域可迁移引导注意力块”（Transferable-guided Attention Block，DTAB）的新颖模块。"

Example 4:
- Original: Extensive experiments were conducted  on  UCF-HMDB, Kinetics-Gameplay, and Kinetics-NEC Drone datasets, with different backbones, like ResNet101, I3D, and STAM, to verify the effectiveness of TransferAttn compared with state-of-the-art approaches.
- Bad Translation: 在UCF-HMDB、Kinetics-Gameplay和Kinetics-NEC Drone数据集上进行了广泛的实验，使用了不同的骨干网络，如ResNet101、I3D和STAM，以验证TransferAttn与现有最先进方法相比的有效性。
- Issues: 1) 改变语言顺序后，主语缺失。应当填充主语“本研究”或者“本文”。2) 举例时，表述不够简洁。
- Good Translation: 本研究在UCF-HMDB、Kinetics-Gameplay和Kinetics-NEC Drone数据集上进行了广泛的实验, 使用了ResNet101、I3D和STAM等骨干网络来验证TransferAttn与现有最先进方法相比的有效性。

Rate the translation on a scale of 0-3:

0 = Severely impairs readability (multiple critical errors from the categories above that make the text difficult to understand)
1 = Does not impair readability, but numerous errors or significantly reduces Chinese reading efficiency (many instances of the error types above)
2 = Does not impair readability, few errors and not severe (minor instances of the error types above)
3 = No errors from the example categories detected (translation is free of the specific error types demonstrated)

Note:
* For each key issue found, provide the specific error, its type, and where it appears in the translation.
* Be precise about which error category each issue belongs to.
* Focus on objective errors matching the example patterns, not subjective preferences.

Think carefully before flagging any error. Ask yourself: Does this match one of the specific error types from the examples? Is this truly an objective error or just a stylistic preference?

Return your response in this format:
<score>X</score>
<reasoning>Your detailed step-by-step reasoning analyzing the translation against the error categories</reasoning>
<key_issues>
- Error Type: [category]. Error: [specific issue]. Location: [where it appears in the translation]
</key_issues>

The score must be 0, 1, 2, or 3. Each key issue should be on its own line starting with a dash. If no errors are found, the key_issues section should be empty or state "None detected".
"""


TRANSLATION_QUALITY_USER_PROMPT = """
Evaluate the quality of this Chinese translation based on the specific error types demonstrated in the examples.

Original English text:
{original}

Chinese translation to evaluate:
{translation}
"""


def parse_translation_quality_response(text: str) -> dict:
    """Parse XML-formatted translation quality response."""
    score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    issues_match = re.search(r"<key_issues>(.*?)</key_issues>", text, re.DOTALL)

    score = int(score_match.group(1)) if score_match else 3
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text

    key_issues = []
    if issues_match:
        issues_text = issues_match.group(1)
        # Filter out empty lines and "None detected" type messages
        key_issues = [
            line.strip().lstrip("- ")
            for line in issues_text.strip().split("\n")
            if line.strip() and not line.strip().lstrip("- ").lower().startswith("none")
        ]

    return {"score": score, "reason": reasoning, "key_issues": key_issues}


def build_translation_quality_messages(original_text: str, translation: str) -> List[dict]:
    """Build messages for translation quality evaluation."""
    return [
        {"role": "system", "content": get_translation_quality_system_prompt()},
        {
            "role": "user",
            "content": TRANSLATION_QUALITY_USER_PROMPT.format(
                original=original_text,
                translation=translation
            ),
        },
    ]


class TranslationQualityGrader(LLMGrader):
    """Grader for evaluating translation quality based on specific error patterns.

    Score range: 0-3
        0 = Severely impairs readability (multiple critical errors)
        1 = Does not impair readability, but numerous errors or reduces efficiency
        2 = Does not impair readability, few errors and not severe
        3 = No errors from the example categories detected
    """

    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="translation_quality",
            mode=GraderMode.POINTWISE,
            description="Evaluate translation quality based on specific error patterns",
            model=model,
            template="",  # Placeholder, not used
        )

    async def aevaluate(self, original_text: str, translation: str) -> GraderScore:
        """Evaluate translation quality.

        Args:
            original_text: Original English text
            translation: Chinese translation to evaluate

        Returns:
            GraderScore with score 0-3 and identified issues
        """
        try:
            messages = build_translation_quality_messages(original_text, translation)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_translation_quality_response(content)

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["reason"],
                metadata={"key_issues": parsed["key_issues"]},
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))


async def extract_response_content(response) -> str:
    """Extract content from model response."""
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, dict) and 'content' in response:
        return response['content']
    elif isinstance(response, str):
        return response
    else:
        raise ValueError(f"Unable to extract content from response: {type(response)}")
