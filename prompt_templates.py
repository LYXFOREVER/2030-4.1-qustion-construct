"""Versioned text templates used to construct science-question prompts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A complete prompt template and its repeatable exemplar block."""

    prompt: str
    example: str


DEFAULT_PROMPT_VERSION = "v1"


PROMPT_TEMPLATES = {
    "v1": PromptTemplate(
        prompt="""你将看到若干来自真实开放式科学问答 benchmark 的问题。

这些问题属于相同的科学子领域。

【Topic】
{topic}

【Subtopic】
{subtopic}

请参考这些问题的知识深度、问题形式和解题要求，生成新的开放式科学问题。

{examples}

请生成 {num_questions} 个新的科学问题。

要求：

1. 必须是开放式问答，不要生成选择题；
2. 新问题应与示例处于相同或相近的科学子领域；
3. 不得只替换示例中的数字、实体、材料名、物种名等；
4. 不要直接改写或轻微变形原问题；
5. 新问题应具有明确、可验证的标准答案；
6. 问题应尽量需要一定的科学分析、解释、计算或推导；
7. 避免纯粹询问一个简单的孤立事实；
8. 不要生成依赖图片才能理解的问题；
9. 每个问题同时给出 question、answer 和 reasoning。

只输出合法的 JSON 数组，不要输出 Markdown 代码块或其他说明。""",
        example="""【示例 {index}】

问题：
{question}

参考答案：
{answer}""",
    ),
}

