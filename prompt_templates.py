"""Versioned text templates used to construct science-question prompts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A complete prompt template and its repeatable exemplar block."""

    prompt: str
    example: str


@dataclass(frozen=True)
class GenerationStrategy:
    """A named set of structural requirements for generated questions."""

    title: str
    instruction: str


DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_STRATEGY = "general"


GENERATION_STRATEGIES = {
    "general": GenerationStrategy(
        title="通用生成",
        instruction=(
            "生成与 exemplar 在科学领域、知识深度和问题风格上相近的问题。"
            "问题形式可以多样，但必须具有明确的科学解答目标。"
        ),
    ),
    "multi_step": GenerationStrategy(
        title="多步推理",
        instruction=(
            "每个问题必须至少包含 2～3 个相互依赖的推理步骤；后一步需要使用前一步的"
            "结论，不能通过单个公式代入或单个事实直接得到答案。reasoning 应清楚展示"
            "完整推理链。"
        ),
    ),
    "multi_constraint": GenerationStrategy(
        title="多条件约束",
        instruction=(
            "每个问题必须给出多个彼此相关且确实影响结论的条件。正确作答必须综合所有"
            "关键条件，忽略其中任意一个都可能得到错误或不完整的结论；reasoning 应说明"
            "每个条件发挥的作用。"
        ),
    ),
    "boundary_condition": GenerationStrategy(
        title="边界条件",
        instruction=(
            "围绕科学规律的适用范围、极限情形、临界点或失效前提设计问题。问题应要求"
            "判断结论何时成立、何时改变或为何在边界处失效；答案必须明确指出适用条件，"
            "不能把局部规律无限外推。"
        ),
    ),
    "causal_mechanism": GenerationStrategy(
        title="因果机制",
        instruction=(
            "生成要求解释某种科学现象为什么发生、通过哪些过程产生的问题。答案必须给出"
            "从原因到结果的机制链，区分因果关系与表面相关性，不能只复述现象。"
        ),
    ),
    "competing_mechanisms": GenerationStrategy(
        title="竞争机制",
        instruction=(
            "问题中必须存在至少两种科学上合理、可能影响结果的机制，并提供足以比较它们的"
            "具体条件。要求判断在当前条件下哪种机制占主导、为什么占主导，以及次要机制"
            "为何被抑制或贡献较小。"
        ),
    ),
    "counterfactual": GenerationStrategy(
        title="反事实推理",
        instruction=(
            "先建立一个科学上成立的基准情景，再明确改变一个关键因素，并默认其他条件保持"
            "不变。问题应要求比较改变前后的结果并解释变化原因，不能同时含糊地改变多个"
            "变量。"
        ),
    ),
    "numerical_derivation": GenerationStrategy(
        title="数值推导",
        instruction=(
            "每个问题必须需要公式、计算、比例、量纲或数量级推导。题目应提供求解所需的"
            "数值、单位和必要常数，不依赖未说明的数据；reasoning 应列出关键公式、计算"
            "过程和单位或量纲检查。"
        ),
    ),
    "experimental_reasoning": GenerationStrategy(
        title="实验推理",
        instruction=(
            "给出自包含的实验设置、关键控制条件以及观测结果，要求据此反推原因、机制或"
            "合理结论。答案应区分观测直接支持的结论与尚不能排除的解释，并在必要时指出"
            "对照或额外测量的作用。"
        ),
    ),
    "confusable_concepts": GenerationStrategy(
        title="易混淆概念",
        instruction=(
            "围绕两个或多个相近但不同的科学概念设计具体情景，要求根据判别标准选择并"
            "解释正确概念。答案必须明确比较这些概念的关键差异，不能只分别罗列定义。"
        ),
    ),
}


PROMPT_TEMPLATES = {
    "v1": PromptTemplate(
        prompt="""你将看到若干来自真实开放式科学问答 benchmark 的问题。

这些问题属于相同的科学子领域。

【Topic】
{topic}

【Subtopic】
{subtopic}

请参考这些问题的知识深度、问题形式和解题要求，生成新的开放式科学问题。

【生成策略】
{strategy_title}

【策略要求】
{strategy_instruction}

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
9. 科学正确性优先于策略复杂度。不得为了满足策略形式而使用错误前提、虚构规律、
   缺失必要条件，或强行构造没有明确答案的问题；
10. 每个问题同时给出 question、answer 和 reasoning。

只输出合法的 JSON 数组，不要输出 Markdown 代码块或其他说明。""",
        example="""【示例 {index}】

问题：
{question}

参考答案：
{answer}""",
    ),
}
