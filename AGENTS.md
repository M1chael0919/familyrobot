# Family Robot Project - Agent Instructions
# Family Robot 项目 - 协作说明

## Purpose / 目的

This repository builds a family interaction robot demo.
本仓库用于实现一个家庭交互机器人 Demo。

The system should:
系统需要做到：

* detect people in camera frames
* 识别画面中的人
* track people across frames
* 跨帧跟踪同一个人
* recognize family members
* 识别家庭成员身份
* re-identify people after disappearance
* 人离开后再回来时尽量恢复到原身份
* generate simple personalized greetings
* 生成简单的个性化问候

## Required Reading / 必读文档

Before making any change, read:
开始任何修改前，必须先阅读：

1. `SPEC.md`
2. `TASKS.md`

If either file is missing, empty, or unclear, treat that as a documentation gap and fix the foundation first.
如果任一文件缺失、为空或内容不清晰，要先补齐文档基础，再继续实现。

## Task Rules / 任务规则

* Work on exactly one task at a time.
* 一次只处理一个任务。
* Always take the first unchecked task in `TASKS.md`.
* 永远从 `TASKS.md` 中第一个未勾选任务开始。
* Do not start future tasks early.
* 不要提前做后面的任务。
* Do not add unrelated files.
* 不要新增无关文件。
* Do not modify completed tasks unless the task explicitly requires it.
* 除非任务明确要求，否则不要改动已完成任务。

## Development Rules / 开发规则

* Use Python 3.11+（3.12）.
* 使用 Python 3.11 及以上版本（当前环境为 3.12）。
* Use type hints for new code.
* 新代码使用类型注解。
* Prefer clear, small classes and functions.
* 优先使用清晰、短小的类和函数。
* Use `dataclass` when it improves clarity.
* 需要时优先使用 `dataclass` 提高清晰度。
* Follow PEP 8.
* 遵循 PEP 8。
* Add tests for new modules or behavior.
* 新模块或新行为要补测试。

## Architecture Rules / 架构规则

Required stack:
必需技术栈：

* YOLO for person detection
* YOLO 用于人体检测
* DeepSORT for multi-object tracking
* DeepSORT 用于多目标跟踪

Optional additions:
可选增强：

* InsightFace
* FAISS
* OpenCV
* Edge-TTS

Important identity rule:
重要身份规则：

* Track ID is temporary.
* Track ID 只是临时编号。
* Identity is permanent.
* Identity 才是长期身份。
* Never use Track ID as the final identity.
* 绝不能把 Track ID 当成最终身份。

Example:
示例：

* Track 7 -> Father
* Track 7 -> Father
* Track 7 disappears
* Track 7 消失
* Track 12 -> Father
* Track 12 -> Father
* Identity stays Father
* 身份始终保持为 Father

## Communication Rules / 沟通规则

Before implementing:
实现前：

1. Identify the current task.
2. 明确当前任务。
3. Explain the implementation plan.
3. 说明实现计划。
4. Wait for approval if the user asks for it.
4. 如果用户要求，先等待确认。

After implementing:
实现后：

1. Run tests.
2. 运行测试。
3. Update `TASKS.md`.
4. 更新 `TASKS.md`。
5. Summarize the change.
6. 总结本次修改。
7. Stop.
8. 停止，不要自动进入下一个任务。

Do not continue to the next task automatically.
不要自动继续下一个任务。

## Out of Scope / 不在范围内

Do not implement these unless explicitly requested:
除非用户明确要求，否则不要实现：

* LLM conversation systems
* LLM 对话系统
* RAG systems
* RAG 系统
* agent frameworks
* agent 框架
* cloud deployment
* 云端部署

## Pipeline / 主链路

Camera -> YOLO -> DeepSORT -> Face Recognition -> Identity -> Greeting
摄像头 -> YOLO -> DeepSORT -> 人脸识别 -> 身份 -> 问候
