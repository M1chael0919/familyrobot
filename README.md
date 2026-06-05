# Family Robot

这是一个面向家庭场景的本地交互机器人 Demo。
系统目标不是“只会看见人”，而是能在家庭里长期区分成员、记住身份、在合适的时机给出简单问候，并支持语音唤醒与本地语音交互。

## 场景说明

这个项目模拟的是机器人进入家庭后的常见工作方式：

- 客厅里会同时出现爸爸、妈妈、孩子、老人
- 同一个人会反复进出画面，Track ID 可能变化，但身份不能变
- 人可能戴眼镜、口罩、侧脸、半遮挡，或者在光线变化下出现
- 机器人需要知道“这是谁”，而不只是“现在画面里有一个人”
- 语音交互更像唤醒式助手，不是持续聊天系统
- 语音唤醒时，需要知道是哪个人唤醒的
所以，这个系统更接近“家庭陪伴 + 家庭看护 + 轻交互”场景，而不是单纯的视频检测工具。

## 为什么只用 YOLO + DeepSORT 不够

`YOLO + DeepSORT` 很适合做两件事：

1. 检测画面里有没有人
2. 给这个人一个临时的 Track ID 并维持短期连续性

但它有几个天然局限：

- `Track ID` 只是临时编号，不是身份
- 人一旦离开画面再回来，Track ID 很容易变
- DeepSORT 解决的是“跟踪同一个轨迹”，不是“识别同一个家庭成员”
- 它无法回答“这是爸爸还是妈妈”
- 它也无法支撑“离开再回来后，身份仍然保持不变”

所以，若只用 `YOLO + DeepSORT`，系统最多只能说“有一个人，编号 7”，却不能稳定地说“这是爸爸”。

## 这套系统的合理之处

本项目把视觉问题拆成了三层：

1. **检测层**：YOLO 负责找出人
2. **跟踪层**：DeepSORT 负责维持短期轨迹连续性
3. **身份层**：人脸登记 + embedding 匹配 + ReID 恢复，负责长期身份稳定

这样设计的好处是：

- 轨迹可以变，身份不变
- 人离开再回来，仍可恢复到同一个家庭成员
- 识别结果能支撑后面的问候和语音交互
- 语音只负责“唤起和触发”，不替代身份判断

一句话概括：
`YOLO + DeepSORT` 负责“看见并跟住”，`Identity Layer` 负责“认出来并记住”。

## 当前功能

- 人体检测与跟踪
- 本地人脸登记
- 人脸 embedding 提取与身份匹配
- 轨迹丢失后的 ReID 恢复
- 未知人员处理
- 实时 GUI
- 被动问候
- TTS 输出
- 唤醒词 + ASR
- 录制视频原音播放与音视频事件对齐

## 目录结构

- `src/`：主代码
- `tests/`：测试
- `docs/`：架构和设计说明
- `data/enrollment/`：人脸登记数据
- `models/`：本地模型文件 release处下载

## 模型与数据位置

为了方便复现，模型和登记数据都放在项目根目录下：

- `models/yolov8n.pt`
- `models/vosk-model-small-cn-0.22`
- `models/insightface/buffalo_l`
- `data/enrollment/`

其中：

- `data/enrollment/enrollment.json` 是登记清单
- `data/enrollment/images/` 下是每个成员的登记图片
- `insightface` 不再默认写到用户目录缓存

## 环境要求

- Python 3.12
- 建议使用 Conda
- Windows 本地摄像头或本地视频文件
- demo测试环境：i5 12400 rtx4060 cuda13.1 torch2.5.1+cu121 torchvision0.20.1+cu121
## 安装

1. 创建环境

```bash
conda create -n familyrobot312 python=3.12 -y
```

2. 激活环境

```bash
conda activate familyrobot312
```

3. 安装依赖
建议先安装cuda版本的torch 本项目对环境有一定要求
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121                  #安装带cuda版本的torch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"              #检验下torch版本
pip install -r requirements.txt
```

4. 模型下载
模型已经上传到release上，下载解压后放在根目录就行

当前 `requirements.txt` 已按已验证环境锁定版本，方便直接复现；如果你要切换 Python / CUDA / GPU 环境，建议先单独确认 `torch` 和 `onnxruntime` 的安装组合。

## 运行方式

主入口顺序建议：

1. 先打开登记 GUI，录入家庭成员身份
2. 再用实时识别入口看摄像头或视频推理
3. 如果要验证视频里的音频唤醒，再走视频联动入口

### 主要入口

```bash
python .\src\enrollment_main.py                  # 打开身份登记 GUI，新增/查看/删除本地成员
python src/realtime_main.py --source 0 --speak   # 实时摄像头识别 + 被动问候 + TTS
python src/realtime_main.py --source ".\test.mp4" --video-voice  # 本地视频识别 + 音频联动回放
```

### 其他辅助入口

```bash
python src/main.py --source sample               # 视觉链路最小 Demo，只看检测和跟踪结果
python src/realtime_main.py --source sample      # 实时摄像头或视频输入，只显示检测框、轨迹 ID、身份标签
python src/voice_main.py --audio mic              # 单独测试唤醒词、ASR 和语音路由
python src/voice_main.py --audio path\to\test.wav  # 单独测试一段音频文件的转录和唤醒
python src/video_alignment_main.py ".\test.mp4"   # 只看视频音频和帧时间轴的对齐结果
python src/video_voice_main.py ".\test.mp4" --display  # 视频音频联动回放并显示对齐后的事件
```

## 运行说明

- `enrollment_main.py` 先用于登记家庭成员
- `realtime_main.py --speak` 用于摄像头或视频的主识别演示
- `realtime_main.py --video-voice` 用于本地视频的音视频联动测试
- 其余入口主要用于单独调试视觉、语音和对齐链路

## 设计原则

- Track ID 是临时的
- Identity 是长期的
- 唤醒词只负责触发，不负责身份判断
- 身份问候依赖身份层，不依赖单帧轨迹编号
- 所有核心能力尽量本地运行，便于面试展示和离线复现

## 开发复盘

这次实现时没有单独先抽出 `harness` 层，而是优先把视觉、身份、语音和 GUI 的主链路跑通了。这样做的好处是能更快验证 Demo 闭环，但在视频/摄像头复现、音视频对齐、单元与集成测试切分这些地方，调试成本会更高。后续如果继续扩展，建议把运行入口、测试脚本和复现步骤收拢到独立的 `harness` 文档或目录里，开发体验会更稳。

## 文档

- `AGENTS.md`：项目协作规则和开发约束
- `SPEC.md`：需求说明
- `PRD.md`：产品定义
- `TASKS.md`：任务清单
- `docs/ai-collab.md`：与 AI 协作的可复用方法论和经验记录
- `docs/architecture.md`：架构说明
- `docs/design-decisions.md`：关键设计决策
- `docs/blockers.md`：已知阻塞项
