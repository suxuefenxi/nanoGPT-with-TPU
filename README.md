# nanoGPT-with-TPU

![loss curve](/assets/loss_curve.png)

用 Kaggle 的 TPU v5e-8 进行分布式预训练，使用 FineWeb-Edu 10B 语料，大概训练 7.5 小时就可以得到一个“能说会道”的 GPT。
另外，目前的大模型对 PyTorch + TPU 的训练经验普遍不足（TPU 生态以 JAX/TF 为主，PyTorch 资料稀少），生成的代码容易踩坑。我基于本项目的实战经验，用 AI 整理了一份 `SKILL.md`，供大模型参考如何用 Accelerate 库在 TPU 上训练。你可以直接使用，也可以自己用 AI 再蒸馏一份（看到这里还不点点STAR⭐）。

## 模型概况

| 项目 | 配置 |
|------|------|
| 参数量 | 1.23 亿 |
| 架构 | 12 层 Transformer, 12 头, 768 维 |
| 位置编码 | RoPE (旋转位置编码) |
| 激活函数 | ReLU² |
| 训练精度 | bf16 混合精度 |
| 序列长度 | 1024 |
| 全局 batch size | 524288 tokens |
| 训练吞吐量 | ~40 万 tokens/s |
| 训练时长        | ~7.5 小时 (1 epoch)              |

### 模型结构与优化亮点
* **TPU 专属混合精度防 NaN 设计**：在 bf16 混合精度下，对 LayerNorm 的方差计算、RoPE 旋转编码和 Attention softmax 强制使用 `fp32`，避免数值溢出。
* **移除 `lm_head`**：通过 `F.linear(x, wte.weight)` 替代，处理了 TPU 下 weight tying（权重共享）与 Accelerate 初始化带来的限制。
* **XLA 与显存优化**：开启梯度检查点 (Gradient Checkpointing) 将中间激活显存降至 O(1)，并规避了多卡数据并行与 XLA 图编译时的死锁与 OOM 陷阱。

---

## 如何运行

使用fineweb.py获得100个shard作为数据集(99个作为训练集，1个作为验证集), 放到kaggle上，在kaggle导入build-nanogpt.ipynb，记得修改路径，导入刚刚你放到kaggle的fineweb数据集，并添加swanlab的api（可选）。然后耐心等待7.5个小时就好啦🤗

---

## 结果展示

训练 5000 步的采样结果：
```
[prompt] I'm a computer science student,
  sample 0: I'm a computer science student, or a computer scientist. ...
- There are a class of computer scientists called computer scientists, and in many cases computer scientists  （看来很喜欢计算机科学家了）
  sample 1: I'm a computer science student, who works with computer science as part of a science department, whose goal, in this regard, was to establish the world's
  sample 2: I'm a computer science student, I think of a computer science course. Maybe the students can have a textbook lesson about how to improve the computer science concepts.
  sample 3: I'm a computer science student, and I'm trying to develop my next computer program.
```

训练完成（1 epoch）的采样结果：
```
[prompt] I'm a computer science student,
  sample 0: I'm a computer science student, working in the areas of computer technology and computer science, so I've come up with a lot of these concepts before a while
  sample 1: I'm a computer science student, and the only people in my class who don't want to be in college are students.  （你不要乱说啊）
This is because my teacher has
  sample 2: I'm a computer science student, who has studied computer science for 25 years. It is pretty easy to find the answer - the question, "How do I  （好家伙学了25年是吧）
  sample 3: I'm a computer science student, and you can watch me through the program.
```

我这里采样长度比较短，你可以调大一些。可以看到模型还是能说出语法比较正确，但有趣的语句。

---

## 致谢 (Acknowledgements)

本项目受 Andrej Karpathy 的 [build-nanogpt](https://github.com/karpathy/build-nanogpt) 教程的大量启发。他的教学视频讲得非常好，通俗易懂且干货满满。如果你想从零开始真正理解和手搓大语言模型，强烈推荐大家去看他的原版视频！
