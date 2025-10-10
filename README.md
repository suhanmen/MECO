# $MEC^3O$: Multi-Expert Consensus for Code Time Complexity Prediction

📖 [Paper](https://~~~~~)

This repo support the paper "**$MEC^3O$**: Multi-Expert Consensus for Code Time Complexity Prediction".


## Updates
- 10/10/2025

## Abstract
Predicting the complexity of source code is essential for software development and algorithm analysis. Recently, [Baik et al.](https://github.com/sybaik1/CodeComplex/tree/main?tab=readme-ov-file) (2025) introduced CodeComplex for code time complexity prediction. The paper shows that LLMs without fine-tuning struggle with certain complexity classes. This suggests that no single LLM excels at every class, but rather each model shows advantages in certain classes. 

We propose **$MEC^3O$**, a multi-expert consensus system, which extends the multi-agent debate frameworks. $MEC^3O$ assigns LLMs to complexity classes based on their performance and provides them with class-specialized instructions, turning them into experts. These experts engage in structured debates, and their predictions are integrated through a weighted consensus mechanism. 

Our expertise assignments to LLMs effectively handle Degeneration-of-Thought, reducing reliance on a separate judge model, and preventing convergence to incorrect majority opinions. Experiments on CodeComplex show that $MEC^3O$ outperforms the open-source baselines, achieving at least 10% higher accuracy and macro-F1 scores. It also surpasses GPT-4o-mini in macro-F1 scores on average and demonstrates competitive on-par F1 scores to GPT-4o and GPT-o4-mini on average. This demonstrates the effectiveness of multi-expert debates and weight consensus strategy to generate the final predictions.

## About $MEC^3O$
![Full Picture](figures/overview.png)

**$MEC^3O$** (**M**ulti-**E**xpert **C**onsensus for **C**ode **C**omplexity Prediction) is a novel approach for predicting the **time complexity of source code**. Unlike traditional **single-LLM** approaches or **multi-agent debate (MAD)** frameworks, $MEC^3O$ strategically assigns **expert roles** to LLMs based on their **demonstrated proficiency** in different complexity classes and integrates their opinions through a **weighted consensus mechanism**.

---

### 🔹 How $MEC^3O$ Works

$MEC^3O$ operates in three key steps:

### **Step 1: Expertise Assignment**
- A **development subset (X_dev)** is randomly sampled from the full dataset (**10%** of the data).  
- This subset maintains a **balanced distribution** across **seven complexity classes**:  
  `O(1), O(log n), O(n), O(n log n), O(n²), O(n³), O(2ⁿ)`.
- Each **LLM's class-specific performance** is evaluated using **macro-F1 scores**, and the best-performing model for each class is designated as the **expert** for that class.
- Each expert receives **class-specialized instructions**, fine-tuning their reasoning for their assigned complexity category.

### **Step 2: Multi-Expert Debate**
- For each test code snippet, $MEC^3O$ assigns it to the **experts** selected in Step 1.  
- **Experts generate independent predictions**, providing both a **complexity label** and a **justification** for their reasoning.  
- The experts **exchange opinions**, allowing them to refine their predictions.  
- If an expert **finds a valid counterargument** from another expert, it can **revise its answer**, preventing premature convergence to incorrect reasoning.

### **Step 3: Weighted Consensus**
- $MEC^3O$ uses a **Weighted Expertise-Confidence Consensus (WECC) function** to determine the final complexity classification.  
- The final complexity class **ĉ** is determined by weighting:
  - **Expertise weight (`w_E`)** → Prioritizes experts predicting within their assigned class.  
  - **Confidence weight (`w_conf`)** → Derived from the expert’s **logit-based confidence** in its prediction.  
- This prevents over-reliance on a **majority vote**, ensuring that **domain specialists** have more influence.

---

### **🚀 $MEC^3O$ Achievements**
✅ **Avoids Degeneration-of-Thought (DoT)** → Experts can **challenge incorrect reasoning** rather than getting stuck on initial biases.  
✅ **No Separate Judge Model Needed** → Instead of relying on an external judge, $MEC^3O$ **trusts class-specialized experts** and weights their decisions accordingly.  
✅ **Superior Performance** → Achieves **at least 10% higher accuracy & macro-F1 scores** than open-source baselines and surpasses **GPT-4** in overall performance.  

---

### Results of $MEC^3O$ on CodeComplex Benchmark
| Model Category | Model | Java Acc. | Java F1 | Python Acc. | Python F1 | Average Acc. | Average F1 |
|----------------|-------------------------------|:-----:|:----:|:------:|:----:|:------:|:----:|
| **Single LLM** | Zero-Shot Instruction | 52.00 | 44.00 | 50.20 | 40.60 | 51.10 | 42.30 |
|                | Seven-Shot Instruction | 56.30 | 48.90 | 48.00 | 39.40 | 52.15 | 44.15 |
|                | CoT | 54.08 | 45.79 | 52.86 | 44.06 | 53.47 | 44.93 |
|                | Self-Consistency | 51.84 | 42.45 | 51.22 | 40.73 | 51.53 | 41.59 |
|                | Reflexion | 53.47 | 43.89 | 52.24 | 41.96 | 52.86 | 42.93 |
| **Multi-Agent Debate** | Multiagent (Majority) | 54.49 | 50.21 | 52.86 | 49.97 | 53.68 | 50.09 |
|                     | Multiagent (Judge) | 54.90 | 45.10 | 55.30 | 44.60 | 55.10 | 44.85 |
|                     | MAD | 46.33 | 39.72 | 40.00 | 36.36 | 43.17 | 38.04 |
|                     | RECONCILE | 55.92 | 52.79 | 55.31 | 51.11 | 55.62 | 51.95 |
|                     | CMD | 56.53 | 47.07 | 55.31 | 45.69 | 55.92 | 46.38 |
| **Commercial LLMs** | GPT-4o | **71.72** | 62.22 | 61.09 | 53.08 | **66.41** | **57.65** |
|                     | GPT-4o-mini | 64.96 | 55.68 | 56.09 | 48.40 | 60.53 | 52.04 |
|                     | GPT-o4-mini | 65.12 | **62.31** | **62.31** | **54.23** | 63.72 | 58.27 |
| **Multi-Expert (Ours)** | **MEC³O** | **61.02** | **61.16** | **57.55** | **53.51** | **59.29** | **57.34** |

---

## Installation
~~~shell
conda env create --file setting/envirometn.yaml
conda activate meco
~~~
Clone the repository and set up the environment.

## Getting Started
The following scripts guide you through running $MEC^3O$ step by step:

### **1️⃣ Single LLM Inference**
~~~shell
sh ./scripts/main.sh Single <your_gpu_devices>
~~~
Runs single-LLM inference to evaluate model performance across all complexity classes.
This step identifies potential candidates for expert assignment.

### **2️⃣ Find Expertise**
~~~shell
sh ./scripts/main.sh Find_expertise <your_gpu_devices>
~~~
Analyzes results from Step 1 to assign each LLM an expert role for its strongest complexity class.
Each expert receives class-specialized instructions optimized for reasoning within its domain.

### **3️⃣ Multi-Expert Debate**
~~~shell
sh ./scripts/main.sh MECO <your_gpu_devices>
~~~
Launches the multi-expert debate phase, where class-specific experts exchange, refine, and justify their predictions.
Finally, $MEC^3O$ integrates their outputs using a weighted consensus mechanism to produce the final complexity classification.
