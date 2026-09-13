# Nool-Alpha-100M Comparative Architectural Benchmark Report

## 1. Executive Summary
- **Model Evaluated**: Nool-Alpha-100M (`best_sft_checkpoint.pt`, Step 900)
- **Architecture**: Grouped-Subspace Latent Attention (GSLA) + Heterogeneous Factorized MoE (HFK-MoE)
- **KV-Cache VRAM Savings**: **87.5% reduction** compared to standard Dense MHA (GPT-2 Small 124M).
- **Compute Efficiency**: **21.3% fewer active parameters** (97.9M active vs 124.4M dense).
- **Average Generation Speed**: 17.6 tokens/second on cpu.
- **Python Syntax Pass Rate (AST)**: 4/5 (80.0%).

## 2. KV-Cache VRAM Footprint: GSLA vs Standard MHA (GPT-2 Small 124M)
| Context Length | Batch Size | GPT-2 Small MHA | GQA 4:1 (Llama-style) | Nool-Alpha GSLA | VRAM Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 512 tokens | 1 | 14.4 MB | 3.6 MB | **2.2 MB** | **-84.7%** |
| 512 tokens | 8 | 115.2 MB | 28.8 MB | **17.6 MB** | **-84.7%** |
| 1024 tokens | 1 | 28.8 MB | 7.2 MB | **4.4 MB** | **-84.7%** |
| 1024 tokens | 8 | 230.4 MB | 57.6 MB | **35.2 MB** | **-84.7%** |
| 2048 tokens | 1 | 57.6 MB | 14.4 MB | **8.8 MB** | **-84.7%** |
| 2048 tokens | 8 | 460.8 MB | 115.2 MB | **70.4 MB** | **-84.7%** |
| 4096 tokens | 8 | 921.6 MB | 230.4 MB | **140.8 MB** | **-84.7%** |

*Note: GSLA compresses KV cache into low-rank latent content (d_c=192) and decoupled RoPE (d_pe=32), requiring only 224 floats per token layer instead of 1536 floats in GPT-2.*

## 3. Parameter & Computational Efficiency
| Architecture Attribute | GPT-2 Small (124M) | SmolLM-135M | Nool-Alpha-100M |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 124.4M | 135.0M | 111.2M |
| **Active Parameters / Token** | 124.4M (100%) | 135.0M (100%) | **97.9M (88.0%)** |
| **Attention Mechanism** | Dense MHA | Dense GQA (3:1) | **GSLA (Latent Subspace)** |
| **Feed-Forward Layer** | Dense GELU MLP | Dense SwiGLU MLP | **HFK-MoE (Top-2 of 8 Experts)** |
| **Global Residual Highway** | No | No | **Yes (tanh(alpha) * RMSNorm)** |
| **Logit Soft-Capping** | No | No | **Yes (30.0 tanh)** |
| **FLOPs per Token** | ~248.8 MFLOPs | ~270.0 MFLOPs | **~195.8 MFLOPs (-21.3%)** |

## 4. Multi-Domain Task & AST Syntax Validation
### [Code] is_prime
- **Prompt**: `Write a Python function `is_prime(n)` that returns True if n is a prime number.`
- **Generated Output**:
```
def factorial(n): 
    if n == 0: 
        return 1
    elif n == 0: 
      
```
- **Tokens**: 45 | **Speed**: 15.9 tok/s | **Status**: [bold red]FAIL (SyntaxError: expecte)[/bold red]

### [Code] reverse_string
- **Prompt**: `Write a Python function `reverse_str(s)` to reverse a given string.`
- **Generated Output**:
```
def reverse_string(s):
    if s == s[0] in s[1]:
        return False
```
- **Tokens**: 34 | **Speed**: 17.3 tok/s | **Status**: [bold green]PASS (Valid AST)[/bold green]

### [Code] find_max
- **Prompt**: `Write a Python function `find_max(numbers)` that returns the largest number in a list.`
- **Generated Output**:
```
def sum_of_number(n):
    sum = 0
    for i in range(2, len(nums)):
        sum += 1

```
- **Tokens**: 45 | **Speed**: 18.2 tok/s | **Status**: [bold green]PASS (Valid AST)[/bold green]

### [Code] factorial
- **Prompt**: `Write a Python function `factorial(n)` that computes the factorial of n.`
- **Generated Output**:
```
def is_prime(n):
    if n == 0:
        return False
    elif n == 1:
        return
```
- **Tokens**: 45 | **Speed**: 18.3 tok/s | **Status**: [bold green]PASS (Valid AST)[/bold green]

### [Code] count_vowels
- **Prompt**: `Write a Python function `count_vowels(text)` that counts vowels in a string.`
- **Generated Output**:
```
def count_palindrome(string):
    return string.split(' ')
```
- **Tokens**: 21 | **Speed**: 20.0 tok/s | **Status**: [bold green]PASS (Valid AST)[/bold green]

### [Indonesian] definisi_ai
- **Prompt**: `Jelaskan apa yang dimaksud dengan kecerdasan buatan (Artificial Intelligence) dalam satu paragraf.`
- **Generated Output**:
```
Pada tahun 2020, kita mengambil kemampuan untuk membantu mereka menjadi lebih efektif dan menggunakan teknologi
```
- **Tokens**: 45 | **Speed**: 16.6 tok/s | **Status**: N/A

### [Indonesian] tata_surya
- **Prompt**: `Sebutkan nama planet terbesar di tata surya kita dan jelaskan ciri utamanya.`
- **Generated Output**:
```
Salah satu contoh dari laut adalah bintang yang menggunakan warna-warna yang berbeda.
```
- **Tokens**: 35 | **Speed**: 17.7 tok/s | **Status**: N/A

### [Indonesian] manfaat_olahraga
- **Prompt**: `Sebutkan 3 manfaat olahraga secara teratur bagi kesehatan jantung.`
- **Generated Output**:
```
1. Ketergantungan: Penyakit yang lebih baik untuk menghasilkan kesehatan mental dan mental yang tepat, serta mening
```
- **Tokens**: 45 | **Speed**: 17.5 tok/s | **Status**: N/A

### [English] photosynthesis
- **Prompt**: `Explain the process of photosynthesis in simple terms for a student.`
- **Generated Output**:
```
The process of using artificial intelligence is to reduce the risk of developing and manage their tasks, as well as needed. This means that it can be used to increase efficiency, improve efficiency, and ensure that the environment is being used
```
- **Tokens**: 45 | **Speed**: 18.6 tok/s | **Status**: N/A

### [English] speed_of_light
- **Prompt**: `What is the speed of light in a vacuum and why is it important in physics?`
- **Generated Output**:
```
The maximum of the Earth's atmosphere is one of the most significant impact on the world. The total cost of water is approximately $2,000. 

The temperature of the Earth's atmosphere is the number of water
```
- **Tokens**: 45 | **Speed**: 17.9 tok/s | **Status**: N/A

### [English] healthy_habits
- **Prompt**: `List 3 daily habits to improve mental focus and productivity.`
- **Generated Output**:
```
1. **Predicting:** A good understanding of the quality of life is essential for the environment, and it can be used to help you make a more effective way to achieve your overall health. This helps to reduce
```
- **Tokens**: 45 | **Speed**: 17.0 tok/s | **Status**: N/A

