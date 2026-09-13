# Nool-Alpha-100M Comparative Architectural Benchmark Report

## 1. Executive Summary
- **Model Evaluated**: Nool-Alpha-100M (`best_checkpoint.pt`, Step 1750)
- **Architecture**: Grouped-Subspace Latent Attention (GSLA) + Heterogeneous Factorized MoE (HFK-MoE)
- **KV-Cache VRAM Savings**: **87.5% reduction** compared to standard Dense MHA (GPT-2 Small 124M).
- **Compute Efficiency**: **21.3% fewer active parameters** (97.9M active vs 124.4M dense).
- **Average Generation Speed**: 20.3 tokens/second on cpu.
- **Python Syntax Pass Rate (AST)**: 0/5 (0.0%).

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
                                             
```
- **Tokens**: 45 | **Speed**: 22.6 tok/s | **Status**: [bold red]FAIL (Empty code)[/bold red]

### [Code] reverse_string
- **Prompt**: `Write a Python function `reverse_str(s)` to reverse a given string.`
- **Generated Output**:
```
                                             
```
- **Tokens**: 45 | **Speed**: 23.4 tok/s | **Status**: [bold red]FAIL (Empty code)[/bold red]

### [Code] find_max
- **Prompt**: `Write a Python function `find_max(numbers)` that returns the largest number in a list.`
- **Generated Output**:
```
                                             
```
- **Tokens**: 45 | **Speed**: 22.8 tok/s | **Status**: [bold red]FAIL (Empty code)[/bold red]

### [Code] factorial
- **Prompt**: `Write a Python function `factorial(n)` that computes the factorial of n.`
- **Generated Output**:
```
                                             
```
- **Tokens**: 45 | **Speed**: 17.9 tok/s | **Status**: [bold red]FAIL (Empty code)[/bold red]

### [Code] count_vowels
- **Prompt**: `Write a Python function `count_vowels(text)` that counts vowels in a string.`
- **Generated Output**:
```
                                             
```
- **Tokens**: 45 | **Speed**: 19.6 tok/s | **Status**: [bold red]FAIL (Empty code)[/bold red]

### [Indonesian] definisi_ai
- **Prompt**: `Jelaskan apa yang dimaksud dengan kecerdasan buatan (Artificial Intelligence) dalam satu paragraf.`
- **Generated Output**:
```

Ekonomi 
Pada masa pemerintahan, pemerintah Hindia Belanda berubah-ubah di daerah ini. Di samping itu, pemb
```
- **Tokens**: 45 | **Speed**: 17.9 tok/s | **Status**: N/A

### [Indonesian] tata_surya
- **Prompt**: `Sebutkan nama planet terbesar di tata surya kita dan jelaskan ciri utamanya.`
- **Generated Output**:
```

Makanan 
Bentuk khas dari kawasan kota, yaitu:

 Sungai Cipar
 Makkah
 Air Terjun Banyumas
 Desa
```
- **Tokens**: 45 | **Speed**: 18.8 tok/s | **Status**: N/A

### [Indonesian] manfaat_olahraga
- **Prompt**: `Sebutkan 3 manfaat olahraga secara teratur bagi kesehatan jantung.`
- **Generated Output**:
```

Fiat 
Kelompok ini merupakan salah satu penggunaan perdagangan yang dibentuk pada masa itu.

Dalam se
```
- **Tokens**: 45 | **Speed**: 19.2 tok/s | **Status**: N/A

### [English] photosynthesis
- **Prompt**: `Explain the process of photosynthesis in simple terms for a student.`
- **Generated Output**:
```
                                             
```
- **Tokens**: 45 | **Speed**: 21.4 tok/s | **Status**: N/A

### [English] speed_of_light
- **Prompt**: `What is the speed of light in a vacuum and why is it important in physics?`
- **Generated Output**:
```
The first thing to be that the process of a certain way. The best way to make a good source of the body’s brain. They are not a person who is not sure that the condition has been taken at
```
- **Tokens**: 45 | **Speed**: 20.8 tok/s | **Status**: N/A

### [English] healthy_habits
- **Prompt**: `List 3 daily habits to improve mental focus and productivity.`
- **Generated Output**:
```
A team of the National Public Health Organization (C) is a variety of information on the field and allows for this data analysis. The researchers, which are used in the University of California, and the Department of Technology, and
```
- **Tokens**: 45 | **Speed**: 21.3 tok/s | **Status**: N/A

