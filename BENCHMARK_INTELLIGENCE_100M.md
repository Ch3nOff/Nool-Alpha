# Nool-Alpha-100M vs. 100M-Class Peer Intelligence & Reasoning Benchmark

**Date:** 2026-09-13 16:37:19  
**Evaluation Scope:** Cognitive & Intelligence Benchmark across 100M-parameter models on Math, Logic, Python Code, and Instruction Following.  
**Hardware Platform:** CPU Execution (Direct head-to-head on identical hardware).

---

## 1. Executive Scorecard

| Evaluation Dimension | GPT-2 Small (124M) | SmolLM-135M-Instruct | Nool-Alpha-100M (Reasoning SFT) |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 124.4M | 134.5M | **111.17M** |
| **Active Parameters / Token** | 124.4M (100%) | 134.5M (100%) | **97.90M (88.0%)** |
| **Attention Architecture** | Dense MHA | Dense GQA (3:1) | **GSLA (Latent Subspace)** |
| **KV-Cache Memory Footprint** | 1536 floats/token/layer | 384 floats/token/layer | **224 floats/token/layer (-87.8%)** |
| **Average Decoding Speed** | 61.6 tok/s | 58.3 tok/s | **14.3 tok/s** |
| **Step-by-Step CoT Emergence Rate** | 0.0% | 0.0% | **54.5%** |
| **Python AST Syntax Pass Rate** | 0/3 (0.0%) | 0/3 (0.0%) | **1/3 (33.3%)** |

---

## 2. Qualitative Output Analysis by Domain

### Test 1 [Math]: `discount_tax`
**Prompt:**
> A store offers a 20% discount on a $150 jacket. If sales tax is 8%, what is the final price? Calculate step by step.

#### Nool-Alpha-100M (Reasoning SFT):
```text
<llm-code>
# distance from the given information
number_of_first_price = 30 / 100
# how much the cost of the original price
cost_per_c = total_price_in_bought + amount_cost_in_cost
total_price_in_cost
</llm-code>
<llm-code-output>
25.0
</llm-code-output>
Thus there are \
```
- Speed: `15.4 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
To calculate the final price, we need to subtract the sale tax amount from the total cost of the item and divide it by 100. Let's say that the total cost of the item is $300. The sale tax rate for this item is 4%. We can use the formula:

$(sale_tax - (total_cost * 100)) / 100 = $300 ÷ 100
= ($
```
- Speed: `55.0 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
The customer service reps have been very helpful in helping me out with this issue, and I would like to thank them for their help! Thank you again!!
```
- Speed: `57.8 tok/s` | CoT Triggered: `False` | AST: `N/A`

### Test 2 [Math]: `algebra_solve`
**Prompt:**
> Solve for x: 3x + 7 = 22. Show your step-by-step reasoning.

#### Nool-Alpha-100M (Reasoning SFT):
```text
<think>
Okay, so I have this equation to solve: x equals 8. Hmm, I remember that when you have a function like f(x) = x + 7, the function is defined as x = -7. So, I need to find where x is the interval of x and y are positive or negative.

Alright, let's start by recalling what the value of a function is. The function is given as x squared plus 6 times (x minus 2).
```
- Speed: `15.9 tok/s` | CoT Triggered: `True` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
To solve this problem, we need to find the value of x that makes the given expression equal to zero. Since 3x+7 is a linear equation with two variables and one coefficient, it can be represented as 10(x) - 6(x). Therefore, the equation becomes 5x - 9 = 18. To isolate x on one side of the equation, let's first multiply both sides by 5: 5x - 9 *
```
- Speed: `54.6 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
, 1 | 2 , 4 : { "1" , 5 } . ( a ) == b ? [ 0 ] : 'A' ; # => return true; end function test_theorem () {} assertTrue(a) if @test && !@endfunction else $(( - 6 * 16 ))[0] || 8 <= 20 === 30 // error? here is the result of testing! Test it out and see how you do with this example :) End Function
```
- Speed: `66.0 tok/s` | CoT Triggered: `False` | AST: `N/A`

### Test 3 [Math]: `speed_distance`
**Prompt:**
> A car travels at a constant speed of 60 km/h for 2.5 hours. How far does the car travel in total?

#### Nool-Alpha-100M (Reasoning SFT):
```text
Let's solve this problem using Python code.
<llm-code>
# amount of water in a day
number_of_per_hour = 2 * 3 / 2
# the total time is the sum of the two hours
time_of_slices_per_day = distance_in_week * (1 - distance_of_pens)

# calculate the distance by the time
total_time = time_of_first_days
```
- Speed: `13.0 tok/s` | CoT Triggered: `True` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
To find the distance traveled, we need to know the time taken by the car and its initial velocity (v0) as well as its final velocity (Vf). We can use the formula:

distance = sqrt(3 * (v0 / Vf))

where v is the initial velocity and Vf is the final velocity.

Let's denote the initial velocity as v0 and the final velocity as Vf. Then, the distance traveled can be calculated using the
```
- Speed: `62.0 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
"The motorcyclist is moving fast and very slowly, but there are no signs that he has stopped." The driver's hand moves as if it were going to slow down or stop abruptly (in fact I would suggest stopping). If you see any indication of slowing your vehicle then do not proceed! This may be because an unexpected movement occurs during braking so make sure this happens before proceeding with caution on these roads when using brake pedal pedals". It can also occur by accident due either to bad driving habits
```
- Speed: `62.0 tok/s` | CoT Triggered: `False` | AST: `N/A`

### Test 4 [Logic]: `odd_numbers`
**Prompt:**
> Explain step by step why the sum of two odd integers is always an even integer.

#### Nool-Alpha-100M (Reasoning SFT):
```text
<think>
Okay, so I have this problem where I need to find all positive integers that are even numbers in a number. Hmm, let me try to figure this out step by step.

First, I remember that if two numbers are odd, there's no remainder. So, if I have a divisor n, then the sum of the primes is less than or equal to the sum of its digits. That means I can use that to compute their product.
```
- Speed: `15.2 tok/s` | CoT Triggered: `True` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
The sum of two odd integers can be calculated using a simple formula:

$$S = 2^{n} \left(\frac{1}{3}\right)^{-k} + S_0$$

where $N$ is the number of elements in the array, $K$ is the index at which to start calculating the sum, and $\left(\frac{1}{3}\right)$ denotes the factorial function (e.g., `factorial(6
```
- Speed: `54.7 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
"The following example demonstrates how to solve a problem that can be solved using multiple solutions."
```
- Speed: `62.1 tok/s` | CoT Triggered: `False` | AST: `N/A`

### Test 5 [Logic]: `apple_riddle`
**Prompt:**
> There are 5 apples in a basket. You take away 3 apples. How many apples do you have? Think carefully.

#### Nool-Alpha-100M (Reasoning SFT):
```text
Let's solve this problem using Python code.
<llm-code>
# number of days
number_of_tables = 3 * 3
# number of people
number_of_bought_in_one_tuesday = 3
# number of students in the day
number_of_bought_per_bought = number_of_seats_in_tables * number_of_bought_for_a_per_day
```
- Speed: `14.5 tok/s` | CoT Triggered: `True` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
The question asks us to find the number of apples that remain after taking out three apples from the basket, which is 2. We can use the concept of "remaining" as follows: if we took away 1 apple from the basket and added it back again, then there would still be one apple remaining. If we took away 4 apples from the basket and added them back together again, then there would still be two apples left over. In this case, we need to figure out
```
- Speed: `59.9 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
"I'm sure there is no way I can be certain that the apple was all chopped up and left on top of my salad."
, "What about your favorite food?" (the last one)
- - - -
```
- Speed: `63.2 tok/s` | CoT Triggered: `False` | AST: `N/A`

### Test 6 [Logic]: `syllogism`
**Prompt:**
> Premise 1: All cats are animals. Premise 2: Some animals are black. Question: Does it logically follow that all cats are black? Explain.

#### Nool-Alpha-100M (Reasoning SFT):
```text
<think>
Okay, so I need to figure out how many ways there are in a lot of ways. Let's start by understanding the problem step by step. 

First, the question is asking if the number of ways to choose the number of ways to arrange them is formed from the other two of them. So, for example, if I have three people, each with a single cell, and they can't be either the number of ways to pick any one or one.
```
- Speed: `13.8 tok/s` | CoT Triggered: `True` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
Here's the premise for a paragraph based on this reasoning:

All cats are animals, and some of them are indeed black. This means that they don't belong to any specific species or category (animals). In fact, most animals have characteristics in common with humans, such as being able to move their limbs independently, having fur, eyes, ears, noses, mouths, etc., which is what we commonly associate with living things like humans. So, if you're a cat who loves chasing
```
- Speed: `60.0 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
(A) No, not at this point in time; only the first sentence of paragraph A is true and correct! This does not mean there can't be some other possibility for a cat to have brown fur or white hair on its body but rather we must ask why so many people believe such things about them because they're false facts.[2] We know from research by Drs. Drexel University's Michael Sperling and Robert Buss [3] who found similar results when studying
```
- Speed: `64.9 tok/s` | CoT Triggered: `False` | AST: `N/A`

### Test 7 [Code]: `is_palindrome`
**Prompt:**
> Write a Python function `is_palindrome(s)` that returns True if a string s is a palindrome and False otherwise.

#### Nool-Alpha-100M (Reasoning SFT):
```text
Sure, I will write a Python function to generate all the characters in a given list of strings. Here's how you can create a function in Python:

```python
def find_array(lst):
    if len(s) == 0:
        return [num for i in range(i+1)]
    return num

# Test the function
n = [num for num in arr if num
```
- Speed: `13.4 tok/s` | CoT Triggered: `False` | AST: `[green]PASS[/green]`

#### SmolLM-135M-Instruct:
```text
To check whether a string is a palindrome, we can use the following steps:

1. Split the string into substrings of equal length using the `split()` method.
2. Check if each substring matches the original string. If it does not match, return False.
3. Return True if all substrings are palindromes.
4. Handle cases where the input string contains spaces or other non-alphanumeric characters as substrings by converting them to
```
- Speed: `56.3 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: inval)[/red]`

#### GPT-2 Small (124M):
```text
- Returns an object of the form (0, 1). If no response was received from this program then return None . - Note: This will not work on Windows due to some limitations in PyPy's library for handling strings with spaces instead than underscores or other special characters as well; see below about "string formatting" above --- See also http://www3djs.org/pypi/docs/#python2#strings , https:/ / www 3DScripting Wiki page #5
```
- Speed: `60.0 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: unter)[/red]`

### Test 8 [Code]: `factorial`
**Prompt:**
> Write a Python function `factorial(n)` that computes the factorial of a positive integer n.

#### Nool-Alpha-100M (Reasoning SFT):
```text
Sure, I will write a Python program that takes an integer n and returns the sum of all prime numbers less than 1 and itself. 

Here's the code to solve this problem:

```python
def is_prime(n):
    if n <= 1:
        return "1"
    for i in range(2, int(n**0.5) + 1):
```
- Speed: `14.3 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: expec)[/red]`

#### SmolLM-135M-Instruct:
```text
To compute the factorial of a positive integer, we can use a recursive approach to calculate it recursively until it reaches 1 or becomes negative. We will start with an initial value of 0 and increment it by one for each digit in the number:
- If the current digit is even (i.e., its digits are not zero), return 1 because there are only two possible values: 1 and 2. This is because when you add 1 to any nonzero
```
- Speed: `59.9 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: inval)[/red]`

#### GPT-2 Small (124M):
```text
, (1), "a number with an odd value" , 1; } ; Example : import math from matplotlib . pygms def sum_of_points () { return np . matrix ([ 2 * 1000000 + 4 - 3 / 10 ]) % 5f # get all points in our range as integers if not p = 0 then print '\t$' else i += int ((p/2)*10 ) end while True do xs[i] <- pow((
```
- Speed: `63.7 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: unmat)[/red]`

### Test 9 [Code]: `find_max`
**Prompt:**
> Write a Python function `find_max(numbers)` that returns the largest number in a list.

#### Nool-Alpha-100M (Reasoning SFT):
```text
```python
def is_prime(n):
    if n == 0:
        return 1
    for i in range(2, n+1):
        if n % i == 0:
            return False
    return n *=0

def sum_sum(n):
    prime_of_
```
- Speed: `13.3 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: inval)[/red]`

#### SmolLM-135M-Instruct:
```text
To find the largest number, we can use a loop to iterate through each element of the list and compare it with the current largest number. If they are equal or within a certain range (e.g., 10-20), then the maximum is found. Otherwise, we return -1.

Here's how you could implement this:

```python
def find_max(*numbers):
    max_num = numbers[0]

    for num in numbers:
```
- Speed: `60.0 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: expec)[/red]`

#### GPT-2 Small (124M):
```text
- The following code snippet is provided for debugging purposes only, and should not be used to evaluate any of these functions as they are described below (for more information see http://www/python2.7.3). - If you use this example with an interpreter running on Windows XP or later systems then it may cause some problems if your program does not support Unicode characters! See https:/ / python.org/#pypi#include "py4". Note This can sometimes result from unexpected behaviour
```
- Speed: `62.8 tok/s` | CoT Triggered: `False` | AST: `[red]FAIL (SyntaxError: inval)[/red]`

### Test 10 [Instruct]: `planets_list`
**Prompt:**
> List 3 distinct planets in our solar system as numbered bullet points with one key feature for each.

#### Nool-Alpha-100M (Reasoning SFT):
```text
To generate a given set of two sets, we can use the following code:

```python
import numpy as np
from numpy import mn

def sum_array(n):
    return n * factorial(n) * factorial(n) + 1

# Calculate the product of two variables
print("Enter the values: ")
print(sum_of_numbers)
</llm-code>
```
- Speed: `14.3 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
Here are three distinct planets in our solar system, numbered accordingly:

1. **Venus** (Mercury) - Mercury is the smallest planet and innermost of the four planets that make up our solar system. It's a hot, rocky world surrounded by an atmosphere composed mostly of carbon dioxide. Despite its small size, Venus has some fascinating features worth highlighting. For example, it's incredibly close to the Sun, making it extremely difficult to survive without light or heat from Earth. Additionally, Venus
```
- Speed: `58.2 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
(click to expand)
```
- Speed: `49.9 tok/s` | CoT Triggered: `False` | AST: `N/A`

### Test 11 [Instruct]: `photosynthesis`
**Prompt:**
> Explain the concept of photosynthesis in two clear sentences.

#### Nool-Alpha-100M (Reasoning SFT):
```text
<think>
Okay, so I need to figure out how many different types of three groups are in a certain case. Hmm, let's start by recalling what a black hole is. From what I remember, when you have two different types, the number of ways to choose one person from the other to each other. So, if I have four options, say "for the same number of ways to be one of them." But wait, maybe there's a way to approach this?
```
- Speed: `14.7 tok/s` | CoT Triggered: `True` | AST: `N/A`

#### SmolLM-135M-Instruct:
```text
Photosynthesis is a fundamental process that occurs within plants and other organisms to convert light energy into chemical energy, which they can then use for growth, development, and survival. Here are two clear sentences explaining photosynthesis:

1. **Light-dependent reactions**: Photosynthesis involves the production of ATP (adenosine triphosphate) from NADPH (nicotinamide adenine dinucleotide phosphate). This reaction is catalyzed by enzymes called RuBisCO, which acts as a catalyst. The
```
- Speed: `60.5 tok/s` | CoT Triggered: `False` | AST: `N/A`

#### GPT-2 Small (124M):
```text
"The idea that a single molecule is responsible for all light, and therefore does not require any other molecules to be present at its action on it (i.) seems very strange."
- Robert Binder
```
- Speed: `65.6 tok/s` | CoT Triggered: `False` | AST: `N/A`

