# Python:从入门到入土

# 基本运算符

## 数学运算

```python
math_score = 85
english_score = 92
total_score = math_score + english_score  # 加法：177
average_score = total_score / 2           # 除法：88.5
half_score = math_score // 2              # 整除：42
remainder = english_score % 10            # 取余：2
square = math_score ** 2  # 85的平方：7225

math_score += 5  # 等同于 math_score = math_score + 5 → 90
#同理，也有-=, /=, **=之类的
```

## 逻辑运算

```sql
attendance_rate = 95.0
pass_threshold = 90.0
is_pass = attendance_rate >= pass_threshold  # True
is_full = attendance_rate == 100.0           # False

has_homework = True
has_exam = False
can_leave = has_homework and (not has_exam)  # True and True → True
can_leave = has_homework or (has_exam)  # True or False → False
```

## 逻辑运算——实用操作

1. **Python 的布尔逻辑**

- Python 中任何对元素都可以在判断真假（如 在 `if` 语句中，你可以写 `if (0)` ）。
- 对象会被转换为 `True` 或 `False`，遵循固定的规则。

1. **假值（Falsy Values）规则**

Python 将以下值视为 `False`：

- `None`
- `False`
- **数值零**​：`0`（整数）、`0.0`（浮点数）、`0j`（复数 0）
- **空序列/集合**​：`""`、`[]`、`()`、`{}`、`set()`、`range(0)`

---

1. **也就是说，你在判定一个元素是否为 0/一个列表是否为空时​**

```
a=[]

if a :
    print("a非空列表")
else:
    print("a是空列表")#a是空列表时，if a判定为假，执行else
```

```
count = 0
if count:  # 等价于 if count != 0
    print("Count is non-zero")
else:
    print("Count is zero")  # 实际输出
```

# 控制流符号

## `if` 条件语句（条件分支）

`if` 语句根据条件表达式的布尔值决定代码执行路径。

### 基本语法：

```
if condition1:
    # 当 condition1 为 True 时执行
elif condition2:
    # 当 condition2 为 True 时执行
else:
    # 当所有条件都为 False 时执行
```

### 关键特性：

- **真值测试**​：Python 使用隐式布尔转换（Truthy/Falsy）

```
if 0:        # False（数值0）
if "":       # False（空字符串）
if []:       # False（空列表）
if None:     # False
if 1:        # True
if "hello":  # True
```

- **链式比较**​：

```
if 18 <= age < 65:  # 检查 age 是否在 18-65 范围内
```

- **三元表达式**​（简化版 if-else）：

```
result = "Even" if num % 2 == 0 else "Odd"
```

## `for` 循环（迭代循环）

`for` 循环用于遍历可迭代对象（iterable）中的元素。（白话：意思就是你能一个一个取值的东西，比如这次取某个值，下次取另一个值，取值思路满足某种规则，称为可迭代）

### 基本语法：

```
for item in iterable:
    # 对每个元素执行的操作
```

### 关键特性：

- **遍历任何可迭代对象**​：

```
# 列表
for fruit in ["apple", "banana", "cherry"]:
    print(fruit)

# 字典
for key, value in {"a": 1, "b": 2}.items():#这里的.items是取出所有key-value对
    print(key, value)
```

- **range() 函数**​：

```
for i in range(5):        # 0-4
for i in range(1, 10, 2): # 1,3,5,7,9
```

- **enumerate() 获取索引**​：

enumerate：你简单理解，就是遍历一个列表，但是第一个值把列表索引也搞出来，第二个值是列表元素值，和字典那个 key-value 很像

```
for index, value in enumerate(["a", "b", "c"]):
    print(index, value)  # (0,'a'), (1,'b'), (2,'c')
```

## `while` 循环（条件循环）

`while` 在条件为真时重复执行代码块。

### 基本语法：

```
while condition:
    # 循环体
```

### 关键特性：

- **无限循环模式**​：

```
while True:
    # 需要配合 break 退出
    if exit_condition:
        break
```

- **循环控制语句**​：

```
while condition:
    if skip_condition:
        continue  # 跳过本次迭代，但继续执行下一次循环
    if exit_condition:
        break     # 退出循环
```

- **else 子句**​：

```
n = 5
while n > 0:
    n -= 1
else:
    print("Loop completed")  # 会执行
```

## Continue and Break:

单独写一条，千万记住 continue 是取消本次循环，但继续下一次，而 break 是直接不循环了。

# 数据类型

Python 的数据类型可以分为两大类：​**基本数据类型**和**容器数据类型**。Python 是动态类型语言，变量类型在运行时确定，无需显式声明。

## 基础数据类型

一般都指数值类型 (Numeric Types)

- **整型 (int)**​：任意大小的整数

```
a = 10       # 正整数
b = -5       # 负整数
c = 0        # 零
d = 100_000  # 使用下划线增强可读性（Python 3.6+）
```

- **浮点型 (float)**​：带小数点的数

```
pi = 3.14159
temperature = -10.5
scientific = 2.5e-4  # 0.00025
```

- **复数 (complex)**​：实部 + 虚部

```
z = 3 + 4j
print(z.real)  # 3.0
print(z.imag)  # 4.0
```

- **布尔型 (bool)**​：True 或 False

```
is_valid = True
is_empty = False
```

## 字符串常用方法

- **字符串 (str)**​：不可变的 Unicode 字符序列
  1. 创建字符串

  ```
  ```

# 单引号

s1 = 'Hello World'

# 双引号

s2 = "Python Programming"

# 三引号（多行字符串）

s3 = """This is a
multi-line
string"""

# 空字符串

empty = ""

```
	1. 字符串特性 

- **不可变性**​：字符串创建后不能修改（一般来说就是不能赋值）

```

s = "hello"
s[0] = "H"  # 报错：TypeError

```

- **序列特性**​：支持索引和切片操作
	```
s = "Python"
print(s[0])    # 'P' - 索引访问
print(s[2:5])  # 'tho' - 切片操作
```

```
1. 基本操作 
```

```
# 连接
s1 = "Hello" + " " + "World"  # "Hello World"

# 重复
s2 = "Ha" * 3  # "HaHaHa"

# 自动连接（相邻字符串字面量）
s3 = "This " "is " "joined"  # "This is joined"
```

检查

```python
s = "apple"
print("a" in s)   # True
print("app" in s) # True
print("ban" in s) # False
```

长度与索引

```
s = "Python"
print(len(s))     # 6
print(s[-1])      # 'n'（负索引从末尾开始）
```

遍历字符串

```
for char in "Hello":
    print(char)
    
# 输出：
# H
# e
# l
# l
# o
```

### 大小写转换

```
s = "Python Programming"
print(s.lower())      # "python programming"
print(s.upper())      # "PYTHON PROGRAMMING"
print(s.title())      # "Python Programming"
print(s.capitalize()) # "Python programming"
print(s.swapcase())   # "pYTHON pROGRAMMING"
```

### 查找与替换

```
s = "Hello World"

# 查找
print(s.find("World"))    # 6（返回索引位置）
print(s.find("Python"))   # -1（未找到）
print(s.index("o"))       # 4（类似find，但未找到会报错）

# 替换
print(s.replace("World", "Python"))  # "Hello Python"
```

### 拆分与连接

```
# 拆分
csv = "apple,banana,orange"
fruits = csv.split(",")  # ['apple', 'banana', 'orange']

# 连接
words = ["Python", "is", "awesome"]
sentence = " ".join(words)  # "Python is awesome"
```

### 去除空白

```
s = "  Hello World  \n"
print(s.strip())      # "Hello World"（去除两端空白）
print(s.lstrip())     # "Hello World  \n"（去除左端空白）
print(s.rstrip())     # "  Hello World"（去除右端空白）
```

### 格式化与对齐

```
s = "Python"

# 对齐
print(s.ljust(10, "-"))  # "Python----"
print(s.rjust(10, "-"))  # "----Python"
print(s.center(10, "-")) # "--Python--"

# 格式化
print("Value: {:>8}".format(42))  # "Value:       42"
```

### 判断方法

```
s = "Python3"
print(s.startswith("Py"))  # True
print(s.endswith("3"))     # True
print(s.isalpha())         # False（包含数字）
print(s.isdigit())         # False
print(s.isalnum())         # True（字母或数字）
print(" \t\n".isspace())   # True
```

## f-string（格式化字符串）

f-string 是 Python 3.6 引入的字符串格式化方法，语法简洁高效，是当前推荐的字符串格式化方式。因此我把它单独拉出来看了（希望你也认真学）

### 基本语法

```
name = "Alice"
age = 30

# 基本用法
print(f"My name is {name} and I'm {age} years old.")
# 输出: My name is Alice and I'm 30 years old.
```

### 表达式支持

```
a = 5
b = 10

# 直接嵌入表达式
print(f"{a} + {b} = {a + b}")  # "5 + 10 = 15"
```

### 格式化选项

```
pi = 3.1415926535

# 浮点数格式化
print(f"Pi: {pi:.2f}")  # "Pi: 3.14"

# 宽度和对齐
print(f"|{name:<10}|")  # |Alice     |（左对齐）
print(f"|{name:>10}|")  # |     Alice|（右对齐）
print(f"|{name:^10}|")  # |  Alice   |（居中）

# 数字格式化
number = 123456789
print(f"{number:,}")   # "123,456,789"（千位分隔符）
print(f"{number:09}")  # "123456789"（前导零填充）
```

### 特殊字符处理

```
# 大括号转义
print(f"{{This is in braces}}")  # "{This is in braces}"

# 日期格式化
from datetime import datetime
now = datetime.now()
print(f"Today is {now:%Y-%m-%d}")  # "Today is 2023-10-18"
```

### 多行 f-string

```
name = "Bob"
age = 25
occupation = "Engineer"

# 多行格式化
message = (
    f"Name: {name}\n"
    f"Age: {age}\n"
    f"Occupation: {occupation}"
)
print(message)
```

### 性能优势

f-string 是编译时求值，相比 `%` 格式化和 `str.format()` 方法有显著性能优势：

```
# 性能比较
%timeit f"Value: {42}"
# 约 0.1 μs

%timeit "Value: {}".format(42)
# 约 0.3 μs

%timeit "Value: %d" % 42
# 约 0.2 μs
```

### 其他字符串格式化方法

虽然 f-string 是首选，但了解其他方法也有价值：

1. % 格式化（旧式）

```
name = "Charlie"
print("Hello, %s!" % name)  # "Hello, Charlie!"
```

2. str.format() 方法

```
print("{} + {} = {}".format(5, 3, 5+3))  # "5 + 3 = 8"
print("{1} before {0}".format("B", "A")) # "A before B"
```

### 字符串高级技巧

1. 原始字符串（忽略转义）

```
path = r"C:\Users\Name\Documents"  # 反斜杠不会被转义
```

2. 字符串模板

```
from string import Template
t = Template("Hello, $name!")
print(t.substitute(name="David"))  # "Hello, David!"
```

## 列表 list

列表（list）是 Python 中最灵活、最常用的数据结构之一。它是一个**有序、可变**的元素集合，可以包含不同类型的元素。下面详细介绍列表的主要操作：

### 列表创建与基础操作

#### 1. 创建列表

```
# 空列表
empty_list = []
empty_list = list()

# 包含元素的列表
numbers = [1, 2, 3, 4, 5]
mixed = [10, "hello", 3.14, True]
nested = [[1, 2], [3, 4], [5, 6]]
```

#### 2. 列表特性

- **有序性**​：元素保持插入顺序
- **可变性**​：可以修改元素
- **异构性**​：可以包含不同类型元素
- **可重复性**​：元素可以重复

### 访问列表元素

#### 1. 索引访问

```
fruits = ["apple", "banana", "cherry"]
print(fruits[0])   # "apple"（正向索引）
print(fruits[-1])  # "cherry"（负向索引）
```

#### 2. 切片操作

```
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:5])    # [2, 3, 4]（索引2到5）
print(numbers[:3])     # [0, 1, 2]（开头到索引3）
print(numbers[6:])     # [6, 7, 8, 9]（索引6到结尾）
print(numbers[::2])    # [0, 2, 4, 6, 8]（步长为2）
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]（反转列表）
```

### 修改列表元素

#### 1. 修改单个元素

```
fruits = ["apple", "banana", "cherry"]
fruits[1] = "blueberry"  # ["apple", "blueberry", "cherry"]
```

#### 2. 修改切片范围

```
numbers = [1, 2, 3, 4, 5]
numbers[1:4] = [20, 30, 40]  # [1, 20, 30, 40, 5]
```

### 添加元素

#### 1. append() - 末尾添加单个元素

```
fruits = ["apple", "banana"]
fruits.append("orange")  # ["apple", "banana", "orange"]
fruits.append(["aaa","bbb"])  # ["apple", "banana", "orange",["aaa","bbb"]]
```

#### 2. extend() - 末尾添加多个元素

```
fruits = ["apple", "banana"]
fruits.extend(["orange", "grape"])  # ["apple", "banana", "orange", "grape"]
```

#### 3. insert() - 指定位置插入元素

```
fruits = ["apple", "banana"]
fruits.insert(1, "mango")  # ["apple", "mango", "banana"]
```

#### 4. 使用 + 运算符

```
list1 = [1, 2]
list2 = [3, 4]
combined = list1 + list2  # [1, 2, 3, 4]
```

### 删除元素

#### 1. remove() - 删除指定值

```
fruits = ["apple", "banana", "cherry"]
fruits.remove("banana")  # ["apple", "cherry"]
```

#### 2. pop() - 删除并返回指定索引元素

```
fruits = ["apple", "banana", "cherry"]
popped = fruits.pop(1)  # popped = "banana", fruits = ["apple", "cherry"]
```

#### 3. del - 删除元素或切片

```
numbers = [0, 1, 2, 3, 4]
del numbers[2]     # [0, 1, 3, 4]（删除索引2）
del numbers[1:3]   # [0, 4]（删除索引1-3）
```

#### 4. clear() - 清空列表

```
fruits = ["apple", "banana", "cherry"]
fruits.clear()  # []
```

### 查找与计数

#### 1. index() - 查找元素索引

```
fruits = ["apple", "banana", "cherry"]
print(fruits.index("banana"))  # 1
```

#### 2. count() - 统计元素出现次数

```
numbers = [1, 2, 3, 2, 4, 2]
print(numbers.count(2))  # 3
```

#### 3. in 运算符 - 检查元素是否存在

```
fruits = ["apple", "banana", "cherry"]
print("banana" in fruits)  # True
```

### 排序与反转

#### 1. sort() - 原地排序

```
numbers = [3, 1, 4, 1, 5, 9, 2]
numbers.sort()  # [1, 1, 2, 3, 4, 5, 9]

# 降序排序
numbers.sort(reverse=True)  # [9, 5, 4, 3, 2, 1, 1]

# 自定义排序
words = ["apple", "Banana", "cherry"]
words.sort(key=str.lower)  # ["apple", "Banana", "cherry"]
```

#### 2. sorted() - 返回新排序列表

```
numbers = [3, 1, 4]
sorted_nums = sorted(numbers)  # [1, 3, 4]，原列表不变
```

#### 3. reverse() - 原地反转

```
fruits = ["apple", "banana", "cherry"]
fruits.reverse()  # ["cherry", "banana", "apple"]
```

#### 4. reversed() - 返回反转迭代器

```
fruits = ["apple", "banana", "cherry"]
reversed_fruits = list(reversed(fruits))  # ["cherry", "banana", "apple"]
```

### 列表遍历

#### 1. 基本遍历

```
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

#### 2. 带索引遍历

```
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
```

#### 3. 列表推导式(其实就是把 for 循环写成一行)

```
# 创建新列表
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件过滤
even_squares = [x**2 for x in range(10) if x % 2 == 0]  # [0, 4, 16, 36, 64]

# 嵌套循环
matrix = [[1, 2], [3, 4], [5, 6]]
flattened = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6]
```

### 其他实用操作

#### 1. 列表长度

```
fruits = ["apple", "banana", "cherry"]
print(len(fruits))  # 3
```

#### 2. 列表比较

```
list1 = [1, 2, 3]
list2 = [1, 2, 4]
print(list1 == list2)  # False（元素不同）
print([1, 2] < [1, 2, 3])  # True（长度不同）
```

#### 3. 列表乘法

```
print([0] * 5)  # [0, 0, 0, 0, 0]
```

## 字典 dict

字典是 Python 中最强大的数据结构之一，它存储键值对（key-value pairs），提供高效的查找和插入操作。字典是无序的（Python 3.7+ 保持插入顺序）、可变的，且键必须是不可变类型。

### 字典创建与基础

1. 创建字典

```
# 空字典
empty_dict = {}
empty_dict = dict()

# 带初始键值对
person = {"name": "Alice", "age": 30}

# 使用dict()构造函数
person = dict(name="Alice", age=30)  # 注意：键此时是字符串且不加引号

# 从键值对序列创建
person = dict([("name", "Alice"), ("age", 30)])

# 字典推导式
squares = {x: x*x for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}
```

1. 字典特性

- **键唯一性**​：键必须是唯一的（重复键会覆盖）
- **键不可变性**​：键必须是不可变类型（字符串、数字、元组等）
- **值任意性**​：值可以是任意类型
- **无序性**​：Python 3.6 之前无序，3.7+ 保持插入顺序

### 访问字典元素

1. 通过键访问

```
person = {"name": "Alice", "age": 30}
print(person["name"])  # "Alice"
```

1. get() 方法（安全访问）

```
print(person.get("age"))       # 30
print(person.get("address"))   # None（不会引发错误）
print(person.get("address", "N/A"))  # "N/A"（提供默认值）
```

1. 检查键是否存在

```
if "name" in person:
    print("Name exists")
```

### 添加/修改元素

1. 添加新键值对

```
person["city"] = "New York"  # {"name": "Alice", "age": 30, "city": "New York"}
```

1. 修改现有值

```
person["age"] = 31  # 更新年龄
```

1. update() 方法（批量更新）

```
person.update({"age": 32, "city": "Boston"})  # 更新多个值
person.update([("job", "Engineer"), ("salary", 80000)])  # 使用键值对序列
```

1. setdefault() 方法

```
# 如果键存在，返回其值；如果不存在，添加键值对并返回值
job = person.setdefault("job", "Developer")  # 添加新键值对
```

### 删除元素

1. del 语句

```
del person["age"]  # 删除键为"age"的项
```

1. pop() 方法

```
age = person.pop("age")  # 删除并返回"age"的值
age = person.pop("age", None)  # 安全删除，提供默认值
```

1. popitem() 方法

```
# 删除并返回最后插入的键值对（Python 3.7+）
key, value = person.popitem()
```

1. clear() 方法

```
person.clear()  # 清空字典 {}
```

### 遍历字典

1. 遍历键

```
for key in person:
    print(key)

for key in person.keys():
    print(key)
```

1. 遍历值

```
for value in person.values():
    print(value)
```

1. 遍历键值对

```
for key, value in person.items():
    print(f"{key}: {value}")
```

### 字典视图对象（三个常用列表）

字典提供三个视图对象，它们会随着字典的变化动态更新：

```
keys = person.keys()      # 键视图
values = person.values()   # 值视图
items = person.items()    # 键值对视图
```

### 字典推导式

```
# 基本形式
squares = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}

# 条件过滤
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}

# 键值互换
inverted = {v: k for k, v in squares.items()}
```

### 字典排序

1. 按键排序

```
sorted_by_key = {k: person[k] for k in sorted(person)}
```

1. 按值排序

```
sorted_by_value = dict(sorted(person.items(), key=lambda item: item[1]))
```

### 字典好习惯

1. **使用 get()避免 KeyError**​：

```
# 不好
if key in my_dict:
    value = my_dict[key]

# 好
value = my_dict.get(key, default_value)
```

1. **使用字典推导式创建字典**​：

```
# 优于循环
squares = {x: x*x for x in range(10)}
```

1. **使用 items()同时访问键值**​：

```
for key, value in my_dict.items():
    # 处理键值对
```

1. **使用 f-string 格式化输出**​：

```
print(f"Name: {person['name']}, Age: {person['age']}")
```

你想了解 Python 中**赋值、浅拷贝、深拷贝**这三种操作针对列表（list）、字典（dict）和集合（set）的核心区别，我会从内存原理到实际案例，用最易懂的方式帮你理清它们的差异。

## 其他数据类型

- **元组 (tuple)**​：有序、不可变、可重复

```
coordinates = (10, 20)
rgb = (255, 0, 128)
single = (42,)  # 单元素元组需要逗号
```

- **范围 (range)**​：不可变的数字序列

```
r = range(5)        # 0,1,2,3,4
r2 = range(1, 10, 2) # 1,3,5,7,9
```

- **集合 (set)**​：无序、可变、不重复

```
unique_nums = {1, 2, 3, 3, 4}  # {1, 2, 3, 4}
unique_nums.add(5)
```

- **冻结集合 (frozenset)**​：不可变集合

```
frozen = frozenset([1, 2, 3])
```

## 数据拷贝

明确三个操作的本质（关键看**内存地址**和**元素引用**）：

- **赋值（=）**：仅给原对象绑定一个新变量名，新旧变量指向**同一个内存地址**，修改任意一个都会影响另一个。
- **浅拷贝**：创建新的容器对象，但容器内的元素仍指向原对象的引用（仅“顶层”独立，嵌套可变元素共享）。
- **深拷贝**：创建全新的容器对象，且递归复制所有嵌套的元素（新旧对象完全独立，无任何共享引用）。

### 赋值（=）：共享同一个对象

赋值操作不会创建新对象，只是给原对象加了一个“别名”。无论修改原对象还是新变量，都会互相影响。

```python
# 1. 列表（含嵌套可变元素）
a_list = [1, 2, [3, 4]]
b_list = a_list  # 赋值
b_list[0] = 100   # 修改顶层元素
b_list[2][0] = 300  # 修改嵌套可变元素
print("原列表a_list:", a_list)  # [100, 2, [300, 4]]（被修改）
print("新列表b_list:", b_list)  # [100, 2, [300, 4]]

# 2. 字典（含嵌套可变值）
a_dict = {"name": "Tom", "scores": [90, 80]}
b_dict = a_dict  # 赋值
b_dict["name"] = "Jerry"  # 修改顶层值
b_dict["scores"][0] = 95  # 修改嵌套可变值
print("原字典a_dict:", a_dict)  # {'name': 'Jerry', 'scores': [95, 80]}（被修改）
print("新字典b_dict:", b_dict)  # {'name': 'Jerry', 'scores': [95, 80]}

# 3. 集合（无嵌套可变元素，set元素必须可哈希）
a_set = {1, 2, 3}
b_set = a_set  # 赋值
b_set.add(4)   # 修改集合
print("原集合a_set:", a_set)  # {1, 2, 3, 4}（被修改）
print("新集合b_set:", b_set)  # {1, 2, 3, 4}
```

### 浅拷贝：顶层独立，嵌套共享

浅拷贝会创建新的容器对象，但容器内的**嵌套可变元素**仍与原对象共享引用（不可变元素如 int/str 修改时会重新赋值，不影响原对象）。

<table>
<tr>
<td>数据类型<br/></td><td>浅拷贝方法<br/></td></tr>
<tr>
<td>list<br/></td><td>`list.copy()`、`list(原列表)`、`原列表[:]`<br/></td></tr>
<tr>
<td>dict<br/></td><td>`dict.copy()`、`dict(原字典)`<br/></td></tr>
<tr>
<td>set<br/></td><td>`set.copy()`、`set(原集合)`<br/></td></tr>
</table>

```python
# 1. 列表浅拷贝（嵌套可变元素共享）
a_list = [1, 2, [3, 4]]
b_list = a_list.copy()  # 浅拷贝
b_list[0] = 100         # 修改顶层不可变元素（不影响原列表）
b_list[2][0] = 300      # 修改嵌套可变元素（影响原列表）
print("原列表a_list:", a_list)  # [1, 2, [300, 4]]
print("新列表b_list:", b_list)  # [100, 2, [300, 4]]

# 2. 字典浅拷贝（嵌套可变值共享）
a_dict = {"name": "Tom", "scores": [90, 80]}
b_dict = a_dict.copy()  # 浅拷贝
b_dict["name"] = "Jerry"  # 修改顶层不可变值（不影响原字典）
b_dict["scores"][0] = 95  # 修改嵌套可变值（影响原字典）
print("原字典a_dict:", a_dict)  # {'name': 'Tom', 'scores': [95, 80]}
print("新字典b_dict:", b_dict)  # {'name': 'Jerry', 'scores': [95, 80]}

# 3. 集合浅拷贝（无嵌套可变元素，表现为完全独立）
a_set = {1, 2, 3}
b_set = a_set.copy()  # 浅拷贝
b_set.add(4)          # 修改集合（不影响原集合）
print("原集合a_set:", a_set)  # {1, 2, 3}
print("新集合b_set:", b_set)  # {1, 2, 3, 4}
```

### 深拷贝：完全独立的新对象

深拷贝需要导入 `copy` 模块的 `deepcopy` 函数，它会递归复制原对象的所有层级（包括嵌套的可变元素），新旧对象完全独立，修改互不影响。

```python
import copy

# 1. 列表深拷贝（完全独立）
a_list = [1, 2, [3, 4]]
b_list = copy.deepcopy(a_list)  # 深拷贝
b_list[0] = 100
b_list[2][0] = 300
print("原列表a_list:", a_list)  # [1, 2, [3, 4]]（无任何影响）
print("新列表b_list:", b_list)  # [100, 2, [300, 4]]

# 2. 字典深拷贝（完全独立）
a_dict = {"name": "Tom", "scores": [90, 80]}
b_dict = copy.deepcopy(a_dict)  # 深拷贝
b_dict["name"] = "Jerry"
b_dict["scores"][0] = 95
print("原字典a_dict:", a_dict)  # {'name': 'Tom', 'scores': [90, 80]}（无任何影响）
print("新字典b_dict:", b_dict)  # {'name': 'Jerry', 'scores': [95, 80]}

# 3. 集合深拷贝（完全独立）
a_set = {1, 2, 3}
b_set = copy.deepcopy(a_set)  # 深拷贝
b_set.add(4)
print("原集合a_set:", a_set)  # {1, 2, 3}（无任何影响）
print("新集合b_set:", b_set)  # {1, 2, 3, 4}
```

### 总结

1. **赋值（=）**：新旧变量指向同一内存地址，修改任意一个都会影响另一个，无“拷贝”行为。
2. **浅拷贝**：仅创建顶层新容器，嵌套可变元素仍共享引用，仅适用于无嵌套可变元素的对象。
3. **深拷贝**：递归复制所有层级的元素，新旧对象完全独立，适合含嵌套可变元素的复杂对象（需导入 `copy` 模块）。

## 类型检查与转换

### 类型检查

```
type(10)          # <class 'int'>
isinstance(3.14, float)  # True
```

### 类型转换

```
int("10")       # 10
float(5)        # 5.0
str(100)        # "100"
list("abc")     # ['a', 'b', 'c']
tuple([1,2,3])  # (1,2,3)
set([1,1,2])    # {1,2}
```

# 类（Class）

类是 Python 面向对象编程（OOP）的基础概念，它允许开发者创建自定义的数据类型，封装数据和行为。类是创建对象的蓝图，定义了对象的属性和方法。

## 通俗解释

在 Python 中，​**类（Class）​** 就像是一个 **班级的蓝图**，而 **对象（Object）​** 则是根据这个蓝图创建出来的 **具体的学生**。让我们用班级的概念来类比：

### **类 = 班级的模板（蓝图）​**

- **类** 定义了共同的特征和行为规则，就像学校制定的 **《班级管理手册》**​：
  - **属性**​：手册里规定了班级应有的特征（如班级名称、班主任、教室位置）。
  - **方法**​：手册里还规定了班级能做什么（如上课、考试、开班会）。

```
# 定义一个"班级"的模板（类）
class Classroom:
    # 初始化方法：创建班级时必须有的属性（像新生入学登记）
    def __init__(self, name, teacher, room):
        self.class_name = name    # 班级名称（属性）
        self.teacher = teacher    # 班主任（属性）
        self.room = room          # 教室位置（属性）
        self.students = []        # 学生名单（属性，初始为空）

    # 方法：添加学生（像班长录入新同学）
    def add_student(self, student_name):
        self.students.append(student_name)
    
    # 方法：举行考试（班级的共同行为）
    def hold_exam(self):
        print(f"{self.class_name}正在考试！")
```

---

### **对象 = 具体的班级**

- 根据模板创建 **具体的班级**​（对象），每个班级有自己独立的数据：
  - `class_301` 和 `class_302` 是两个不同的对象，就像学校里真实的两个班级。

```
# 创建两个具体的班级（对象）
class_301 = Classroom("三年级一班", "张老师", "教学楼301")
class_302 = Classroom("三年级二班", "李老师", "教学楼302")

# 为班级添加学生（调用方法）
class_301.add_student("小明")
class_301.add_student("小红")
class_302.add_student("小刚")

# 每个班级独立行动
class_301.hold_exam()  # 输出: "三年级一班正在考试！"
class_302.hold_exam()  # 输出: "三年级二班正在考试！"
```

## 类的基本概念

1. 类与对象的关系

- **类（Class）​**​：对象的模板或蓝图
- **对象（Object）​**​：类的具体实例
- **属性（Attributes）​**​：对象的状态（数据，比如班里有多少张桌子，多少板凳）
- **方法（Methods）​**​：对象的行为（函数，比如开班会）

```
# 类定义
class Dog:
    pass

# 创建对象（实例化）
my_dog = Dog()
```

1. 为什么使用类？

- **封装**​：将数据和操作数据的方法绑定在一起
- **继承**​：创建层次化的类关系，实现代码重用
- **多态**​：不同类的对象对同一消息做出不同响应
- **抽象**​：隐藏复杂实现细节，提供简单接口

## 定义类的基本结构

1. 最简单的类

```
class EmptyClass:
    pass
```

1. 包含属性和方法的类

特别注意，def __init__(self, name, age)是每个类必须有的东西（还是一个函数，name 和 age 都是在外面给定的）

```
class Dog:
    # 类属性（所有实例共享，也就是说，由这个类产生的所有东西都能输出"Canis familiaris"）
    species = "Canis familiaris"
    
    # 初始化方法（构造函数）
    def __init__(self, name, age):
        # 实例属性（每个对象独有）
        self.name = name
        self.age = age
    
    # 实例方法
    def bark(self):
        return f"{self.name} says Woof!"
    
    def description(self):
        return f"{self.name} is {self.age} years old"
```

## 创建和使用对象

1. 实例化对象

你看，这里 Dog("Buddy", 5)，Dog 实际上就是你之前定义好的"类"，当你用 Dog()这样调用时，就生成了一个具体的 Dog,输入参数第一个被赋值为 self.name，名叫"Buddy"

```
my_dog = Dog("Buddy", 5)
your_dog = Dog("Lucy", 3)
```

1. 访问属性

```
print(my_dog.name)        # "Buddy"
print(my_dog.age)         # 5
print(my_dog.species)     # "Canis familiaris"
```

1. 调用方法

```
print(my_dog.bark())      # "Buddy says Woof!"
print(your_dog.description()) # "Lucy is 3 years old"
```

## 类的重要概念详解

1. `init` 方法（构造函数）

- 在创建新实例时自动调用
- 用于初始化对象属性
- `self` 参数代表当前实例（约定名称，可改但强烈建议保留）

```
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.created_at = datetime.now()  # 设置创建时间
```

1. 实例方法

- 第一个参数总是 `self`（表示当前对象）
- 可以访问和修改对象属性
- 可以调用其他方法

```
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius ** 2
    
    def circumference(self):
        return 2 * 3.14 * self.radius
```

1. 类属性 vs 实例属性

```
class Car:
    # 类属性
    wheels = 4
    
    def __init__(self, brand):
        # 实例属性
        self.brand = brand

# 使用
car1 = Car("Toyota")
car2 = Car("Honda")

print(Car.wheels)  # 4
print(car1.wheels) # 4

# 修改类属性
Car.wheels = 6
print(car2.wheels) # 6（所有实例都受影响）

# 修改实例属性
car1.wheels = 4    # 创建了实例属性
print(car1.wheels) # 4
print(car2.wheels) # 6（类属性不变）
```

## 继承：代码重用的核心机制

1. 基本继承

```
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement this method")

class **Dog(Animal)**:  # 继承Animal类
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# 使用
dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  # "Buddy says Woof!"
print(cat.speak())  # "Whiskers says Meow!"
```

1. 方法重写（Override）

```
class Vehicle:
    def __init__(self, brand):
        self.brand = brand
    
    def drive(self):
        return "Driving a generic vehicle"

class Car(Vehicle):
    def drive(self):  # 重写父类方法，因为之前已经有个drive了
        return f"Driving a {self.brand} car"
```

1. 使用 `super()` 调用父类方法

```
class ElectricCar(Car):
    def __init__(self, brand, battery_size):
        super().__init__(brand)  # 调用父类构造函数
        self.battery_size = battery_size
    
    def drive(self):
        return super().drive() + " with electric power"
```

## 特殊方法（可选，了解即可）

以双下划线开头和结尾的方法，提供特殊功能：

1. `str` 和 `repr`

```
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name} ({self.age} years old)"
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

p = Person("Alice", 30)
print(p)        # Alice (30 years old) - 调用 __str__
print(repr(p))  # Person('Alice', 30) - 调用 __repr__
```

1. 比较运算符

```
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)
```

1. 数学运算符

```
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
```

## 高级类特性

1. 多重继承

```
class Camera:
    def take_photo(self):
        return "Taking photo"

class Phone:
    def make_call(self):
        return "Making call"

class SmartPhone(Camera, Phone):
    pass

iphone = SmartPhone()
print(iphone.take_photo())  # "Taking photo"
print(iphone.make_call())   # "Making call"
```

1. 数据类（Python 3.7+）

```
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0  # 默认值

    def distance(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

# 自动生成 __init__, __repr__, __eq__ 等方法
p = Point(1.0, 2.0)
print(p)  # Point(x=1.0, y=2.0, z=0.0)
```

## 类的好习惯

1. **遵循命名约定**​：

   - 类名：大驼峰式（`MyClass`）
   - 方法名：小写加下划线（`my_method`）
   - 私有成员：双下划线开头（`__private`）
2. **使用属性而不是直接访问字段**​：

```
# 不好
class Person:
    def __init__(self, age):
        self.age = age

# 好
class Person:
    def __init__(self, age):
        self._age = age

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value
```

1. **优先使用组合而不是继承**​：

```
# 使用组合
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()

    def start(self):
        return self.engine.start()

# 而不是继承
class Car(Engine):  # 不自然的关系
    pass
```

1. **使用类型提示（Python 3.5+）​**​：

```
class Vector:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)
```

1. **文档字符串（Docstrings）​**​：

```
class Calculator:
    """A simple calculator class"""

    def add(self, a: float, b: float) -> float:
        """
        Add two numbers

        :param a: first number
        :param b: second number
        :return: sum of a and b
        """
        return a + b
```
