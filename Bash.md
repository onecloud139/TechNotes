

# Bash 基础

## 1. 第一个脚本：创建与执行  
使用文本编辑器（如 `nano` 或 `vim`）创建一个名为 `hello.sh` 的文件：
```bash
nano hello.sh
```
在文件开头添加 **Shebang**（指定解释器），然后写命令：
```bash
#!/bin/bash

# 这是我的第一个 Bash 脚本
echo "Hello, World!"
```
保存文件后，在终端中运行以下命令使其可执行：
```bash
chmod +x hello.sh
```
```bash
./hello.sh
```
**输出：**
```text
Hello, World!
```

## 2. 注释 (Comments)
注释用于解释代码，不会被执行。

*   **单行注释：** 以 `#` 开头。
*   **多行注释：** 通常使用 `: '` 和 `'` 包裹（利用空命令 `:` 的 Here Document 特性）。

```bash
#!/bin/bash

# 这是一行单行注释

: '
这是多行注释的第一行
这是多行注释的第二行
'
```

## 3. 变量 (Variables)

### 3.1定义与使用变量
*   **定义：** `变量名=值`（注意：等号两边**不能有空格**）。
*   **使用：** 在变量名前加 `$`。

```bash
#!/bin/bash

# 定义变量
name="Alice"
age=25

# 使用变量
echo "名字: $name"
echo "年龄: $age"
```

### 3.2 只读变量
使用 `readonly` 命令防止变量被修改。
```bash
#!/bin/bash

readonly pi=3.14159
echo "圆周率是: $pi"

# pi=3  # 尝试修改会报错
```

### 3.3 环境变量
Bash 预设了一些环境变量（通常全大写）。
```bash
#!/bin/bash

echo "当前用户: $USER"
echo "当前目录: $PWD"
echo "Home目录: $HOME"
```

## 4. 命令行参数 (Command Line Arguments)
运行脚本时传入的参数可以通过特殊变量获取。

*   `$0`: 脚本本身的文件名
*   `$1`: 第一个参数
*   `$2`: 第二个参数...
*   `$#`: 参数的总个数
*   `$@`: 所有参数的列表

```bash
#!/bin/bash

echo "脚本名称: $0"
echo "第一个参数: $1"
echo "第二个参数: $2"
echo "总共有 $# 个参数"
echo "所有参数列表: $@"
```

**运行示例：**
```bash
./hello.sh apple banana
```
**输出：**
```text
脚本名称: ./hello.sh
第一个参数: apple
第二个参数: banana
总共有 2 个参数
所有参数列表: apple banana
```

## 5. 条件判断 (Conditional Statements)

### 5.1 基本语法结构
```bash
if [ 条件 ]; then
    # 条件成立时执行
elif [ 条件 ]; then
    # 其他条件
else
    # 都不成立时执行
fi
```
**注意：** `[` 后面和 `]` 前面必须有空格。

### 5.2 常用比较运算符

#### 数值比较
*   `-eq`: 等于 (equal)
*   `-ne`: 不等于 (not equal)
*   `-gt`: 大于 (greater than)
*   `-lt`: 小于 (less than)
*   `-ge`: 大于等于
*   `-le`: 小于等于

#### 字符串比较
*   `=`: 相等
*   `!=`: 不相等
*   `-z`: 字符串长度为 0

#### 文件测试
*   `-e 文件`: 文件是否存在
*   `-d 文件`: 是否是目录
*   `-f 文件`: 是否是普通文件

### 5.3 示例
```bash
#!/bin/bash

read -p "请输入一个数字: " num

if [ $num -gt 10 ]; then
    echo "这个数字大于 10"
elif [ $num -eq 10 ]; then
    echo "这个数字等于 10"
else
    echo "这个数字小于 10"
fi
```

## 6. 循环 (Loops)

### 6.1 for 循环
```bash
#!/bin/bash

# 遍历列表
for fruit in apple banana orange; do
    echo "我喜欢吃: $fruit"
done

# C语言风格 (遍历数字)
for (( i=1; i<=5; i++ )); do
    echo "循环次数: $i"
done
```

### 6.2 while 循环
条件为真时循环。
```bash
#!/bin/bash

count=1
while [ $count -le 5 ]; do
    echo "While 循环: $count"
    count=$((count + 1)) # 变量自增
done
```

### 6.3 until 循环
条件为假时循环（直到条件为真就停止）。
```bash
#!/bin/bash

count=1
until [ $count -gt 5 ]; do
    echo "Until 循环: $count"
    count=$((count + 1))
done
```


## 7. 函数 (Functions)
将代码封装以便重复使用。

```bash
#!/bin/bash

# 定义函数
greet() {
    echo "你好, $1!"
}

# 调用函数
greet "张三"
greet "李四"
```

# Bash 进阶

## 1. 字符串处理 (String Manipulation)
### 1.1 获取字符串长度
```bash
#!/bin/bash
str="hello world"
echo "字符串长度: ${#str}"  # 输出 11
```

### 1.2 字符串切片 (Substring)
语法：`${变量名:起始位置:长度}`
*注意：起始位置从 0 开始。*

```bash
#!/bin/bash
str="abcdefghijk"

echo ${str:2}    # 从索引2开始截取到最后: cdefghijk
echo ${str:2:3}  # 从索引2开始，截取3个字符: cde
echo ${str: -4}  # 从右边数第4个开始 (注意冒号后有空格): hijk
```

### 1.3 字符串替换
```bash
#!/bin/bash
path="/home/user/docs/user.txt"

# 1. 替换第一个匹配项
echo ${path/user/alice}  # 输出: /home/alice/docs/user.txt

# 2. 替换所有匹配项 (双斜杠)
echo ${path//user/alice} # 输出: /home/alice/docs/alice.txt
```

### 1.4 字符串删除 (掐头去尾)
这是处理文件名和路径最常用的技巧。
*   `#`: 从开头删除 (最短匹配)
*   `##`: 从开头删除 (最长匹配)
*   `%`: 从结尾删除 (最短匹配)
*   `%%`: 从结尾删除 (最长匹配)

```bash
#!/bin/bash
file="photo_vacation_2023.jpg.tar.gz"

# 假设我们要去掉后缀
echo "原始文件: $file"

# # 号：去掉开头匹配到第一个 '_' 的部分
echo "去掉开头(最短): ${file#*_}"  # vacation_2023.jpg.tar.gz

# ## 号：去掉开头匹配到最后一个 '_' 的部分
echo "去掉开头(最长): ${file##*_}" # 2023.jpg.tar.gz

# % 号：去掉结尾匹配到第一个 '.' 的部分
echo "去掉后缀(最短): ${file%.*}"  # photo_vacation_2023.jpg.tar

# %% 号：去掉结尾匹配到最后一个 '.' 的部分
echo "去掉后缀(最长): ${file%%.*}" # photo_vacation
```

## 2. 数组 (Arrays)
Bash 支持一维数组。

### 2.1 定义数组
```bash
#!/bin/bash

# 方式一：逐个赋值
fruits[0]="apple"
fruits[1]="banana"
fruits[2]="orange"

# 方式二：一次性赋值 (推荐)
colors=("red" "green" "blue")
```

### 2.2 访问数组元素
```bash
#!/bin/bash
colors=("red" "green" "blue" "yellow")

echo ${colors[0]}  # 访问第一个元素: red
echo ${colors[@]}  # 访问所有元素: red green blue yellow
echo ${#colors[@]} # 获取数组长度: 4
echo ${!colors[@]} # 获取所有索引: 0 1 2 3
```

### 2.3 遍历数组
```bash
#!/bin/bash
colors=("red" "green" "blue")

echo "--- 方法一：遍历值 ---"
for color in "${colors[@]}"; do
    echo "颜色: $color"
done

echo "--- 方法二：遍历索引 ---"
for i in "${!colors[@]}"; do
    echo "索引 $i 的值是: ${colors[$i]}"
done
```

### 2.4 数组切片与追加
```bash
#!/bin/bash
arr=(a b c d e f)

# 切片：${数组[@]:起始位置:个数}
echo ${arr[@]:2:3} # 输出: c d e

# 追加元素
arr+=("g" "h")
echo "追加后: ${arr[@]}"
```

## 3. 数值计算 (Arithmetic)
Bash 不能直接进行小数运算，但整数运算很方便。

### 3.1 使用 $((...)) 语法 (最推荐)
这是最标准的写法。
```bash
#!/bin/bash
a=10
b=3

echo "加法: $((a + b))"  # 13
echo "减法: $((a - b))"  # 7
echo "乘法: $((a * b))"  # 30
echo "除法: $((a / b))"  # 3 (整数除法，直接取整)
echo "取余: $((a % b))"  # 1
echo "幂指: $((a ** 2))" # 100 (a的平方)

# 变量自增/自减
((a++))
echo "a自增后: $a" # 11
```

### 3.2 逻辑运算
在 `((...))` 里可以像写 C 语言一样写逻辑判断。
```bash
#!/bin/bash
x=5
y=10

# 比较大小 (成立返回1，不成立返回0)
echo $((x > y))  # 0 (False)
echo $((x < y))  # 1 (True)

# 配合 if 使用 (注意这里不需要方括号 [])
if ((x < y && y == 10)); then
    echo "条件成立"
fi
```

## 4. 重定向与管道 (Redirection & Pipes)
这是 Linux 脚本的精髓，用于控制输入输出流。

### 4.1 标准文件描述符
Linux 下一切皆文件，进程默认打开三个文件流：
*   **0 (stdin)**: 标准输入
*   **1 (stdout)**: 标准输出
*   **2 (stderr)**: 标准错误

### 4.2 输出重定向 (写入文件)
```bash
# 1. 覆盖写入 (>)
# 将命令的正常输出写入 file.txt (如果文件存在则清空原内容)
echo "Hello" > output.txt

# 2. 追加写入 (>>)
# 将内容追加到文件末尾
echo "World" >> output.txt

# 3. 重定向错误输出 (2>)
# ls 一个不存在的文件，错误信息会被写入 err.log
ls non_existent_file 2> err.log

# 4. 正常输出和错误输出分别写入
# 1> 可以简写为 >
command > stdout.log 2> stderr.log

# 5. 将所有输出都写入同一个文件 (最常用)
# &> 代表 1和2 都重定向
command &> all_output.log
```

### 4.3 输入重定向 (读取文件)
```bash
# 从文件读取内容作为命令的输入
# 例如，统计行数
wc -l < output.txt
```

### 4.4 管道符 (Pipe |)
将前一个命令的**输出**作为后一个命令的**输入**。
```bash
#!/bin/bash

# 示例1：列出当前目录文件，筛选包含 .sh 的文件
ls -l | grep ".sh"

# 示例2：统计有多少个 .sh 文件
ls -l | grep ".sh" | wc -l

# 示例3：排序并去重
echo -e "2\n1\n3\n2" | sort | uniq
```

### 4.5 特殊文件：/dev/null
这是一个黑洞文件，写进去的东西都会消失。如果你不想看到任何输出，就把它丢到这里。
```bash
# 不管命令执行成功还是失败，我都不要看输出
command &> /dev/null

# 只丢弃错误，保留正常输出
command 2> /dev/null
```

## 5. 小练习
写一个脚本，分析当前目录下的日志文件（假设叫 `server.log`），统计错误行数并保存结果。

```bash
#!/bin/bash

# 1. 定义文件名和数组
log_file="server.log"
report_file="analysis_report.txt"
keywords=("ERROR" "WARNING" "FATAL")

# 2. 检查日志文件是否存在
if [ ! -f "$log_file" ]; then
    echo "错误: 文件 $log_file 不存在，正在创建模拟数据..."
    # 生成一点模拟数据
    echo "2023-10-01 INFO: Server started" > $log_file
    echo "2023-10-01 ERROR: Disk full" >> $log_file
    echo "2023-10-01 WARNING: Low memory" >> $log_file
    echo "2023-10-01 ERROR: Connection timed out" >> $log_file
fi

# 3. 清空旧报告并写入新标题
echo "=== 日志分析报告 ===" > $report_file
echo "生成时间: $(date)" >> $report_file
echo "---------------------" >> $report_file

# 4. 循环统计 (使用数组和管道)
for word in "${keywords[@]}"; do
    # grep -c 用于统计匹配行数
    count=$(grep -c "$word" "$log_file")
    echo "$word 出现次数: $count" >> $report_file
done

echo "分析完成！请查看 $report_file"
cat $report_file
```


