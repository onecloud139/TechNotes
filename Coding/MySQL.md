# SQL

## 一、基础操作

### 1.1 数据定义 (DDL)

#### 创建表
```sql
CREATE TABLE table_name (
    column1 datatype constraints,
    column2 datatype constraints,
    ...
    table_constraints
);

-- 示例
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100),
    age INT CHECK (age >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 修改表结构
```sql
-- 添加列
ALTER TABLE table_name ADD column_name datatype;

-- 修改列
ALTER TABLE table_name MODIFY column_name new_datatype;
ALTER TABLE table_name ALTER COLUMN column_name SET DATA TYPE new_datatype; -- PostgreSQL/SQL Server

-- 删除列
ALTER TABLE table_name DROP COLUMN column_name;

-- 重命名表
ALTER TABLE old_name RENAME TO new_name;
RENAME TABLE old_name TO new_name;
```

#### 删除对象
```sql
DROP TABLE table_name;      -- 删除表（结构+数据）
TRUNCATE TABLE table_name;  -- 仅清空数据，保留结构
DROP DATABASE database_name; -- 删除数据库
```

### 1.2 数据操作 (DML)

#### 查询数据
```sql
-- 基础查询
SELECT column1, column2 FROM table_name;
SELECT * FROM table_name;

-- 条件与排序
SELECT * FROM table_name WHERE condition;
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;
SELECT DISTINCT column_name FROM table_name;

-- 分页
SELECT * FROM table_name LIMIT 10 OFFSET 20;    -- MySQL/PostgreSQL
SELECT TOP 10 * FROM table_name;                -- SQL Server
SELECT * FROM table_name WHERE ROWNUM <= 10;    -- Oracle

-- 聚合查询
SELECT 
    COUNT(*), AVG(column_name), SUM(column_name),
    MAX(column_name), MIN(column_name)
FROM table_name
GROUP BY column_name
HAVING aggregation_condition;
```

#### 数据变更
```sql
-- 插入
INSERT INTO table_name (column1, column2) VALUES (value1, value2);
INSERT INTO table_name (column1, column2) VALUES (value1, value2), (value3, value4);

-- 从其他表插入
INSERT INTO table_name SELECT column1, column2 FROM other_table WHERE condition;

-- 更新（切记加 WHERE）
UPDATE table_name
SET column1 = value1, column2 = value2
WHERE condition;

-- 删除（切记加 WHERE）
DELETE FROM table_name WHERE condition;
```

## 二、进阶查询

### 2.1 多表操作

#### 连接查询 (JOIN)
```sql
-- 内连接
SELECT * FROM table_a a INNER JOIN table_b b ON a.id = b.a_id;

-- 外连接
SELECT * FROM table_a a LEFT JOIN table_b b ON a.id = b.a_id;
SELECT * FROM table_a a RIGHT JOIN table_b b ON a.id = b.a_id;
SELECT * FROM table_a a FULL OUTER JOIN table_b b ON a.id = b.a_id;

-- 其他连接
SELECT * FROM table_a CROSS JOIN table_b;  -- 笛卡尔积
SELECT a.name, b.name AS manager FROM employees a JOIN employees b ON a.manager_id = b.id; -- 自连接
```

#### 子查询
```sql
-- 标量子查询
SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products);

-- IN / EXISTS
SELECT * FROM customers WHERE id IN (SELECT customer_id FROM orders);
SELECT * FROM departments d WHERE EXISTS (SELECT 1 FROM employees e WHERE e.dept_id = d.id);

-- 关联子查询
SELECT e.name, e.salary, d.avg_salary
FROM employees e
JOIN (SELECT dept_id, AVG(salary) as avg_salary FROM employees GROUP BY dept_id) d
ON e.dept_id = d.dept_id;
```

#### 集合操作
```sql
SELECT column FROM table_a UNION SELECT column FROM table_b;      -- 并集去重
SELECT column FROM table_a UNION ALL SELECT column FROM table_b;  -- 并集保留重复
SELECT column FROM table_a INTERSECT SELECT column FROM table_b;  -- 交集
SELECT column FROM table_a EXCEPT SELECT column FROM table_b;     -- 差集（Oracle 用 MINUS）
```

### 2.2 窗口函数 (Window Functions)

#### 核心概念与语法结构
窗口函数对一组行（称为"窗口"）进行计算，并为每行返回一个结果。与聚合函数不同，窗口函数不会将行合并为单行。

```sql
函数名([参数]) OVER (
    [PARTITION BY 分区列1, 分区列2, ...]  -- 定义窗口分区（类似GROUP BY）
    [ORDER BY 排序列1 [ASC|DESC], ...]     -- 定义窗口内排序
    [帧定义窗口范围]                          -- 定义计算范围（滑动窗口）
) AS alias
```

**执行顺序**：FROM → WHERE → GROUP BY → HAVING → SELECT(窗口函数在此执行) → ORDER BY → LIMIT

#### 帧定义（Frame Specification）详解
控制窗口内计算的行范围，默认是 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`。

```sql
-- 行边界模式
ROWS UNBOUNDED PRECEDING                    -- 从分区第一行到当前行
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- 同上（显式写法）
ROWS BETWEEN 2 PRECEDING AND CURRENT ROW    -- 前2行 + 当前行（共3行）
ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING    -- 当前行 + 后2行（共3行）
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING    -- 前1行 + 当前行 + 后1行（共3行）
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- 整个分区所有行

-- 值范围模式（处理重复值）
RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- 所有值小于等于当前行的行
RANGE BETWEEN INTERVAL '1' DAY PRECEDING AND CURRENT ROW  -- 时间范围（Oracle/PostgreSQL）
```

#### 排名函数（Ranking Functions）
```sql
-- ROW_NUMBER(): 连续唯一编号（1,2,3,4...）
SELECT 
    department,
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as global_rank,  -- 全局排名
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank  -- 部门内排名
FROM employees;

-- RANK(): 排名（跳号，相同值同排名，下一名次跳过）
-- DENSE_RANK(): 密集排名（不跳号）
SELECT 
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) as rank_num,         -- 1,1,3,4,4,6...
    DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank, -- 1,1,2,3,3,4...
    ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num      -- 1,2,3,4,5,6...
FROM employees;

-- PERCENT_RANK(): 相对排名百分比（0到1之间）
-- CUME_DIST(): 累积分布（当前行值小于等于它的行数占总行数的比例）
SELECT 
    name,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary DESC) as pct_rank,  -- (rank-1)/(总行数-1)
    CUME_DIST() OVER (ORDER BY salary DESC) as cum_dist      -- 小于等于当前值的行数/总行数
FROM employees;

-- NTILE(n): 将分区分为n个桶（分位数）
SELECT 
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary DESC) as quartile,   -- 四分位数（1,2,3,4）
    NTILE(10) OVER (ORDER BY salary DESC) as decile     -- 十分位数
FROM employees;
```

#### 偏移函数（Value Functions / Navigation Functions）
```sql
-- LAG/LEAD: 访问前后行的数据
SELECT 
    year,
    month,
    amount,
    -- LAG(列名, 偏移量, 默认值) 默认值在超出范围时返回
    LAG(amount, 1, 0) OVER (ORDER BY year, month) as prev_month,
    LEAD(amount, 1, 0) OVER (ORDER BY year, month) as next_month,
    amount - LAG(amount, 1) OVER (ORDER BY year, month) as diff,
    -- 计算增长率
    (amount - LAG(amount, 1) OVER (ORDER BY year, month)) / 
        NULLIF(LAG(amount, 1) OVER (ORDER BY year, month), 0) * 100 as growth_rate
FROM sales;

-- FIRST_VALUE/LAST_VALUE: 窗口内第一个/最后一个值
-- NTH_VALUE: 窗口内第n个值
SELECT 
    department,
    name,
    salary,
    FIRST_VALUE(name) OVER (PARTITION BY department ORDER BY salary DESC) as highest_paid,
    LAST_VALUE(name) OVER (
        PARTITION BY department 
        ORDER BY salary DESC 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as lowest_paid,
    NTH_VALUE(name, 3) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as third_highest
FROM employees;
```

#### 聚合窗口函数（Aggregate Window Functions）
```sql
-- 累计计算（Running Totals）
SELECT 
    date,
    daily_sales,
    SUM(daily_sales) OVER (ORDER BY date) as cum_sum,  -- 累计销售额
    SUM(daily_sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as week_sum, -- 近7天
    AVG(daily_sales) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as month_avg  -- 近30天平均
FROM sales;

-- 分组聚合占比
SELECT 
    department,
    name,
    salary,
    SUM(salary) OVER (PARTITION BY department) as dept_total,
    salary * 1.0 / SUM(salary) OVER (PARTITION BY department) as dept_ratio,
    AVG(salary) OVER (PARTITION BY department) as dept_avg,
    MAX(salary) OVER (PARTITION BY department) as dept_max,
    MIN(salary) OVER (PARTITION BY department) as dept_min,
    COUNT(*) OVER (PARTITION BY department) as dept_count,
    -- 与部门平均的差值
    salary - AVG(salary) OVER (PARTITION BY department) as diff_from_avg
FROM employees;

-- 全局占比（不指定PARTITION BY）
SELECT 
    product,
    sales,
    SUM(sales) OVER () as total_sales,
    sales * 1.0 / SUM(sales) OVER () as pct_of_total
FROM product_sales;
```

#### 高级实用场景

**场景1：Top-N 问题（每组取前N条）**
```sql
-- 方法1：使用ROW_NUMBER()
SELECT * FROM (
    SELECT 
        department,
        name,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rn
    FROM employees
) t WHERE rn <= 3;

-- 方法2：使用RANK()（处理并列情况）
SELECT * FROM (
    SELECT 
        department,
        name,
        salary,
        RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rk
    FROM employees
) t WHERE rk <= 3;  -- 如果有并列第3，都会包含进来
```

**场景2：去重保留最新记录（保留每个用户最新记录）**
```sql
-- 方法1：保留最新一条
SELECT * FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) as rn
    FROM user_logs
) t WHERE rn = 1;

-- 方法2：标记所有最新记录（处理时间戳相同的情况）
SELECT * FROM (
    SELECT 
        *,
        RANK() OVER (PARTITION BY user_id ORDER BY updated_at DESC) as rk
    FROM user_logs
) t WHERE rk = 1;
```

**场景3：同比环比计算（YoY/MoM）**
```sql
SELECT 
    year,
    month,
    amount,
    -- 环比（与上月比）
    LAG(amount, 1) OVER (ORDER BY year, month) as last_month,
    (amount - LAG(amount, 1) OVER (ORDER BY year, month)) / 
        NULLIF(LAG(amount, 1) OVER (ORDER BY year, month), 0) as mom_growth,
    -- 同比（与上年同月比）
    LAG(amount, 12) OVER (ORDER BY year, month) as last_year,
    (amount - LAG(amount, 12) OVER (ORDER BY year, month)) / 
        NULLIF(LAG(amount, 12) OVER (ORDER BY year, month), 0) as yoy_growth,
    -- 年初至今累计（YTD）
    SUM(amount) OVER (PARTITION BY year ORDER BY month) as ytd_amount
FROM monthly_sales;
```

**场景4：连续区间识别（Gaps and Islands问题）**
```sql
-- 识别连续登录天数
WITH login_with_grp AS (
    SELECT 
        user_id,
        login_date,
        DATE_SUB(login_date, INTERVAL 
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) DAY
        ) as grp
    FROM user_logins
)
SELECT 
    user_id,
    MIN(login_date) as start_date,
    MAX(login_date) as end_date,
    COUNT(*) as consecutive_days
FROM login_with_grp
GROUP BY user_id, grp;
```

**场景5：移动平均与平滑处理**
```sql
-- 3日移动平均（中心对齐）
SELECT 
    date,
    value,
    AVG(value) OVER (
        ORDER BY date 
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) as moving_avg,
    -- 指数移动平均近似（加权）
    AVG(value) OVER (
        ORDER BY date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as weighted_avg
FROM time_series;
```

**场景6：数据分组采样（每组取一条）**
```sql
-- 随机抽取每个类别的一条记录
SELECT * FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY RAND()) as rn
    FROM products
) t WHERE rn = 1;
```

#### 性能优化注意事项
1. **索引策略**：确保 `PARTITION BY` 和 `ORDER BY` 的列有索引
2. **内存使用**：大数据集窗口计算可能消耗大量内存，考虑分批处理
3. **避免嵌套**：多层嵌套窗口函数会显著降低性能
4. **简化帧定义**：默认帧定义通常性能最好，复杂 `RANGE` 定义较慢

### 2.3 条件逻辑 (CASE)

#### 基础语法
```sql
-- 简单 CASE（等值比较）
CASE expression
    WHEN value1 THEN result1
    WHEN value2 THEN result2
    ELSE default_result
END

-- 搜索 CASE（条件判断）
CASE 
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ELSE default_result
END
```

#### 实战应用
```sql
-- 计算列/分类标签
SELECT 
    name,
    score,
    CASE 
        WHEN score >= 90 THEN '优秀'
        WHEN score >= 80 THEN '良好'
        WHEN score >= 60 THEN '及格'
        ELSE '不及格'
    END AS grade
FROM students;

-- 条件聚合（行转列/透视表）
SELECT 
    year,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count,
    SUM(CASE WHEN type = 'income' THEN amount ELSE -amount END) as balance
FROM orders
GROUP BY year;

-- 自定义排序
SELECT * FROM tasks
ORDER BY 
    CASE priority
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        ELSE 3
    END,
    created_at DESC;

-- 批量条件更新
UPDATE products
SET price = CASE 
    WHEN category = 'electronics' THEN price * 0.9
    WHEN category = 'clothing' THEN price * 0.8
    ELSE price * 0.95
END;
```

## 三、常用函数与优化

### 3.1 常用函数

#### 字符串
```sql
CONCAT(str1, str2)          -- 拼接
SUBSTRING(str, start, len)  -- 截取
UPPER(str) / LOWER(str)     -- 大小写
LENGTH(str) / LEN(str)      -- 长度
TRIM(str)                   -- 去空格
REPLACE(str, old, new)      -- 替换
LIKE '%pattern%'            -- 模糊匹配（%通配符）
```

#### 数值
```sql
ROUND(num, decimals)    -- 四舍五入
FLOOR(num)              -- 向下取整
CEIL(num)               -- 向上取整
ABS(num)                -- 绝对值
MOD(a, b)               -- 取模
RAND() / RANDOM()       -- 随机数
```

#### 日期
```sql
NOW() / CURRENT_TIMESTAMP   -- 当前时间
CURDATE() / CURRENT_DATE    -- 当前日期
DATE_FORMAT(date, '%Y-%m')  -- 格式化（MySQL）
TO_CHAR(date, 'YYYY-MM')    -- 格式化（Oracle/PostgreSQL）
DATEDIFF(day, d1, d2)       -- 日期差（SQL Server）
DATE_PART('year', date)     -- 提取部分（PostgreSQL）
```

#### NULL 处理
```sql
COALESCE(val1, val2, ..., default)  -- 返回第一个非 NULL
NULLIF(val1, val2)                  -- 相等返回 NULL
IFNULL(val, default)                -- MySQL
ISNULL(val, default)                -- SQL Server
```

### 3.2 数据类型强制转换（Type Casting）

#### 标准 SQL 语法（CAST）
```sql
-- 通用语法：CAST(表达式 AS 目标类型)
SELECT 
    CAST('123' AS INT) as num,
    CAST(123.456 AS VARCHAR(10)) as str,
    CAST('2024-01-01' AS DATE) as date_val,
    CAST(price AS DECIMAL(10,2)) as formatted_price
FROM products;

-- 在运算中使用
SELECT 
    product_name,
    CAST(price AS DECIMAL(10,2)) * CAST(quantity AS INT) as total
FROM orders;
```

### 3.3 约束与索引

#### 约束定义
```sql
PRIMARY KEY (column)                                    -- 主键
FOREIGN KEY (column) REFERENCES parent_table(parent_col) ON DELETE CASCADE -- 外键
UNIQUE (column)                                         -- 唯一
NOT NULL                                                -- 非空
CHECK (condition)                                       -- 检查约束
DEFAULT value                                           -- 默认值
```

#### 索引操作
```sql
CREATE INDEX idx_name ON table_name(column);
CREATE UNIQUE INDEX idx_name ON table_name(column);
DROP INDEX idx_name ON table_name;
```

#### 组合索引与覆盖索引
```sql
-- 组合索引（最左前缀原则）
CREATE INDEX idx_name ON table_name(col1, col2, col3);

-- 覆盖索引（查询的所有列都在索引中）
CREATE INDEX idx_cover ON table_name(col1, col2) INCLUDE (col3, col4);  -- SQL Server
```

#### 索引优化提示
```sql
-- 强制使用索引（MySQL）
SELECT * FROM table_name USE INDEX (idx_name) WHERE condition;

-- 查看执行计划
EXPLAIN SELECT * FROM table_name WHERE condition;
```
