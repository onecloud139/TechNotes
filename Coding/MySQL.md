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

#### 基本语法
```sql
函数名() OVER (
    [PARTITION BY 分组列]
    [ORDER BY 排序列]
    [ROWS/RANGE BETWEEN 窗口范围]
)
```

#### 排名函数
```sql
-- 行号（连续不重复）
SELECT 
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as rn
FROM employees;

-- 排名（跳号，相同值同排名）
SELECT 
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) as rk,
    DENSE_RANK() OVER (ORDER BY salary DESC) as drk  -- 密集排名（不跳号）
FROM employees;

-- 分组排名（每个部门内排名）
SELECT 
    department,
    name,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rn
FROM employees;
```

#### 偏移函数（取值）
```sql
-- 前后行取值
SELECT 
    name,
    salary,
    LAG(salary, 1) OVER (ORDER BY id) as prev_salary,    -- 上一行
    LEAD(salary, 1) OVER (ORDER BY id) as next_salary,   -- 下一行
    salary - LAG(salary, 1) OVER (ORDER BY id) as diff   -- 差值计算
FROM employees;

-- 首尾取值
SELECT 
    name,
    salary,
    FIRST_VALUE(salary) OVER (ORDER BY salary) as lowest,
    LAST_VALUE(salary) OVER (ORDER BY salary ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as highest
FROM employees;
```

#### 聚合窗口函数（累计计算）
```sql
-- 累计求和/平均
SELECT 
    year,
    month,
    amount,
    SUM(amount) OVER (ORDER BY year, month) as cum_sum,           -- 累计收入
    AVG(amount) OVER (ORDER BY year, month ROWS 6 PRECEDING) as moving_avg  -- 移动平均（近7个月）
FROM sales;

-- 分组聚合（每组的总计/平均）
SELECT 
    department,
    name,
    salary,
    SUM(salary) OVER (PARTITION BY department) as dept_total,     -- 部门总工资
    salary / SUM(salary) OVER (PARTITION BY department) as ratio,   -- 占比
    AVG(salary) OVER (PARTITION BY department) as dept_avg          -- 部门平均
FROM employees;
```

#### 窗口边界控制
```sql
-- 行范围控制
ROWS UNBOUNDED PRECEDING          -- 从分区第一行到当前行
ROWS BETWEEN 2 PRECEDING AND CURRENT ROW  -- 前2行到当前行（共3行）
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING  -- 当前行到最后一行
RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW -- 值范围（相同值视为一行）
```

#### 实用场景示例
```sql
-- 场景1：Top-N 问题（每组取前N条）
SELECT * FROM (
    SELECT 
        department,
        name,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rn
    FROM employees
) t WHERE rn <= 3;  -- 每个部门工资前3名

-- 场景2：同比环比（与上月/上年比较）
SELECT 
    year,
    month,
    amount,
    LAG(amount, 1) OVER (ORDER BY year, month) as last_month,
    LAG(amount, 12) OVER (ORDER BY year, month) as last_year,
    (amount - LAG(amount, 1) OVER (ORDER BY year, month)) / LAG(amount, 1) OVER (ORDER BY year, month) as mom_growth
FROM monthly_sales;

-- 场景3：去重保留最新（保留每个用户最新记录）
SELECT * FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) as rn
    FROM user_logs
) t WHERE rn = 1;
```


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

### 3.2 约束与索引

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
