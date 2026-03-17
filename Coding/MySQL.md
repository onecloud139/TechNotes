
## 1. 数据定义语言 (DDL)

### 创建表
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

### 修改表结构
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
-- 或
RENAME TABLE old_name TO new_name;
```

### 删除表/数据库
```sql
-- 删除表（结构+数据）
DROP TABLE table_name;

-- 仅清空数据，保留结构
TRUNCATE TABLE table_name;

-- 删除数据库
DROP DATABASE database_name;
```

---

## 2. 数据操作语言 (DML)

### 查询数据 (SELECT)
```sql
-- 基础查询
SELECT column1, column2 FROM table_name;

-- 查询所有列
SELECT * FROM table_name;

-- 条件查询
SELECT * FROM table_name WHERE condition;

-- 排序
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;

-- 去重
SELECT DISTINCT column_name FROM table_name;

-- 分页 (MySQL/PostgreSQL)
SELECT * FROM table_name LIMIT 10 OFFSET 20;
-- SQL Server
SELECT TOP 10 * FROM table_name;
-- Oracle
SELECT * FROM table_name WHERE ROWNUM <= 10;

-- 聚合函数
SELECT 
    COUNT(*),
    AVG(column_name),
    SUM(column_name),
    MAX(column_name),
    MIN(column_name)
FROM table_name
GROUP BY column_name
HAVING aggregation_condition;
```

### 插入数据 (INSERT)
```sql
-- 插入单行
INSERT INTO table_name (column1, column2) 
VALUES (value1, value2);

-- 插入多行
INSERT INTO table_name (column1, column2) 
VALUES 
    (value1, value2),
    (value3, value4),
    (value5, value6);

-- 从其他表插入
INSERT INTO table_name (column1, column2)
SELECT column1, column2 FROM other_table WHERE condition;
```

### 更新数据 (UPDATE)
```sql
UPDATE table_name
SET column1 = value1, column2 = value2
WHERE condition;  -- 切记加 WHERE，否则全表更新
```

### 删除数据 (DELETE)
```sql
DELETE FROM table_name
WHERE condition;  -- 切记加 WHERE，否则清空全表
```

---

## 3. 数据控制语言 (DCL)

```sql
-- 授权
GRANT SELECT, INSERT ON table_name TO user_name;
GRANT ALL PRIVILEGES ON database_name.* TO 'user'@'host';

-- 撤销权限
REVOKE SELECT ON table_name FROM user_name;

-- 拒绝权限 (SQL Server)
DENIX SELECT ON table_name TO user_name;
```

---

## 4. 事务控制语言 (TCL)

```sql
-- 开启事务
BEGIN TRANSACTION;
-- 或
START TRANSACTION;

-- 提交事务
COMMIT;

-- 回滚事务
ROLLBACK;

-- 保存点
SAVEPOINT savepoint_name;
ROLLBACK TO savepoint_name;
```

---

## 5. 高级查询

### 连接查询 (JOIN)
```sql
-- 内连接
SELECT * FROM table_a a
INNER JOIN table_b b ON a.id = b.a_id;

-- 左连接
SELECT * FROM table_a a
LEFT JOIN table_b b ON a.id = b.a_id;

-- 右连接
SELECT * FROM table_a a
RIGHT JOIN table_b b ON a.id = b.a_id;

-- 全外连接
SELECT * FROM table_a a
FULL OUTER JOIN table_b b ON a.id = b.a_id;

-- 交叉连接（笛卡尔积）
SELECT * FROM table_a
CROSS JOIN table_b;

-- 自连接
SELECT a.name, b.name AS manager
FROM employees a
JOIN employees b ON a.manager_id = b.id;
```

### 子查询
```sql
-- WHERE 子句中的子查询
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- IN 子查询
SELECT * FROM customers
WHERE id IN (SELECT customer_id FROM orders);

-- EXISTS 子查询
SELECT * FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.dept_id = d.id);

-- 关联子查询
SELECT e.name, e.salary, d.avg_salary
FROM employees e
JOIN (SELECT dept_id, AVG(salary) as avg_salary 
      FROM employees GROUP BY dept_id) d
ON e.dept_id = d.dept_id;
```

### 集合操作
```sql
-- 并集（去重）
SELECT column FROM table_a
UNION
SELECT column FROM table_b;

-- 并集（不去重）
SELECT column FROM table_a
UNION ALL
SELECT column FROM table_b;

-- 交集
SELECT column FROM table_a
INTERSECT
SELECT column FROM table_b;

-- 差集
SELECT column FROM table_a
EXCEPT  -- 或 MINUS (Oracle)
SELECT column FROM table_b;
```

---

## 6. 常用函数

### 字符串函数
```sql
CONCAT(str1, str2, ...)          -- 字符串拼接
SUBSTRING(str, start, length)      -- 截取子串
UPPER(str) / LOWER(str)            -- 大小写转换
LENGTH(str) / LEN(str)             -- 字符串长度
TRIM(str)                          -- 去除空格
REPLACE(str, old, new)             -- 替换
LIKE '%pattern%'                   -- 模糊匹配 (_, %通配符)
```

### 数值函数
```sql
ROUND(num, decimals)    -- 四舍五入
FLOOR(num)              -- 向下取整
CEIL(num) / CEILING(num) -- 向上取整
ABS(num)                -- 绝对值
MOD(a, b)               -- 取模
RAND() / RANDOM()       -- 随机数
```

### 日期函数
```sql
NOW() / CURRENT_TIMESTAMP   -- 当前日期时间
CURDATE() / CURRENT_DATE    -- 当前日期
DATE_FORMAT(date, '%Y-%m')  -- 格式化日期 (MySQL)
TO_CHAR(date, 'YYYY-MM')    -- 格式化日期 (Oracle/PostgreSQL)
DATEDIFF(day, date1, date2) -- 日期差 (SQL Server)
DATE_PART('year', date)     -- 提取日期部分 (PostgreSQL)
```

### 条件函数
```sql
-- IF 函数 (MySQL)
IF(condition, true_value, false_value)

-- CASE 表达式（通用）
CASE 
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ELSE default_result
END

-- COALESCE（返回第一个非 NULL 值）
COALESCE(value1, value2, ..., default_value)

-- NULLIF（相等返回 NULL）
NULLIF(value1, value2)
```

---

## 7. 约束与索引

```sql
-- 主键
PRIMARY KEY (column)

-- 外键
FOREIGN KEY (column) REFERENCES parent_table(parent_column)
ON DELETE CASCADE/SET NULL/RESTRICT
ON UPDATE CASCADE

-- 唯一约束
UNIQUE (column)

-- 非空约束
NOT NULL

-- 检查约束
CHECK (condition)

-- 默认值
DEFAULT value

-- 创建索引
CREATE INDEX idx_name ON table_name(column);
CREATE UNIQUE INDEX idx_name ON table_name(column);

-- 删除索引
DROP INDEX idx_name ON table_name;
```

---

## 8. 视图与常用操作

```sql
-- 创建视图
CREATE VIEW view_name AS
SELECT column1, column2 FROM table_name WHERE condition;

-- 创建视图并替换（Oracle/PostgreSQL）
CREATE OR REPLACE VIEW view_name AS ...;

-- 删除视图
DROP VIEW view_name;

-- 查看表结构
DESC table_name;           -- MySQL/Oracle
\d table_name              -- PostgreSQL
sp_help 'table_name'       -- SQL Server
```
