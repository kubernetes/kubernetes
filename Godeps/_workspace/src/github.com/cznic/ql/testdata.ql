// Copyright (c) 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// vi:filetype=sql

-- 0
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int);
	INSERT INTO t VALUES(11, 22, 33);
COMMIT;
SELECT * FROM t;
|lc1, lc2, lc3
[11 22 33]

-- 1
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int);
	CREATE TABLE t (c1 int);
COMMIT;
||table.*exists

-- 2
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c1 int, c4 int);
COMMIT;
||duplicate column

-- 3
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int);
	ALTER TABLE t ADD c4 string;
	INSERT INTO t VALUES (1, 2, 3, "foo");
COMMIT;
SELECT * FROM t;
|lc1, lc2, lc3, sc4
[1 2 3 foo]

-- 4
BEGIN TRANSACTION;
	ALTER TABLE none ADD c1 int;
COMMIT;
||table .* not exist

-- 5
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int);
	ALTER TABLE t ADD c2 int;
COMMIT;
||column .* exists

-- 6
BEGIN TRANSACTION;
	ALTER TABLE none DROP COLUMN c1;
COMMIT;
||table .* not exist

-- 7
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int);
	ALTER TABLE t DROP COLUMN c4;
COMMIT;
||column .* not exist

-- 8
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int);
	ALTER TABLE t DROP COLUMN c2;
	INSERT INTO t VALUES (1, 2);
COMMIT;
SELECT * FROM t;
|lc1, lc3
[1 2]

-- 9
BEGIN TRANSACTION;
	DROP TABLE none;
COMMIT;
||table .* not exist

-- 10
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int);
	DROP TABLE t;
COMMIT;
SELECT * FROM t;
||table .* not exist

-- 11
BEGIN TRANSACTION;
	INSERT INTO none VALUES (1, 2);
COMMIT;
||table .* not exist

-- 12
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int);
	INSERT INTO t VALUES (1);
COMMIT;
||expect

-- 13
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int);
	INSERT INTO t VALUES (1, 2, 3);
COMMIT;
||expect

-- 14
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int);
	INSERT INTO t VALUES (1, 2/(3*5-15));
COMMIT;
||division by zero

-- 15
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int);
	INSERT INTO t VALUES (2+3*4, 2*3+4);
COMMIT;
SELECT * FROM t;
|lc1, lc2
[14 10]

-- 16
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int, c4 int);
	INSERT INTO t (c2, c4) VALUES (1);
COMMIT;
||expect

-- 17
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int, c4 int);
	INSERT INTO t (c2, c4) VALUES (1, 2, 3);
COMMIT;
||expect

-- 18
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int, c4 int);
	INSERT INTO t (c2, none) VALUES (1, 2);
COMMIT;
||unknown

-- 19
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int, c4 int);
	INSERT INTO t (c2, c3) VALUES (2+3*4, 2*3+4);
	INSERT INTO t VALUES (1, 2, 3, 4, );
COMMIT;
SELECT * FROM t;
|lc1, lc2, lc3, lc4
[1 2 3 4]
[<nil> 14 10 <nil>]

-- 20
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 int, c3 int, c4 int);
	ALTER TABLE t DROP COLUMN c3;
	INSERT INTO t (c1, c4) VALUES (42, 314);
	INSERT INTO t (c1, c2) VALUES (2+3*4, 2*3+4);
	INSERT INTO t VALUES (1, 2, 3);
COMMIT;
SELECT * FROM t;
|lc1, lc2, lc4
[1 2 3]
[14 10 <nil>]
[42 <nil> 314]

-- 21
BEGIN TRANSACTION;
	TRUNCATE TABLE none;
COMMIT;
||table .* not exist

-- 22
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int);
	INSERT INTO t VALUES(278);
	TRUNCATE TABLE t;
	INSERT INTO t VALUES(314);
COMMIT;
SELECT * FROM t;
|lc1
[314]

-- 23
SELECT * FROM none;
||table .* not exist

-- 24
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (2, "b");
	INSERT INTO t VALUES (1, "a");
COMMIT;
SELECT * FROM t;
|lc1, sc2
[1 a]
[2 b]

-- 25
SELECT c1 FROM none;
||table .* not exist

-- 26
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
COMMIT;
SELECT none FROM t;
||unknown

-- 27
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
COMMIT;
SELECT c1, none, c2 FROM t;
||unknown

-- 28
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int32, c2 string);
	INSERT INTO t VALUES (2, "b");
	INSERT INTO t VALUES (1, "a");
COMMIT;
SELECT 3*c1 AS v FROM t;
|kv
[3]
[6]

-- 29
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (2, "b");
	INSERT INTO t VALUES (1, "a");
COMMIT;
SELECT c2 FROM t;
|sc2
[a]
[b]

-- 30
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (2, "b");
	INSERT INTO t VALUES (1, "a");
COMMIT;
SELECT c1 AS X, c2 FROM t;
|lX, sc2
[1 a]
[2 b]

-- 31
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (2, "b");
	INSERT INTO t VALUES (1, "a");
COMMIT;
SELECT c2, c1 AS Y FROM t;
|sc2, lY
[a 1]
[b 2]

-- 32
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
COMMIT;
SELECT * FROM t WHERE c3 == 1;
||unknown

-- 33
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
COMMIT;
SELECT * FROM t WHERE c1 == 1;
|lc1, sc2
[1 a]

-- 34
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
COMMIT;
SELECT * FROM t ORDER BY c3;
||unknown

-- 35
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int32, c2 string);
	INSERT INTO t VALUES (22, "bc");
	INSERT INTO t VALUES (11, "ab");
	INSERT INTO t VALUES (33, "cd");
COMMIT;
SELECT * FROM t ORDER BY c1;
|kc1, sc2
[11 ab]
[22 bc]
[33 cd]

-- 36
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
COMMIT;
SELECT * FROM t ORDER BY c1 ASC;
|lc1, sc2
[1 a]
[2 b]

-- 37
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
COMMIT;
SELECT * FROM t ORDER BY c1 DESC;
|lc1, sc2
[2 b]
[1 a]

-- 38
BEGIN TRANSACTION;
CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "b");
	INSERT INTO t VALUES (3, "c");
	INSERT INTO t VALUES (4, "d");
	INSERT INTO t VALUES (5, "e");
	INSERT INTO t VALUES (6, "f");
	INSERT INTO t VALUES (7, "g");
COMMIT;
SELECT * FROM t
WHERE c1 % 2 == 0
ORDER BY c2 DESC;
|lc1, sc2
[6 f]
[4 d]
[2 b]

-- 39
BEGIN TRANSACTION;
CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (2, "a");
	INSERT INTO t VALUES (3, "b");
	INSERT INTO t VALUES (4, "b");
	INSERT INTO t VALUES (5, "c");
	INSERT INTO t VALUES (6, "c");
	INSERT INTO t VALUES (7, "d");
COMMIT;
SELECT * FROM t
ORDER BY c1, c2;
|lc1, sc2
[1 a]
[2 a]
[3 b]
[4 b]
[5 c]
[6 c]
[7 d]

-- 40
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int, c2 string);
	INSERT INTO t VALUES (1, "d");
	INSERT INTO t VALUES (2, "c");
	INSERT INTO t VALUES (3, "c");
	INSERT INTO t VALUES (4, "b");
	INSERT INTO t VALUES (5, "b");
	INSERT INTO t VALUES (6, "a");
	INSERT INTO t VALUES (7, "a");
COMMIT;
SELECT * FROM t
ORDER BY c2, c1
|lc1, sc2
[6 a]
[7 a]
[4 b]
[5 b]
[2 c]
[3 c]
[1 d]

-- S 41
SELECT * FROM employee, none;
||table .* not exist

-- S 42
SELECT employee.LastName FROM employee, none;
||table .* not exist

-- S 43
SELECT none FROM employee, department;
||unknown

-- S 44
SELECT employee.LastName FROM employee, department;
|semployee.LastName
[John]
[John]
[John]
[John]
[Smith]
[Smith]
[Smith]
[Smith]
[Robinson]
[Robinson]
[Robinson]
[Robinson]
[Heisenberg]
[Heisenberg]
[Heisenberg]
[Heisenberg]
[Jones]
[Jones]
[Jones]
[Jones]
[Rafferty]
[Rafferty]
[Rafferty]
[Rafferty]

-- S 45
SELECT * FROM employee, department
ORDER by employee.LastName;
|semployee.LastName, lemployee.DepartmentID, ldepartment.DepartmentID, sdepartment.DepartmentName
[Heisenberg 33 35 Marketing]
[Heisenberg 33 34 Clerical]
[Heisenberg 33 33 Engineering]
[Heisenberg 33 31 Sales]
[John <nil> 35 Marketing]
[John <nil> 34 Clerical]
[John <nil> 33 Engineering]
[John <nil> 31 Sales]
[Jones 33 35 Marketing]
[Jones 33 34 Clerical]
[Jones 33 33 Engineering]
[Jones 33 31 Sales]
[Rafferty 31 35 Marketing]
[Rafferty 31 34 Clerical]
[Rafferty 31 33 Engineering]
[Rafferty 31 31 Sales]
[Robinson 34 35 Marketing]
[Robinson 34 34 Clerical]
[Robinson 34 33 Engineering]
[Robinson 34 31 Sales]
[Smith 34 35 Marketing]
[Smith 34 34 Clerical]
[Smith 34 33 Engineering]
[Smith 34 31 Sales]

-- S 46
SELECT *
FROM employee, department
WHERE employee.DepartmentID == department.DepartmentID;
|semployee.LastName, lemployee.DepartmentID, ldepartment.DepartmentID, sdepartment.DepartmentName
[Smith 34 34 Clerical]
[Robinson 34 34 Clerical]
[Heisenberg 33 33 Engineering]
[Jones 33 33 Engineering]
[Rafferty 31 31 Sales]

-- S 47
SELECT department.DepartmentName, department.DepartmentID, employee.LastName, employee.DepartmentID
FROM employee, department
WHERE employee.DepartmentID == department.DepartmentID
ORDER BY department.DepartmentName, employee.LastName;
|sdepartment.DepartmentName, ldepartment.DepartmentID, semployee.LastName, lemployee.DepartmentID
[Clerical 34 Robinson 34]
[Clerical 34 Smith 34]
[Engineering 33 Heisenberg 33]
[Engineering 33 Jones 33]
[Sales 31 Rafferty 31]

-- S 48
SELECT department.DepartmentName, department.DepartmentID, employee.LastName, employee.DepartmentID
FROM employee, department
WHERE department.DepartmentName IN ("Sales", "Engineering", "HR", "Clerical")
ORDER BY employee.LastName;
|sdepartment.DepartmentName, ldepartment.DepartmentID, semployee.LastName, lemployee.DepartmentID
[Clerical 34 Heisenberg 33]
[Engineering 33 Heisenberg 33]
[Sales 31 Heisenberg 33]
[Clerical 34 John <nil>]
[Engineering 33 John <nil>]
[Sales 31 John <nil>]
[Clerical 34 Jones 33]
[Engineering 33 Jones 33]
[Sales 31 Jones 33]
[Clerical 34 Rafferty 31]
[Engineering 33 Rafferty 31]
[Sales 31 Rafferty 31]
[Clerical 34 Robinson 34]
[Engineering 33 Robinson 34]
[Sales 31 Robinson 34]
[Clerical 34 Smith 34]
[Engineering 33 Smith 34]
[Sales 31 Smith 34]

-- S 49
SELECT department.DepartmentName, department.DepartmentID, employee.LastName, employee.DepartmentID
FROM employee, department
WHERE (department.DepartmentID+1000) IN (1031, 1035, 1036)
ORDER BY employee.LastName;
|sdepartment.DepartmentName, ldepartment.DepartmentID, semployee.LastName, lemployee.DepartmentID
[Marketing 35 Heisenberg 33]
[Sales 31 Heisenberg 33]
[Marketing 35 John <nil>]
[Sales 31 John <nil>]
[Marketing 35 Jones 33]
[Sales 31 Jones 33]
[Marketing 35 Rafferty 31]
[Sales 31 Rafferty 31]
[Marketing 35 Robinson 34]
[Sales 31 Robinson 34]
[Marketing 35 Smith 34]
[Sales 31 Smith 34]

-- S 50
SELECT department.DepartmentName, department.DepartmentID, employee.LastName, employee.DepartmentID
FROM employee, department
WHERE department.DepartmentName NOT IN ("Engineering", "HR", "Clerical");
|sdepartment.DepartmentName, ldepartment.DepartmentID, semployee.LastName, ?employee.DepartmentID
[Marketing 35 John <nil>]
[Sales 31 John <nil>]
[Marketing 35 Smith 34]
[Sales 31 Smith 34]
[Marketing 35 Robinson 34]
[Sales 31 Robinson 34]
[Marketing 35 Heisenberg 33]
[Sales 31 Heisenberg 33]
[Marketing 35 Jones 33]
[Sales 31 Jones 33]
[Marketing 35 Rafferty 31]
[Sales 31 Rafferty 31]

-- S 51
SELECT department.DepartmentName, department.DepartmentID, employee.LastName, employee.DepartmentID
FROM employee, department
WHERE department.DepartmentID BETWEEN 34 AND 36
ORDER BY employee.LastName;
|sdepartment.DepartmentName, ldepartment.DepartmentID, semployee.LastName, lemployee.DepartmentID
[Marketing 35 Heisenberg 33]
[Clerical 34 Heisenberg 33]
[Marketing 35 John <nil>]
[Clerical 34 John <nil>]
[Marketing 35 Jones 33]
[Clerical 34 Jones 33]
[Marketing 35 Rafferty 31]
[Clerical 34 Rafferty 31]
[Marketing 35 Robinson 34]
[Clerical 34 Robinson 34]
[Marketing 35 Smith 34]
[Clerical 34 Smith 34]

-- S 52
SELECT department.DepartmentName, department.DepartmentID, employee.LastName, employee.DepartmentID
FROM employee, department
WHERE department.DepartmentID BETWEEN int64(34) AND int64(36)
ORDER BY employee.LastName;
|sdepartment.DepartmentName, ldepartment.DepartmentID, semployee.LastName, lemployee.DepartmentID
[Marketing 35 Heisenberg 33]
[Clerical 34 Heisenberg 33]
[Marketing 35 John <nil>]
[Clerical 34 John <nil>]
[Marketing 35 Jones 33]
[Clerical 34 Jones 33]
[Marketing 35 Rafferty 31]
[Clerical 34 Rafferty 31]
[Marketing 35 Robinson 34]
[Clerical 34 Robinson 34]
[Marketing 35 Smith 34]
[Clerical 34 Smith 34]

-- S 53
SELECT department.DepartmentName, department.DepartmentID, employee.LastName, employee.DepartmentID
FROM employee, department
WHERE department.DepartmentID NOT BETWEEN 33 AND 34
ORDER BY employee.LastName;
|sdepartment.DepartmentName, ldepartment.DepartmentID, semployee.LastName, lemployee.DepartmentID
[Marketing 35 Heisenberg 33]
[Sales 31 Heisenberg 33]
[Marketing 35 John <nil>]
[Sales 31 John <nil>]
[Marketing 35 Jones 33]
[Sales 31 Jones 33]
[Marketing 35 Rafferty 31]
[Sales 31 Rafferty 31]
[Marketing 35 Robinson 34]
[Sales 31 Robinson 34]
[Marketing 35 Smith 34]
[Sales 31 Smith 34]

-- S 54
SELECT LastName, LastName FROM employee;
||duplicate

-- S 55
SELECT LastName+", " AS a, LastName AS a FROM employee;
||duplicate

-- S 56
SELECT LastName AS a, LastName AS b FROM employee
ORDER by a, b;
|sa, sb
[Heisenberg Heisenberg]
[John John]
[Jones Jones]
[Rafferty Rafferty]
[Robinson Robinson]
[Smith Smith]

-- S 57
SELECT employee.LastName AS name, employee.DepartmentID AS id, department.DepartmentName AS department, department.DepartmentID AS id2
FROM employee, department
WHERE employee.DepartmentID == department.DepartmentID
ORDER BY name, id, department, id2;
|sname, lid, sdepartment, lid2
[Heisenberg 33 Engineering 33]
[Jones 33 Engineering 33]
[Rafferty 31 Sales 31]
[Robinson 34 Clerical 34]
[Smith 34 Clerical 34]

-- S 58
SELECT * FROM;
||syntax

-- S 59
SELECT * FROM employee
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 33]
[John <nil>]
[Jones 33]
[Rafferty 31]
[Robinson 34]
[Smith 34]

-- S 60
SELECT * FROM employee AS e
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 33]
[John <nil>]
[Jones 33]
[Rafferty 31]
[Robinson 34]
[Smith 34]

-- S 61
SELECT none FROM (
	SELECT * FROM employee;
	SELECT * FROM department;
);
||syntax

-- S 62
SELECT none FROM (
	SELECT * FROM employee;
);
||unknown

-- S 63
SELECT noneCol FROM (
	SELECT * FROM noneTab
);
||not exist

-- S 64
SELECT noneCol FROM (
	SELECT * FROM employee
);
||unknown

-- S 65
SELECT * FROM (
	SELECT * FROM employee
)
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 33]
[John <nil>]
[Jones 33]
[Rafferty 31]
[Robinson 34]
[Smith 34]

-- S 66
SELECT * FROM (
	SELECT LastName AS Name FROM employee
)
ORDER BY Name;
|sName
[Heisenberg]
[John]
[Jones]
[Rafferty]
[Robinson]
[Smith]

-- S 67
SELECT Name FROM (
	SELECT LastName AS name FROM employee
);
||unknown

-- S 68
SELECT name AS Name FROM (
	SELECT LastName AS name
	FROM employee AS e
)
ORDER BY Name;
|sName
[Heisenberg]
[John]
[Jones]
[Rafferty]
[Robinson]
[Smith]

-- S 69
SELECT name AS Name FROM (
	SELECT LastName AS name FROM employee
)
ORDER BY Name;
|sName
[Heisenberg]
[John]
[Jones]
[Rafferty]
[Robinson]
[Smith]

-- S 70
SELECT employee.LastName, department.DepartmentName, department.DepartmentID FROM (
	SELECT *
	FROM employee, department
	WHERE employee.DepartmentID == department.DepartmentID
)
ORDER BY department.DepartmentName, employee.LastName
|semployee.LastName, sdepartment.DepartmentName, ldepartment.DepartmentID
[Robinson Clerical 34]
[Smith Clerical 34]
[Heisenberg Engineering 33]
[Jones Engineering 33]
[Rafferty Sales 31]

-- S 71
SELECT e.LastName, d.DepartmentName, d.DepartmentID FROM (
	SELECT *
	FROM employee AS e, department AS d
	WHERE e.DepartmentID == d.DepartmentID
)
ORDER by d.DepartmentName, e.LastName;
|se.LastName, sd.DepartmentName, ld.DepartmentID
[Robinson Clerical 34]
[Smith Clerical 34]
[Heisenberg Engineering 33]
[Jones Engineering 33]
[Rafferty Sales 31]

-- S 72
SELECT e.LastName AS name, d.DepartmentName AS department, d.DepartmentID AS id FROM (
	SELECT *
	FROM employee AS e, department AS d
	WHERE e.DepartmentID == d.DepartmentID
)
ORDER by department, name
|sname, sdepartment, lid
[Robinson Clerical 34]
[Smith Clerical 34]
[Heisenberg Engineering 33]
[Jones Engineering 33]
[Rafferty Sales 31]

-- S 73
SELECT name, department, id FROM (
	SELECT e.LastName AS name, e.DepartmentID AS id, d.DepartmentName AS department, d.DepartmentID AS fid
	FROM employee AS e, department AS d
	WHERE e.DepartmentID == d.DepartmentID
)
ORDER by department, name;
|sname, sdepartment, lid
[Robinson Clerical 34]
[Smith Clerical 34]
[Heisenberg Engineering 33]
[Jones Engineering 33]
[Rafferty Sales 31]

-- S 74
SELECT *
FROM
(
	SELECT *
	FROM employee
),
(
	SELECT *
	FROM department
);
|s, ?, l, s
[John <nil> 35 Marketing]
[John <nil> 34 Clerical]
[John <nil> 33 Engineering]
[John <nil> 31 Sales]
[Smith 34 35 Marketing]
[Smith 34 34 Clerical]
[Smith 34 33 Engineering]
[Smith 34 31 Sales]
[Robinson 34 35 Marketing]
[Robinson 34 34 Clerical]
[Robinson 34 33 Engineering]
[Robinson 34 31 Sales]
[Heisenberg 33 35 Marketing]
[Heisenberg 33 34 Clerical]
[Heisenberg 33 33 Engineering]
[Heisenberg 33 31 Sales]
[Jones 33 35 Marketing]
[Jones 33 34 Clerical]
[Jones 33 33 Engineering]
[Jones 33 31 Sales]
[Rafferty 31 35 Marketing]
[Rafferty 31 34 Clerical]
[Rafferty 31 33 Engineering]
[Rafferty 31 31 Sales]

-- S 75
SELECT *
FROM
(
	SELECT *
	FROM employee
) AS e,
(
	SELECT *
	FROM department
)
ORDER BY e.LastName, e.DepartmentID;
|se.LastName, le.DepartmentID, l, s
[Heisenberg 33 35 Marketing]
[Heisenberg 33 34 Clerical]
[Heisenberg 33 33 Engineering]
[Heisenberg 33 31 Sales]
[John <nil> 35 Marketing]
[John <nil> 34 Clerical]
[John <nil> 33 Engineering]
[John <nil> 31 Sales]
[Jones 33 35 Marketing]
[Jones 33 34 Clerical]
[Jones 33 33 Engineering]
[Jones 33 31 Sales]
[Rafferty 31 35 Marketing]
[Rafferty 31 34 Clerical]
[Rafferty 31 33 Engineering]
[Rafferty 31 31 Sales]
[Robinson 34 35 Marketing]
[Robinson 34 34 Clerical]
[Robinson 34 33 Engineering]
[Robinson 34 31 Sales]
[Smith 34 35 Marketing]
[Smith 34 34 Clerical]
[Smith 34 33 Engineering]
[Smith 34 31 Sales]

-- S 76
SELECT *
FROM
(
	SELECT *
	FROM employee
),
(
	SELECT *
	FROM department
) AS d
ORDER BY d.DepartmentID DESC;
|s, l, ld.DepartmentID, sd.DepartmentName
[Rafferty 31 35 Marketing]
[Jones 33 35 Marketing]
[Heisenberg 33 35 Marketing]
[Robinson 34 35 Marketing]
[Smith 34 35 Marketing]
[John <nil> 35 Marketing]
[Rafferty 31 34 Clerical]
[Jones 33 34 Clerical]
[Heisenberg 33 34 Clerical]
[Robinson 34 34 Clerical]
[Smith 34 34 Clerical]
[John <nil> 34 Clerical]
[Rafferty 31 33 Engineering]
[Jones 33 33 Engineering]
[Heisenberg 33 33 Engineering]
[Robinson 34 33 Engineering]
[Smith 34 33 Engineering]
[John <nil> 33 Engineering]
[Rafferty 31 31 Sales]
[Jones 33 31 Sales]
[Heisenberg 33 31 Sales]
[Robinson 34 31 Sales]
[Smith 34 31 Sales]
[John <nil> 31 Sales]

-- S 77
SELECT *
FROM
	employee,
	(
		SELECT *
		FROM department
	)
ORDER BY employee.LastName;
|semployee.LastName, lemployee.DepartmentID, l, s
[Heisenberg 33 35 Marketing]
[Heisenberg 33 34 Clerical]
[Heisenberg 33 33 Engineering]
[Heisenberg 33 31 Sales]
[John <nil> 35 Marketing]
[John <nil> 34 Clerical]
[John <nil> 33 Engineering]
[John <nil> 31 Sales]
[Jones 33 35 Marketing]
[Jones 33 34 Clerical]
[Jones 33 33 Engineering]
[Jones 33 31 Sales]
[Rafferty 31 35 Marketing]
[Rafferty 31 34 Clerical]
[Rafferty 31 33 Engineering]
[Rafferty 31 31 Sales]
[Robinson 34 35 Marketing]
[Robinson 34 34 Clerical]
[Robinson 34 33 Engineering]
[Robinson 34 31 Sales]
[Smith 34 35 Marketing]
[Smith 34 34 Clerical]
[Smith 34 33 Engineering]
[Smith 34 31 Sales]

-- S 78
SELECT *
FROM
(
	SELECT *
	FROM employee
) AS e,
(
	SELECT *
	FROM department
) AS d
WHERE e.DepartmentID == d.DepartmentID
ORDER BY d.DepartmentName, e.LastName;
|se.LastName, le.DepartmentID, ld.DepartmentID, sd.DepartmentName
[Robinson 34 34 Clerical]
[Smith 34 34 Clerical]
[Heisenberg 33 33 Engineering]
[Jones 33 33 Engineering]
[Rafferty 31 31 Sales]

-- S 79
SELECT *
FROM
	employee,
	(
		SELECT *
		FROM department
	) AS d
WHERE employee.DepartmentID == d.DepartmentID
ORDER BY d.DepartmentName, employee.LastName;
|semployee.LastName, lemployee.DepartmentID, ld.DepartmentID, sd.DepartmentName
[Robinson 34 34 Clerical]
[Smith 34 34 Clerical]
[Heisenberg 33 33 Engineering]
[Jones 33 33 Engineering]
[Rafferty 31 31 Sales]

-- S 80
SELECT *
FROM
	employee AS e,
	(
		SELECT *
		FROM department
	) AS d
WHERE e.DepartmentID == d.DepartmentID
ORDER BY d.DepartmentName, e.LastName;
|se.LastName, le.DepartmentID, ld.DepartmentID, sd.DepartmentName
[Robinson 34 34 Clerical]
[Smith 34 34 Clerical]
[Heisenberg 33 33 Engineering]
[Jones 33 33 Engineering]
[Rafferty 31 31 Sales]

-- S 81
SELECT *
FROM
	employee AS e,
	(
		SELECT *
		FROM department
	) AS d
WHERE e.DepartmentID == d.DepartmentID == true
ORDER BY e.DepartmentID, e.LastName;
|se.LastName, le.DepartmentID, ld.DepartmentID, sd.DepartmentName
[Rafferty 31 31 Sales]
[Heisenberg 33 33 Engineering]
[Jones 33 33 Engineering]
[Robinson 34 34 Clerical]
[Smith 34 34 Clerical]

-- S 82
SELECT *
FROM
	employee AS e,
	(
		SELECT *
		FROM department
	) AS d
WHERE e.DepartmentID != d.DepartmentID == false
ORDER BY e.DepartmentID, e.LastName;
|se.LastName, le.DepartmentID, ld.DepartmentID, sd.DepartmentName
[Rafferty 31 31 Sales]
[Heisenberg 33 33 Engineering]
[Jones 33 33 Engineering]
[Robinson 34 34 Clerical]
[Smith 34 34 Clerical]

-- 83
BEGIN TRANSACTION;
	CREATE TABLE t (c1 bool);
	INSERT INTO t VALUES (1);
COMMIT;
||cannot .* int64.*bool .* c1

-- 84
BEGIN TRANSACTION;
	CREATE TABLE t (c1 bool);
	INSERT INTO t VALUES (true);
COMMIT;
SELECT * from t;
|bc1
[true]

-- 85
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int8);
	INSERT INTO t VALUES ("foo");
COMMIT;
||cannot .* string.*int8 .* c1

-- 86
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int8);
	INSERT INTO t VALUES (0x1234);
COMMIT;
SELECT * from t;
||overflow

-- 87
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int16);
	INSERT INTO t VALUES (87);
COMMIT;
SELECT * from t;
|jc1
[87]

-- 88
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int16);
	INSERT INTO t VALUES (int16(0x12345678));
COMMIT;
SELECT * from t;
|jc1
[22136]

-- 89
BEGIN TRANSACTION;
CREATE TABLE t (c1 int32);
	INSERT INTO t VALUES (uint32(1));
COMMIT;
||cannot .* uint32.*int32 .* c1

-- 90
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int32);
	INSERT INTO t VALUES (0xabcd12345678);
COMMIT;
SELECT * from t;
||overflow

-- 91
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int64);
	INSERT INTO t VALUES (int8(1));
COMMIT;
||cannot .* int8.*int64 .* c1

-- 92
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int64);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t;
|lc1
[1]

-- 93
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int);
	INSERT INTO t VALUES (int8(1));
COMMIT;
||cannot .* int8.*int64 .* c1

-- 94
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int);
	INSERT INTO t VALUES (94);
COMMIT;
SELECT * from t;
|lc1
[94]

-- 95
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint8);
	INSERT INTO t VALUES (95);
COMMIT;
SELECT * from t;
|uc1
[95]

-- 96
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint8);
	INSERT INTO t VALUES (uint8(0x1234));
COMMIT;
SELECT * from t;
|uc1
[52]

-- 97
BEGIN TRANSACTION;
	CREATE TABLE t (c1 byte);
	INSERT INTO t VALUES (int8(1));
COMMIT;
||cannot .* int8.*uint8 .* c1

-- 98
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint8);
	INSERT INTO t VALUES (byte(0x1234));
COMMIT;
SELECT * from t;
|uc1
[52]

-- 99
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint16);
	INSERT INTO t VALUES (int(1));
COMMIT;
||cannot .* int64.*uint16 .* c1

-- 100
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint16);
	INSERT INTO t VALUES (0x12345678);
COMMIT;
SELECT * from t;
||overflow

-- 101
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint32);
	INSERT INTO t VALUES (int32(1));
COMMIT;
||cannot .* int32.*uint32 .* c1

-- 102
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint32);
	INSERT INTO t VALUES (uint32(0xabcd12345678));
COMMIT;
SELECT * from t;
|wc1
[305419896]

-- 103
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint64);
	INSERT INTO t VALUES (int(1));
COMMIT;
||cannot .* int64.*uint64 .* c1

-- 104
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint64);
	INSERT INTO t VALUES (uint64(1));
COMMIT;
SELECT * from t;
|xc1
[1]

-- 105
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint);
	INSERT INTO t VALUES (int(1));
COMMIT;
||cannot .* int64.*uint64 .* c1

-- 106
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t;
|xc1
[1]

-- 107
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float32);
	INSERT INTO t VALUES (107);
COMMIT;
SELECT * from t;
|fc1
[107]

-- 108
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float32);
	INSERT INTO t VALUES (float64(1));
COMMIT;
SELECT * from t;
||cannot .* float64.*float32 .* c1

-- 109
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float32);
	INSERT INTO t VALUES (1.2);
COMMIT;
SELECT * from t;
|fc1
[1.2]

-- 110
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float64);
	INSERT INTO t VALUES (1.2);
COMMIT;
SELECT * from t;
|gc1
[1.2]

-- 111
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float);
	INSERT INTO t VALUES (111.1);
COMMIT;
SELECT * from t;
|gc1
[111.1]

-- 112
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float);
	INSERT INTO t VALUES (-112.1);
COMMIT;
SELECT * from t;
|gc1
[-112.1]

-- 113
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex64);
	INSERT INTO t VALUES (complex(1, 0.5));
COMMIT;
SELECT * from t;
|cc1
[(1+0.5i)]

-- 114
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex64);
	INSERT INTO t VALUES (complex128(complex(1, 0.5)));
COMMIT;
SELECT * from t;
||cannot .* complex128.*complex64 .* c1

-- 115
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex128);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t;
|dc1
[(1+0i)]

-- 116
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex128);
	INSERT INTO t VALUES (complex(1, 0.5));
COMMIT;
SELECT * from t;
|dc1
[(1+0.5i)]

-- 117
BEGIN TRANSACTION;
	CREATE TABLE t (c1 string);
	INSERT INTO t VALUES (1);
COMMIT;
||cannot .* int64.*string .* c1

-- 118
BEGIN TRANSACTION;
	CREATE TABLE t (c1 string);
	INSERT INTO t VALUES ("a"+"b");
COMMIT;
SELECT * from t;
|sc1
[ab]

-- 119
BEGIN TRANSACTION;
	CREATE TABLE t (c1 bool);
	INSERT INTO t VALUES (true);
COMMIT;
SELECT * from t
WHERE c1 > 3;
||operator .* not defined .* bool

-- 120
BEGIN TRANSACTION;
	CREATE TABLE t (c1 bool);
	INSERT INTO t VALUES (true);
COMMIT;
SELECT * from t
WHERE c1;
|bc1
[true]

-- 121
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int8);
	INSERT INTO t VALUES (float(1));
COMMIT;
SELECT * from t
WHERE c1 == 8;
||cannot .* float64.*int8 .* c1

-- 122
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int8);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == int8(1);
|ic1
[1]

-- 123
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int16);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == int(8);
||mismatched .* int16 .* int64

-- 124
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int16);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == 1;
|jc1
[1]

-- 125
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int32);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == int(8);
||mismatched .* int32 .* int64

-- 126
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int32);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == 1;
|kc1
[1]

-- 127
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int64);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == byte(8);
||mismatched .* int64 .* uint8

-- 128
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int64);
	INSERT INTO t VALUES (int64(1));
COMMIT;
SELECT * from t
WHERE c1 == 1;
|lc1
[1]

-- 129
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == 1;
|lc1
[1]

-- 130
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint8);
	INSERT INTO t VALUES (byte(1));
COMMIT;
SELECT * from t
WHERE c1 == int8(8);
||mismatched .* uint8 .* int8

-- 131
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint8);
	INSERT INTO t VALUES (byte(1));
COMMIT;
SELECT * from t
WHERE c1 == 1;
|uc1
[1]

-- 132
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint16);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == byte(8);
||mismatched .* uint16 .* uint8

-- 133
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint16);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == 1;
|vc1
[1]

-- 134
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint32);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == 1;
|wc1
[1]

-- 135
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint64);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == int(8);
||mismatched .* uint64 .* int64

-- 136
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint64);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == 1;
|xc1
[1]

-- 137
BEGIN TRANSACTION;
	CREATE TABLE t (c1 uint);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * from t
WHERE c1 == 1;
|xc1
[1]

-- 138
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float32);
	INSERT INTO t VALUES (8);
COMMIT;
SELECT * from t
WHERE c1 == byte(8);
||mismatched .* float32 .* uint8

-- 139
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float32);
	INSERT INTO t VALUES (8);
COMMIT;
SELECT * from t
WHERE c1 == 8;
|fc1
[8]

-- 140
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float64);
	INSERT INTO t VALUES (2);
COMMIT;
SELECT * from t
WHERE c1 == byte(2);
||mismatched .* float64 .* uint8

-- 141
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float64);
	INSERT INTO t VALUES (2);
COMMIT;
SELECT * from t
WHERE c1 == 2;
|gc1
[2]

-- 142
BEGIN TRANSACTION;
	CREATE TABLE t (c1 float);
	INSERT INTO t VALUES (2.);
COMMIT;
SELECT * from t
WHERE c1 == 2;
|gc1
[2]

-- 143
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex64);
	INSERT INTO t VALUES (complex(2., 5.));
COMMIT;
SELECT * from t
WHERE c1 == "foo";
||mismatched .* complex64 .* string

-- 144
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex64);
	INSERT INTO t VALUES (complex(2, 5.));
COMMIT;
SELECT * from t
WHERE c1 == 2+5i;
|cc1
[(2+5i)]

-- 145
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex128);
	INSERT INTO t VALUES (2+5i);
COMMIT;
SELECT * from t
WHERE c1 == "2";
||mismatched .* complex128 .* string

-- 146
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex128);
	INSERT INTO t VALUES (2+5i);
COMMIT;
SELECT * from t
WHERE c1 == complex(2, 5);
|dc1
[(2+5i)]

-- 147
BEGIN TRANSACTION;
	CREATE TABLE t (c1 string);
	INSERT INTO t VALUES ("foo");
COMMIT;
SELECT * from t
WHERE c1 == 2;
||mismatched .* string .* int64

-- 148
BEGIN TRANSACTION;
	CREATE TABLE t (c1 string);
	INSERT INTO t VALUES ("f"+"oo");
COMMIT;
SELECT * from t
WHERE c1 == "fo"+"o";
|sc1
[foo]

-- 149
SELECT 2/(3*5-15) AS foo FROM bar;
||division by zero

-- 150
SELECT 2.0/(2.0-2.0) AS foo FROM bar;
||division by zero

-- 151
SELECT 2i/(2i-2i) AS foo FROM bar;
||division by zero

-- 152
SELECT 2/(3*5-x) AS foo FROM bar;
||table .* not exist

-- S 153
SELECT 314, 42 AS AUQLUE, DepartmentID, DepartmentID+1000, LastName AS Name
FROM employee
ORDER BY Name;
|l, lAUQLUE, lDepartmentID, l, sName
[314 42 33 1033 Heisenberg]
[314 42 <nil> <nil> John]
[314 42 33 1033 Jones]
[314 42 31 1031 Rafferty]
[314 42 34 1034 Robinson]
[314 42 34 1034 Smith]

-- S 154
SELECT *
FROM
	employee AS e,
	( SELECT * FROM department)
ORDER BY e.LastName;
|se.LastName, le.DepartmentID, l, s
[Heisenberg 33 35 Marketing]
[Heisenberg 33 34 Clerical]
[Heisenberg 33 33 Engineering]
[Heisenberg 33 31 Sales]
[John <nil> 35 Marketing]
[John <nil> 34 Clerical]
[John <nil> 33 Engineering]
[John <nil> 31 Sales]
[Jones 33 35 Marketing]
[Jones 33 34 Clerical]
[Jones 33 33 Engineering]
[Jones 33 31 Sales]
[Rafferty 31 35 Marketing]
[Rafferty 31 34 Clerical]
[Rafferty 31 33 Engineering]
[Rafferty 31 31 Sales]
[Robinson 34 35 Marketing]
[Robinson 34 34 Clerical]
[Robinson 34 33 Engineering]
[Robinson 34 31 Sales]
[Smith 34 35 Marketing]
[Smith 34 34 Clerical]
[Smith 34 33 Engineering]
[Smith 34 31 Sales]

-- S 155
SELECT * FROM employee AS e, ( SELECT * FROM department) AS d
ORDER BY e.LastName;
|se.LastName, le.DepartmentID, ld.DepartmentID, sd.DepartmentName
[Heisenberg 33 35 Marketing]
[Heisenberg 33 34 Clerical]
[Heisenberg 33 33 Engineering]
[Heisenberg 33 31 Sales]
[John <nil> 35 Marketing]
[John <nil> 34 Clerical]
[John <nil> 33 Engineering]
[John <nil> 31 Sales]
[Jones 33 35 Marketing]
[Jones 33 34 Clerical]
[Jones 33 33 Engineering]
[Jones 33 31 Sales]
[Rafferty 31 35 Marketing]
[Rafferty 31 34 Clerical]
[Rafferty 31 33 Engineering]
[Rafferty 31 31 Sales]
[Robinson 34 35 Marketing]
[Robinson 34 34 Clerical]
[Robinson 34 33 Engineering]
[Robinson 34 31 Sales]
[Smith 34 35 Marketing]
[Smith 34 34 Clerical]
[Smith 34 33 Engineering]
[Smith 34 31 Sales]

-- 156
BEGIN TRANSACTION;
	CREATE TABLE t (c1 int32, c2 string);
	INSERT INTO t VALUES (1, "a");
	INSERT INTO t VALUES (int64(2), "b");
COMMIT;
SELECT c2 FROM t;
||cannot .*int64.*int32 .* c1

-- 157
BEGIN TRANSACTION;
	CREATE TABLE t (c1 complex64);
	INSERT INTO t VALUES(1);
COMMIT;
SELECT * FROM t;
|cc1
[(1+0i)]

-- 158
BEGIN TRANSACTION;
	CREATE TABLE p (p bool);
	INSERT INTO p VALUES (NULL), (false), (true);
COMMIT;
SELECT * FROM p;
|bp
[true]
[false]
[<nil>]

-- 159
BEGIN TRANSACTION;
	CREATE TABLE p (p bool);
	INSERT INTO p VALUES (NULL), (false), (true);
COMMIT;
SELECT p.p AS p, q.p AS q, p.p &oror; q.p AS p_or_q, p.p && q.p aS p_and_q FROM p, p AS q;
|bp, bq, bp_or_q, bp_and_q
[true true true true]
[true false true false]
[true <nil> true <nil>]
[false true true false]
[false false false false]
[false <nil> <nil> false]
[<nil> true true <nil>]
[<nil> false <nil> false]
[<nil> <nil> <nil> <nil>]

-- 160
BEGIN TRANSACTION;
	CREATE TABLE p (p bool);
	INSERT INTO p VALUES (NULL), (false), (true);
COMMIT;
SELECT p, !p AS not_p FROM p;
|bp, bnot_p
[true false]
[false true]
[<nil> <nil>]

-- S 161
SELECT * FROM department WHERE DepartmentID >= 33
ORDER BY DepartmentID;
|lDepartmentID, sDepartmentName
[33 Engineering]
[34 Clerical]
[35 Marketing]

-- S 162
SELECT * FROM department WHERE DepartmentID <= 34
ORDER BY DepartmentID;
|lDepartmentID, sDepartmentName
[31 Sales]
[33 Engineering]
[34 Clerical]

-- S 163
SELECT * FROM department WHERE DepartmentID < 34
ORDER BY DepartmentID;
|lDepartmentID, sDepartmentName
[31 Sales]
[33 Engineering]

-- S 164
SELECT +DepartmentID FROM employee;
|?
[<nil>]
[34]
[34]
[33]
[33]
[31]

-- S 165
SELECT * FROM employee
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 33]
[John <nil>]
[Jones 33]
[Rafferty 31]
[Robinson 34]
[Smith 34]

-- S 166
SELECT *
FROM employee
ORDER BY LastName DESC;
|sLastName, lDepartmentID
[Smith 34]
[Robinson 34]
[Rafferty 31]
[Jones 33]
[John <nil>]
[Heisenberg 33]

-- S 167
SELECT 1023+DepartmentID AS y FROM employee
ORDER BY y DESC;
|ly
[1057]
[1057]
[1056]
[1056]
[1054]
[<nil>]

-- S 168
SELECT +DepartmentID AS y FROM employee
ORDER BY y DESC;
|ly
[34]
[34]
[33]
[33]
[31]
[<nil>]

-- S 169
SELECT * FROM employee ORDER BY DepartmentID, LastName DESC;
|sLastName, lDepartmentID
[Smith 34]
[Robinson 34]
[Jones 33]
[Heisenberg 33]
[Rafferty 31]
[John <nil>]

-- S 170
SELECT * FROM employee ORDER BY 0+DepartmentID DESC;
|sLastName, lDepartmentID
[Robinson 34]
[Smith 34]
[Jones 33]
[Heisenberg 33]
[Rafferty 31]
[John <nil>]

-- S 171
SELECT * FROM employee ORDER BY +DepartmentID DESC;
|sLastName, lDepartmentID
[Robinson 34]
[Smith 34]
[Jones 33]
[Heisenberg 33]
[Rafferty 31]
[John <nil>]

-- S 172
SELECT ^DepartmentID AS y FROM employee
ORDER BY y DESC;
|ly
[-32]
[-34]
[-34]
[-35]
[-35]
[<nil>]

-- S 173
SELECT ^byte(DepartmentID) AS y FROM employee ORDER BY y DESC;
|uy
[224]
[222]
[222]
[221]
[221]
[<nil>]

-- 174
BEGIN TRANSACTION;
	CREATE TABLE t (r RUNE);
	INSERT INTO t VALUES (1), ('A'), (rune(int(0x21)));
COMMIT;
SELECT * FROM t
ORDER BY r;
|kr
[1]
[33]
[65]

-- 175
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (-2), (-1), (0), (1), (2);
COMMIT;
SELECT i^1 AS y FROM t
ORDER by y;
|ly
[-2]
[-1]
[0]
[1]
[3]

-- 176
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (-2), (-1), (0), (1), (2);
COMMIT;
SELECT i&or;1 AS y FROM t
ORDER BY y;
|ly
[-1]
[-1]
[1]
[1]
[3]

-- 177
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (-2), (-1), (0), (1), (2);
COMMIT;
SELECT i&1 FROM t;
|l
[0]
[1]
[0]
[1]
[0]

-- 178
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (-2), (-1), (0), (1), (2);
COMMIT;
SELECT i&^1 AS y FROM t
ORDER BY y;
|ly
[-2]
[-2]
[0]
[0]
[2]

-- S 179
SELECT * from employee WHERE LastName == "Jones" &oror; DepartmentID IS NULL
ORDER by LastName DESC;
|sLastName, lDepartmentID
[Jones 33]
[John <nil>]

-- S 180
SELECT * from employee WHERE LastName != "Jones" && DepartmentID IS NOT NULL
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 33]
[Rafferty 31]
[Robinson 34]
[Smith 34]

-- 181
SELECT 42[0] FROM foo;
||invalid operation.*index of type

-- 182
SELECT "foo"[-1] FROM foo;
||invalid string index.*index .* non.*negative

-- 183
SELECT "foo"[3] FROM foo;
||invalid string index.*out of bounds

-- 184
SELECT "foo"["bar">true] FROM foo;
||mismatched type

-- S 185
SELECT DepartmentID[0] FROM employee;
||run.time error.*invalid operation.*index of type

-- S 186
SELECT "foo"[-DepartmentID] FROM employee;
||run.time error.*invalid string index.*index .* non.*negative

-- S 187
SELECT LastName[100] FROM employee;
||run.time.error.*invalid string index.*out of bounds

-- S 188
SELECT LastName[0], LastName FROM employee ORDER BY LastName;
|u, sLastName
[72 Heisenberg]
[74 John]
[74 Jones]
[82 Rafferty]
[82 Robinson]
[83 Smith]

-- S 189
SELECT LastName, string(LastName[0]), string(LastName[1]), string(LastName[2]), string(LastName[3])
FROM employee
ORDER BY LastName;
|sLastName, s, s, s, s
[Heisenberg H e i s]
[John J o h n]
[Jones J o n e]
[Rafferty R a f f]
[Robinson R o b i]
[Smith S m i t]

-- S 190
SELECT LastName, LastName[:], LastName[:2], LastName[2:], LastName[1:3]
FROM employee
ORDER by LastName;
|sLastName, s, s, s, s
[Heisenberg Heisenberg He isenberg ei]
[John John Jo hn oh]
[Jones Jones Jo nes on]
[Rafferty Rafferty Ra fferty af]
[Robinson Robinson Ro binson ob]
[Smith Smith Sm ith mi]

-- S 191
SELECT LastName
FROM employee
WHERE department IS NULL;
||unknown field department

-- S 192
SELECT
	DepartmentID,
	LastName,
	LastName[:4],
	LastName[:0*DepartmentID],
	LastName[0*DepartmentID:0],
	LastName[0*DepartmentID:0*DepartmentID],
FROM
	employee,
ORDER BY LastName DESC;
|lDepartmentID, sLastName, s, s, s, s
[34 Smith Smit   ]
[34 Robinson Robi   ]
[31 Rafferty Raff   ]
[33 Jones Jone   ]
[<nil> John John <nil> <nil> <nil>]
[33 Heisenberg Heis   ]

-- S 193
SELECT
	DepartmentID AS x,
	DepartmentID<<1 AS a,
	1<<uint(DepartmentID) AS b,
FROM
	employee,
WHERE DepartmentID IS NOT NULL
ORDER BY x;
|lx, la, lb
[31 62 2147483648]
[33 66 8589934592]
[33 66 8589934592]
[34 68 17179869184]
[34 68 17179869184]

-- S 194
SELECT
	DepartmentID AS x,
	DepartmentID>>1 AS a,
	uint(1)<<63>>uint(DepartmentID) AS b,
FROM
	employee,
WHERE DepartmentID IS NOT NULL
ORDER BY x;
|lx, la, xb
[31 15 4294967296]
[33 16 1073741824]
[33 16 1073741824]
[34 17 536870912]
[34 17 536870912]

-- S 195
SELECT DISTINCT DepartmentID
FROM employee
WHERE DepartmentID IS NOT NULL;
|lDepartmentID
[31]
[33]
[34]

-- S 196
SELECT DISTINCT e.DepartmentID, d.DepartmentID, e.LastName
FROM employee AS e, department AS d
WHERE e.DepartmentID == d.DepartmentID;
|le.DepartmentID, ld.DepartmentID, se.LastName
[31 31 Rafferty]
[33 33 Heisenberg]
[33 33 Jones]
[34 34 Robinson]
[34 34 Smith]

-- S 197
SELECT DISTINCT e.DepartmentID, d.DepartmentID, e.LastName
FROM employee AS e, department AS d
WHERE e.DepartmentID == d.DepartmentID
ORDER BY e.LastName;
|le.DepartmentID, ld.DepartmentID, se.LastName
[33 33 Heisenberg]
[33 33 Jones]
[31 31 Rafferty]
[34 34 Robinson]
[34 34 Smith]

-- S 198, http://en.wikipedia.org/wiki/Join_(SQL)#Cross_join
SELECT *
FROM employee, department
ORDER BY employee.LastName, department.DepartmentID;
|semployee.LastName, lemployee.DepartmentID, ldepartment.DepartmentID, sdepartment.DepartmentName
[Heisenberg 33 31 Sales]
[Heisenberg 33 33 Engineering]
[Heisenberg 33 34 Clerical]
[Heisenberg 33 35 Marketing]
[John <nil> 31 Sales]
[John <nil> 33 Engineering]
[John <nil> 34 Clerical]
[John <nil> 35 Marketing]
[Jones 33 31 Sales]
[Jones 33 33 Engineering]
[Jones 33 34 Clerical]
[Jones 33 35 Marketing]
[Rafferty 31 31 Sales]
[Rafferty 31 33 Engineering]
[Rafferty 31 34 Clerical]
[Rafferty 31 35 Marketing]
[Robinson 34 31 Sales]
[Robinson 34 33 Engineering]
[Robinson 34 34 Clerical]
[Robinson 34 35 Marketing]
[Smith 34 31 Sales]
[Smith 34 33 Engineering]
[Smith 34 34 Clerical]
[Smith 34 35 Marketing]

-- S 199, http://en.wikipedia.org/wiki/Join_(SQL)#Inner_join
SELECT *
FROM employee, department
WHERE employee.DepartmentID == department.DepartmentID
ORDER BY employee.LastName, department.DepartmentID;
|semployee.LastName, lemployee.DepartmentID, ldepartment.DepartmentID, sdepartment.DepartmentName
[Heisenberg 33 33 Engineering]
[Jones 33 33 Engineering]
[Rafferty 31 31 Sales]
[Robinson 34 34 Clerical]
[Smith 34 34 Clerical]

-- S 200
BEGIN TRANSACTION;
	INSERT INTO department (DepartmentID, DepartmentName)
	SELECT DepartmentID+1000, DepartmentName+"/headquarters"
	FROM department;
COMMIT;
SELECT * FROM department
ORDER BY DepartmentID;
|lDepartmentID, sDepartmentName
[31 Sales]
[33 Engineering]
[34 Clerical]
[35 Marketing]
[1031 Sales/headquarters]
[1033 Engineering/headquarters]
[1034 Clerical/headquarters]
[1035 Marketing/headquarters]

-- S 201`
BEGIN TRANSACTION;
	INSERT INTO department (DepartmentName, DepartmentID)
	SELECT DepartmentName+"/headquarters", DepartmentID+1000
	FROM department;
COMMIT;
SELECT * FROM department
ORDER BY DepartmentID;
|lDepartmentID, sDepartmentName
[31 Sales]
[33 Engineering]
[34 Clerical]
[35 Marketing]
[1031 Sales/headquarters]
[1033 Engineering/headquarters]
[1034 Clerical/headquarters]
[1035 Marketing/headquarters]

-- S 202
BEGIN TRANSACTION;
	DELETE FROM department;
COMMIT;
SELECT * FROM department
|?DepartmentID, ?DepartmentName

-- S 203
BEGIN TRANSACTION;
	DELETE FROM department
	WHERE DepartmentID == 35 &oror; DepartmentName != "" && DepartmentName[0] == 'C';
COMMIT;
SELECT * FROM department
ORDER BY DepartmentID;
|lDepartmentID, sDepartmentName
[31 Sales]
[33 Engineering]

-- S 204
SELECT id(), LastName
FROM employee
ORDER BY id();
|l, sLastName
[5 Rafferty]
[6 Jones]
[7 Heisenberg]
[8 Robinson]
[9 Smith]
[10 John]

-- S 205
BEGIN TRANSACTION;
	DELETE FROM employee
	WHERE LastName == "Jones";
COMMIT;
SELECT id(), LastName
FROM employee
ORDER BY id();
|l, sLastName
[5 Rafferty]
[7 Heisenberg]
[8 Robinson]
[9 Smith]
[10 John]

-- S 206
BEGIN TRANSACTION;
	DELETE FROM employee
	WHERE LastName == "Jones";
	INSERT INTO employee (LastName) VALUES ("Jones");
COMMIT;
SELECT id(), LastName
FROM employee
ORDER BY id();
|l, sLastName
[5 Rafferty]
[7 Heisenberg]
[8 Robinson]
[9 Smith]
[10 John]
[11 Jones]

-- S 207
SELECT id(), e.LastName, e.DepartmentID, d.DepartmentID
FROM
	employee AS e,
	department AS d,
WHERE e.DepartmentID == d.DepartmentID
ORDER BY e.LastName;
|?, se.LastName, le.DepartmentID, ld.DepartmentID
[<nil> Heisenberg 33 33]
[<nil> Jones 33 33]
[<nil> Rafferty 31 31]
[<nil> Robinson 34 34]
[<nil> Smith 34 34]

-- S 208
SELECT e.ID, e.LastName, e.DepartmentID, d.DepartmentID
FROM
	(SELECT id() AS ID, LastName, DepartmentID FROM employee;) AS e,
	department AS d,
WHERE e.DepartmentID == d.DepartmentID
ORDER BY e.ID;
|le.ID, se.LastName, le.DepartmentID, ld.DepartmentID
[5 Rafferty 31 31]
[6 Jones 33 33]
[7 Heisenberg 33 33]
[8 Robinson 34 34]
[9 Smith 34 34]

-- S 209
BEGIN TRANSACTION;
	UPDATE none
		DepartmentID = DepartmentID+1000,
	WHERE DepartmentID == 33;
COMMIT;
SELECT * FROM employee;
||table.*not.*exist

-- S 210
BEGIN TRANSACTION;
	UPDATE employee
		FirstName = "John"
	WHERE DepartmentID == 33;
COMMIT;
SELECT * FROM employee;
||unknown.*FirstName

-- S 211
BEGIN TRANSACTION;
	UPDATE employee
		DepartmentID = DepartmentID+1000,
	WHERE DepartmentID == 33;
COMMIT;
SELECT * FROM employee
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 1033]
[John <nil>]
[Jones 1033]
[Rafferty 31]
[Robinson 34]
[Smith 34]

-- S 212
BEGIN TRANSACTION;
	UPDATE employee
		DepartmentID = DepartmentID+1000,
		LastName = "Mr. "+LastName
	WHERE id() == 7;
COMMIT;
SELECT * FROM employee
ORDER BY LastName DESC;
|sLastName, lDepartmentID
[Smith 34]
[Robinson 34]
[Rafferty 31]
[Mr. Heisenberg 1033]
[Jones 33]
[John <nil>]

-- S 213
BEGIN TRANSACTION;
	UPDATE employee
		LastName = "Mr. "+LastName,
		DepartmentID = DepartmentID+1000,
	WHERE id() == 7;
COMMIT;
SELECT * FROM employee
ORDER BY LastName DESC;
|sLastName, lDepartmentID
[Smith 34]
[Robinson 34]
[Rafferty 31]
[Mr. Heisenberg 1033]
[Jones 33]
[John <nil>]

-- S 214
BEGIN TRANSACTION;
	UPDATE employee
		DepartmentID = DepartmentID+1000;
COMMIT;
SELECT * FROM employee
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 1033]
[John <nil>]
[Jones 1033]
[Rafferty 1031]
[Robinson 1034]
[Smith 1034]

-- S 215
BEGIN TRANSACTION;
	UPDATE employee
		DepartmentId = DepartmentID+1000;
COMMIT;
SELECT * FROM employee;
||unknown

-- S 216
BEGIN TRANSACTION;
	UPDATE employee
		DepartmentID = DepartmentId+1000;
COMMIT;
SELECT * FROM employee;
||unknown

-- S 217
BEGIN TRANSACTION;
	UPDATE employee
		DepartmentID = "foo";
COMMIT;
SELECT * FROM employee;
||cannot .* string.*int64 .* DepartmentID

-- S 218
SELECT foo[len()] FROM bar;
||missing argument

-- S 219
SELECT foo[len(42)] FROM bar;
||invalid argument

-- S 220
SELECT foo[len(42, 24)] FROM bar;
||too many

-- S 221
SELECT foo[len("baz")] FROM bar;
||table

-- S 222
SELECT LastName[len("baz")-4] FROM employee;
||invalid string index

-- S 223
SELECT LastName[:len(LastName)-3] AS y FROM employee
ORDER BY y;
|sy
[Heisenb]
[J]
[Jo]
[Raffe]
[Robin]
[Sm]

-- S 224
SELECT complex(float32(DepartmentID+int(id())), 0) AS x, complex(DepartmentID+int(id()), 0)
FROM employee
ORDER by real(x) DESC;
|cx, d
[(43+0i) (43+0i)]
[(42+0i) (42+0i)]
[(40+0i) (40+0i)]
[(39+0i) (39+0i)]
[(36+0i) (36+0i)]
[<nil> <nil>]

-- S 225
SELECT real(complex(float32(DepartmentID+int(id())), 0)) AS x, real(complex(DepartmentID+int(id()), 0))
FROM employee
ORDER BY x DESC;
|fx, g
[43 43]
[42 42]
[40 40]
[39 39]
[36 36]
[<nil> <nil>]

-- S 226
SELECT imag(complex(0, float32(DepartmentID+int(id())))) AS x, imag(complex(0, DepartmentID+int(id())))
FROM employee
ORDER BY x DESC;
|fx, g
[43 43]
[42 42]
[40 40]
[39 39]
[36 36]
[<nil> <nil>]

-- 227
BEGIN TRANSACTION;
	CREATE TABLE t (c string);
	INSERT INTO t VALUES("foo"), ("bar");
	DELETE FROM t WHERE c == "foo";
COMMIT;
SELECT 100*id(), c FROM t;
|l, sc
[200 bar]

-- 228
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE a;
COMMIT;
SELECT * FROM a;
||table a does not exist

-- 229
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE a;
COMMIT;
SELECT * FROM b;
|?b

-- 230
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE a;
COMMIT;
SELECT * FROM c;
|?c

-- 231
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE b;
COMMIT;
SELECT * FROM a;
|?a

-- 232
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE b;
COMMIT;
SELECT * FROM b;
||table b does not exist

-- 233
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE b;
COMMIT;
SELECT * FROM c;
|?c

-- 234
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE c;
COMMIT;
SELECT * FROM a;
|?a

-- 235
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE c;
COMMIT;
SELECT * FROM b;
|?b

-- 236
BEGIN TRANSACTION;
	CREATE TABLE a (a int);
	CREATE TABLE b (b int);
	CREATE TABLE c (c int);
	DROP TABLE c;
COMMIT;
SELECT * FROM c;
||table c does not exist

-- 237
BEGIN TRANSACTION;
	CREATE TABLE a (c int);
	INSERT INTO a VALUES (10), (11), (12);
	CREATE TABLE b (d int);
	INSERT INTO b VALUES (20), (21), (22), (23);
COMMIT;
SELECT * FROM a, b;
|la.c, lb.d
[12 23]
[12 22]
[12 21]
[12 20]
[11 23]
[11 22]
[11 21]
[11 20]
[10 23]
[10 22]
[10 21]
[10 20]

-- 238
BEGIN TRANSACTION;
	CREATE TABLE a (c int);
	INSERT INTO a VALUES (0), (1), (2);
COMMIT;
SELECT
	9*x2.c AS x2,
	3*x1.c AS x1,
	1*x0.c AS x0,
	9*x2.c + 3*x1.c + x0.c AS y,
FROM
	a AS x2,
	a AS x1,
	a AS x0,
ORDER BY y;
|lx2, lx1, lx0, ly
[0 0 0 0]
[0 0 1 1]
[0 0 2 2]
[0 3 0 3]
[0 3 1 4]
[0 3 2 5]
[0 6 0 6]
[0 6 1 7]
[0 6 2 8]
[9 0 0 9]
[9 0 1 10]
[9 0 2 11]
[9 3 0 12]
[9 3 1 13]
[9 3 2 14]
[9 6 0 15]
[9 6 1 16]
[9 6 2 17]
[18 0 0 18]
[18 0 1 19]
[18 0 2 20]
[18 3 0 21]
[18 3 1 22]
[18 3 2 23]
[18 6 0 24]
[18 6 1 25]
[18 6 2 26]

-- 239
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	INSERT INTO t VALUES (242);
	DELETE FROM t WHERE c != 0;
COMMIT;
SELECT * FROM t
|?c

-- 240
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (242), (12, 24);
COMMIT;
||expect

-- 241
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (24, 2), (1224);
COMMIT;
||expect

-- 242
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
ROLLBACK;
SELECT * from t;
||does not exist

-- 243
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
BEGIN TRANSACTION;
	DROP TABLE T;
COMMIT;
SELECT * from t;
||does not exist

-- 244
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
BEGIN TRANSACTION;
	DROP TABLE t;
ROLLBACK;
SELECT * from t;
|?i

-- 245
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (int(1.2));
COMMIT;
SELECT * FROM t;
||truncated

-- 246
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (string(65.0));
COMMIT;
SELECT * FROM t;
||cannot convert

-- 247
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES (string(65));
COMMIT;
SELECT * FROM t;
|ss
[A]

-- 248
BEGIN TRANSACTION;
	CREATE TABLE t (i uint32);
	INSERT INTO t VALUES (uint32(int8(uint16(0x10F0))));
COMMIT;
SELECT i == 0xFFFFFFF0 FROM t;
|b
[true]

-- 249
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		(string('a')),		// "a"
		(string(-1)),		// "\ufffd" == "\xef\xbf\xbd"
		(string(0xf8)),		// "\u00f8" == "ø" == "\xc3\xb8"
		(string(0x65e5)),	// "\u65e5" == "日" == "\xe6\x97\xa5"
	;
COMMIT;
SELECT
	id() == 1 && s == "a" &oror;
	id() == 2 && s == "\ufffd" && s == "\xef\xbf\xbd" &oror;
	id() == 3 && s == "\u00f8" && s == "ø" && s == "\xc3\xb8" &oror;
	id() == 4 && s == "\u65e5" && s == "日" && s == "\xe6\x97\xa5"
FROM t;
|b
[true]
[true]
[true]
[true]

-- 250
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (0);
COMMIT;
SELECT 2.3+1, 1+2.3 FROM t;
|f, f
[3.3 3.3]

-- 251
BEGIN TRANSACTION;
	CREATE TABLE t (i byte);
	INSERT INTO t VALUES (-1+byte(2));
COMMIT;
SELECT * FROM t;
||mismatched

-- 252
BEGIN TRANSACTION;
	CREATE TABLE t (i byte);
	INSERT INTO t VALUES (1+byte(2));
COMMIT;
SELECT * FROM t;
|ui
[3]

-- 253
BEGIN TRANSACTION;
	CREATE TABLE t (i byte);
	INSERT INTO t VALUES (255+byte(2));
COMMIT;
SELECT * FROM t;
|ui
[1]

-- 254
BEGIN TRANSACTION;
	CREATE TABLE t (i byte);
	INSERT INTO t VALUES (256+byte(2));
COMMIT;
SELECT * FROM t;
||mismatched

-- 255
BEGIN TRANSACTION;
	CREATE TABLE t (i int8);
	INSERT INTO t VALUES (127+int8(2));
COMMIT;
SELECT * FROM t;
|ii
[-127]

-- 256
BEGIN TRANSACTION;
	CREATE TABLE t (i int8);
	INSERT INTO t VALUES (-129+int8(2));
COMMIT;
SELECT * FROM t;
||mismatched

-- 257
BEGIN TRANSACTION;
	CREATE TABLE t (i int8);
	INSERT INTO t VALUES (-128+int8(2));
COMMIT;
SELECT * FROM t;
|ii
[-126]

-- 258
BEGIN TRANSACTION;
	CREATE TABLE t (i int8);
	INSERT INTO t VALUES (128+int8(2));
COMMIT;
SELECT * FROM t;
||mismatched

-- S 259
SELECT count(none) FROM employee;
||unknown

-- S 260
SELECT count() FROM employee;
|l
[6]

-- S 261
SELECT count() AS y FROM employee;
|ly
[6]

-- S 262
SELECT 3*count() AS y FROM employee;
|ly
[18]

-- S 263
SELECT count(LastName) FROM employee;
|l
[6]

-- S 264
SELECT count(DepartmentID) FROM employee;
|l
[5]

-- S 265
SELECT count() - count(DepartmentID) FROM employee;
|l
[1]

-- S 266
SELECT min(LastName), min(DepartmentID) FROM employee;
|s, l
[Heisenberg 31]

-- S 267
SELECT max(LastName), max(DepartmentID) FROM employee;
|s, l
[Smith 34]

-- S 268
SELECT sum(LastName), sum(DepartmentID) FROM employee;
||cannot

-- S 269
SELECT sum(DepartmentID) FROM employee;
|l
[165]

-- S 270
SELECT avg(DepartmentID) FROM employee;
|l
[33]

-- S 271
SELECT DepartmentID FROM employee GROUP BY none;
||unknown

-- S 272
SELECT DepartmentID, sum(DepartmentID) AS s FROM employee GROUP BY DepartmentID ORDER BY s DESC;
|lDepartmentID, ls
[34 68]
[33 66]
[31 31]
[<nil> <nil>]

-- S 273
SELECT DepartmentID, count(LastName+string(DepartmentID)) AS y FROM employee GROUP BY DepartmentID ORDER BY y DESC ;
|lDepartmentID, ly
[34 2]
[33 2]
[31 1]
[<nil> 0]

-- S 274
SELECT DepartmentID, sum(2*DepartmentID) AS s FROM employee GROUP BY DepartmentID ORDER BY s DESC;
|lDepartmentID, ls
[34 136]
[33 132]
[31 62]
[<nil> <nil>]

-- S 275
SELECT min(2*DepartmentID) FROM employee;
|l
[62]

-- S 276
SELECT max(2*DepartmentID) FROM employee;
|l
[68]

-- S 277
SELECT avg(2*DepartmentID) FROM employee;
|l
[66]

-- S 278
SELECT * FROM employee GROUP BY DepartmentID;
|sLastName, ?DepartmentID
[John <nil>]
[Rafferty 31]
[Heisenberg 33]
[Smith 34]

-- S 279
SELECT * FROM employee GROUP BY DepartmentID ORDER BY LastName DESC;
|sLastName, lDepartmentID
[Smith 34]
[Rafferty 31]
[John <nil>]
[Heisenberg 33]

-- S 280
SELECT * FROM employee GROUP BY DepartmentID, LastName ORDER BY LastName DESC;
|sLastName, lDepartmentID
[Smith 34]
[Robinson 34]
[Rafferty 31]
[Jones 33]
[John <nil>]
[Heisenberg 33]

-- S 281
SELECT * FROM employee GROUP BY LastName, DepartmentID  ORDER BY LastName DESC;
|sLastName, lDepartmentID
[Smith 34]
[Robinson 34]
[Rafferty 31]
[Jones 33]
[John <nil>]
[Heisenberg 33]

-- 282
BEGIN TRANSACTION;
	CREATE TABLE s (i int);
	CREATE TABLE t (i int);
COMMIT;
BEGIN TRANSACTION;
	DROP TABLE s;
COMMIT;
SELECT * FROM t;
|?i

-- 283
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
COMMIT;
SELECT count() FROM t;
|l
[0]

-- 284
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT count() FROM t;
|l
[2]

-- 285
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT count() FROM t WHERE n < 2;
|l
[2]

-- 286
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT count() FROM t WHERE n < 1;
|l
[1]

-- 287
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT count() FROM t WHERE n < 0;
|l
[0]

-- 288
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT s+10 FROM (SELECT sum(n) AS s FROM t WHERE n < 2);
|l
[11]

-- 289
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT s+10 FROM (SELECT sum(n) AS s FROM t WHERE n < 1);
|l
[10]

-- 290
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT s+10 FROM (SELECT sum(n) AS s FROM t WHERE n < 0);
|?
[<nil>]

-- 291
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT sum(n) AS s FROM t WHERE n < 2;
|ls
[1]

-- 292
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT sum(n) AS s FROM t WHERE n < 1;
|ls
[0]

-- 293
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1);
COMMIT;
SELECT sum(n) AS s FROM t WHERE n < 0;
|?s
[<nil>]

-- 294
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t SELECT count() FROM t;
	INSERT INTO t SELECT count() FROM t;
	INSERT INTO t SELECT count() FROM t;
COMMIT;
SELECT count() FROM t;
|l
[3]

-- 295
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t SELECT count() FROM t;
	INSERT INTO t SELECT count() FROM t;
	INSERT INTO t SELECT count() FROM t;
	INSERT INTO t SELECT * FROM t;
COMMIT;
SELECT count() FROM t;
|l
[6]

-- 296
BEGIN TRANSACTION;
	CREATE TABLE t (n int);
	INSERT INTO t VALUES (0), (1), (2);
	INSERT INTO t SELECT * FROM t;
COMMIT;
SELECT count() FROM t;
|l
[6]

-- 297
BEGIN TRANSACTION;
	CREATE TABLE t(S string);
	INSERT INTO t SELECT "perfect!" FROM (SELECT count() AS cnt FROM t WHERE S == "perfect!") WHERE cnt == 0;
COMMIT;
SELECT count() FROM t;
|l
[1]

-- 298
BEGIN TRANSACTION;
	CREATE TABLE t(S string);
	INSERT INTO t SELECT "perfect!" FROM (SELECT count() AS cnt FROM t WHERE S == "perfect!") WHERE cnt == 0;
	INSERT INTO t SELECT "perfect!" FROM (SELECT count() AS cnt FROM t WHERE S == "perfect!") WHERE cnt == 0;
COMMIT;
SELECT count() FROM t;
|l
[1]

-- 299
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob("a"));
COMMIT;
SELECT * FROM t;
|?c
[[97]]

-- 300
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob(`
0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
`));
COMMIT;
SELECT * FROM t;
|?c
[[10 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 10]]

-- 301
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob(
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF"));
COMMIT;
SELECT * FROM t;
|?c
[[48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 65 66 67 68 69 70]]

-- 302
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob(
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF"+
"!"));
COMMIT;
SELECT * FROM t;
|?c
[[48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 65 66 67 68 69 70 33]]

-- 303
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob("hell\xc3\xb8"));
COMMIT;
SELECT string(c) FROM t;
|s
[hellø]

-- 304
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob(
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF"+
"!"));
COMMIT;
SELECT string(c) FROM t;
|s
[0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]

-- 305
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob(""));
COMMIT;
SELECT string(c) FROM t;
|s
[]

-- 306
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob("hellø"));
COMMIT;
SELECT * FROM t;
|?c
[[104 101 108 108 195 184]]

-- 307
BEGIN TRANSACTION;
	CREATE TABLE t(c blob);
	INSERT INTO t VALUES (blob(""));
COMMIT;
SELECT * FROM t;
|?c
[[]]

-- 308
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
	;
COMMIT;
SELECT * FROM t;
|li, ?b
[0 [48]]

-- 309
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
		(1, blob("1")),
	;
COMMIT;
SELECT * FROM t;
|li, ?b
[1 [49]]
[0 [48]]

-- 310
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
		(1, blob("1")),
		(2, blob("2")),
	;
COMMIT;
SELECT * FROM t;
|li, ?b
[2 [50]]
[1 [49]]
[0 [48]]

-- 311
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
	;
	DELETE FROM t WHERE i == 0;
COMMIT;
SELECT * FROM t;
|?i, ?b

-- 312
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
		(1, blob("1")),
	;
	DELETE FROM t WHERE i == 0;
COMMIT;
SELECT * FROM t;
|li, ?b
[1 [49]]

-- 313
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
		(1, blob("1")),
	;
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t;
|li, ?b
[0 [48]]

-- 314
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
		(1, blob("1")),
		(2, blob("2")),
	;
	DELETE FROM t WHERE i == 0;
COMMIT;
SELECT * FROM t;
|li, ?b
[2 [50]]
[1 [49]]

-- 315
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
		(1, blob("1")),
		(2, blob("2")),
	;
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t;
|li, ?b
[2 [50]]
[0 [48]]

-- 316
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob("0")),
		(1, blob("1")),
		(2, blob("2")),
	;
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT * FROM t;
|li, ?b
[1 [49]]
[0 [48]]

-- 317
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
	;
	DELETE FROM t WHERE i == 0;
COMMIT;
SELECT i, string(b) FROM t;
|?i, ?

-- 318
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(1, blob(
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
	;
	DELETE FROM t WHERE i == 0;
COMMIT;
SELECT i, string(b) FROM t;
|li, s
[1 1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]

-- 319
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(1, blob(
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
	;
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT i, string(b) FROM t;
|li, s
[0 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]

-- 320
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(1, blob(
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(2, blob(
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
	;
	DELETE FROM t WHERE i == 0;
COMMIT;
SELECT i, string(b) FROM t;
|li, s
[2 2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef2123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]
[1 1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]

-- 321
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(1, blob(
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(2, blob(
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
	;
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT i, string(b) FROM t;
|li, s
[2 2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef2123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]
[0 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]

-- 322
BEGIN TRANSACTION;
	CREATE TABLE t(i int, b blob);
	INSERT INTO t VALUES
		(0, blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(1, blob(
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
		(2, blob(
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!")),
	;
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT i, string(b) FROM t;
|li, s
[1 1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]
[0 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!]

-- 323
BEGIN TRANSACTION;
	CREATE TABLE t (c bool);
	INSERT INTO t VALUES (false), (true);
COMMIT;
SELECT * FROM t ORDER BY true, c, false;
||cannot .* bool

-- 324
BEGIN TRANSACTION;
	CREATE TABLE t (c bool, i int);
	INSERT INTO t VALUES (false, 1), (true, 2), (false, 10), (true, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|bc, l
[false 11]
[true 22]

-- 325
BEGIN TRANSACTION;
	CREATE TABLE t (c int8);
	INSERT INTO t VALUES (1), (2);
COMMIT;
SELECT * FROM t ORDER BY 42, c, 24;
|ic
[1]
[2]

-- 326
BEGIN TRANSACTION;
	CREATE TABLE t (c int8, i int);
	INSERT INTO t VALUES (99, 1), (100, 2), (99, 10), (100, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|ic, l
[99 11]
[100 22]

-- 327
BEGIN TRANSACTION;
	CREATE TABLE t (c blob);
	INSERT INTO t VALUES (blob("A")), (blob("B"));
COMMIT;
SELECT * FROM t ORDER BY 42, c, 24;
||cannot .* \[\]uint8

-- 328
BEGIN TRANSACTION;
	CREATE TABLE t (c blob, i int);
	INSERT INTO t VALUES (blob("A"), 1), (blob("B"), 2);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|?c, l
[[65] 1]
[[66] 2]

-- 329
BEGIN TRANSACTION;
	CREATE TABLE t (c blob, i int);
	INSERT INTO t VALUES (blob("A"), 10), (blob("B"), 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|?c, l
[[65] 10]
[[66] 20]

-- 330
BEGIN TRANSACTION;
	CREATE TABLE t (c blob, i int);
	INSERT INTO t VALUES (blob("A"), 1), (blob("B"), 2), (blob("A"), 10), (blob("B"), 20);
COMMIT;
SELECT * FROM t;
|?c, li
[[66] 20]
[[65] 10]
[[66] 2]
[[65] 1]

-- 331
BEGIN TRANSACTION;
	CREATE TABLE t (c string, i int);
	INSERT INTO t VALUES ("A", 1), ("B", 2), ("A", 10), ("B", 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|sc, l
[A 11]
[B 22]

-- 332
BEGIN TRANSACTION;
	CREATE TABLE t (c blob, i int);
	INSERT INTO t VALUES (blob("A"), 1), (blob("B"), 2), (blob("A"), 10), (blob("B"), 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|?c, l
[[65] 11]
[[66] 22]

-- 333
BEGIN TRANSACTION;
	CREATE TABLE t (c byte);
	INSERT INTO t VALUES (42), (314);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
||overflow

-- 334
BEGIN TRANSACTION;
	CREATE TABLE t (c byte);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|uc
[42]
[114]

-- 335
BEGIN TRANSACTION;
	CREATE TABLE t (c byte, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|uc, l
[100 11]
[101 22]

-- 336
BEGIN TRANSACTION;
	CREATE TABLE t (c byte);
	INSERT INTO t VALUES (42), (3.14);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
||truncated

-- 337
BEGIN TRANSACTION;
	CREATE TABLE t (c complex64);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
||cannot order by

-- 338
BEGIN TRANSACTION;
	CREATE TABLE t (c complex64, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|cc, l
[(100+0i) 11]
[(101+0i) 22]

-- 339
BEGIN TRANSACTION;
	CREATE TABLE t (c complex128);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
||cannot order by

-- 340
BEGIN TRANSACTION;
	CREATE TABLE t (c complex128, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|dc, l
[(100+0i) 11]
[(101+0i) 22]

-- 341
BEGIN TRANSACTION;
	CREATE TABLE t (c float);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|gc
[42]
[114]

-- 342
BEGIN TRANSACTION;
	CREATE TABLE t (c float, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|gc, l
[100 11]
[101 22]

-- 343
BEGIN TRANSACTION;
	CREATE TABLE t (c float64);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|gc
[42]
[114]

-- 344
BEGIN TRANSACTION;
	CREATE TABLE t (c float64, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|gc, l
[100 11]
[101 22]

-- 345
BEGIN TRANSACTION;
	CREATE TABLE t (c float32);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|fc
[42]
[114]

-- 346
BEGIN TRANSACTION;
	CREATE TABLE t (c float32, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|fc, l
[100 11]
[101 22]

-- 347
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|lc
[42]
[114]

-- 348
BEGIN TRANSACTION;
	CREATE TABLE t (c int, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|lc, l
[100 11]
[101 22]

-- 349
BEGIN TRANSACTION;
	CREATE TABLE t (c int64);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|lc
[42]
[114]

-- 350
BEGIN TRANSACTION;
	CREATE TABLE t (c int64, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|lc, l
[100 11]
[101 22]

-- 351
BEGIN TRANSACTION;
	CREATE TABLE t (c int8);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|ic
[42]
[114]

-- 352
BEGIN TRANSACTION;
	CREATE TABLE t (c int8, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|ic, l
[100 11]
[101 22]

-- 353
BEGIN TRANSACTION;
	CREATE TABLE t (c int16);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|jc
[42]
[114]

-- 354
BEGIN TRANSACTION;
	CREATE TABLE t (c int16, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|jc, l
[100 11]
[101 22]

-- 355
BEGIN TRANSACTION;
	CREATE TABLE t (c int32);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|kc
[42]
[114]

-- 356
BEGIN TRANSACTION;
	CREATE TABLE t (c int32, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|kc, l
[100 11]
[101 22]

-- 357
BEGIN TRANSACTION;
	CREATE TABLE t (c uint);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|xc
[42]
[114]

-- 358
BEGIN TRANSACTION;
	CREATE TABLE t (c uint, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|xc, l
[100 11]
[101 22]

-- 359
BEGIN TRANSACTION;
	CREATE TABLE t (c uint64);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|xc
[42]
[114]

-- 360
BEGIN TRANSACTION;
	CREATE TABLE t (c uint64, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|xc, l
[100 11]
[101 22]

-- 361
BEGIN TRANSACTION;
	CREATE TABLE t (c uint8);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|uc
[42]
[114]

-- 362
BEGIN TRANSACTION;
	CREATE TABLE t (c uint8, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|uc, l
[100 11]
[101 22]

-- 363
BEGIN TRANSACTION;
	CREATE TABLE t (c uint16);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|vc
[42]
[114]

-- 364
BEGIN TRANSACTION;
	CREATE TABLE t (c uint16, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|vc, l
[100 11]
[101 22]

-- 365
BEGIN TRANSACTION;
	CREATE TABLE t (c uint32);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|wc
[42]
[114]

-- 366
BEGIN TRANSACTION;
	CREATE TABLE t (c uint32, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|wc, l
[100 11]
[101 22]

-- 367
BEGIN TRANSACTION;
	CREATE TABLE t (c blob, i int);
	INSERT INTO t VALUES (blob("A"), 1), (blob("B"), 2);
	UPDATE t c = blob("C") WHERE i == 2;
COMMIT;
SELECT * FROM t;
|?c, li
[[67] 2]
[[65] 1]

-- 368
BEGIN TRANSACTION;
	CREATE TABLE t (c blob, i int);
	INSERT INTO t VALUES
		(blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!"
		), 1),
		(blob(
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!"
		), 2);
	UPDATE t c = blob(
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"2123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!"
	) WHERE i == 2;
COMMIT;
SELECT * FROM t;
|?c, li
[[50 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 50 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 50 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 50 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 65 66 67 68 69 70 33] 2]
[[48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 65 66 67 68 69 70 33] 1]

-- 369
BEGIN TRANSACTION;
	CREATE TABLE t (c blob, i int);
	INSERT INTO t VALUES
		(blob(
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!"
		), 1),
		(blob(
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
			"1123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF!"
		), 2);
	UPDATE t i = 42 WHERE i == 2;
COMMIT;
SELECT * FROM t;
|?c, li
[[49 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 49 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 49 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 49 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 65 66 67 68 69 70 33] 42]
[[48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 97 98 99 100 101 102 48 49 50 51 52 53 54 55 56 57 65 66 67 68 69 70 33] 1]

-- 370
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * FROM t;
|?i
[1]

-- 371
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint("1"));
COMMIT;
SELECT * FROM t;
|?i
[1]

-- 372
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint("12345678901234567890123456789"));
COMMIT;
SELECT * FROM t;
|?i
[12345678901234567890123456789]

-- 373
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint(2e10));
COMMIT;
SELECT * FROM t;
|?i
[20000000000]

-- 374
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint(2e18));
COMMIT;
SELECT * FROM t;
|?i
[2000000000000000000]

-- 375
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint(2e19));
COMMIT;
SELECT * FROM t;
|?i
[20000000000000000000]

-- 376
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint(2e20));
COMMIT;
SELECT * FROM t;
|?i
[200000000000000000000]

-- 377
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint("0x1fffffffffffffff"));
COMMIT;
SELECT * FROM t;
|?i
[2305843009213693951]

-- 378
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	INSERT INTO t VALUES (bigint("0x1ffffffffffffffffffffff"));
COMMIT;
SELECT * FROM t;
|?i
[618970019642690137449562111]

-- 379
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|?c
[42]
[114]

-- 380
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|?c, l
[100 11]
[101 22]

-- 381
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT * FROM t WHERE c > 100 ORDER BY c DESC;
|?c
[111]
[110]
[101]

-- 382
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT * FROM t WHERE c < 110 ORDER BY c;
|?c
[100]
[101]

-- 383
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT * FROM t WHERE c <= 110 ORDER BY c;
|?c
[100]
[101]
[110]

-- 384
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT * FROM t WHERE c >= 110 ORDER BY c;
|?c
[110]
[111]

-- 385
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT * FROM t WHERE c != 110 ORDER BY c;
|?c
[100]
[101]
[111]

-- 386
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT * FROM t WHERE c == 110 ORDER BY c;
|?c
[110]

-- 387
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT (c+1000) as s FROM t ORDER BY s;
|?s
[1100]
[1101]
[1110]
[1111]

-- 388
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT (1000-c) as s FROM t ORDER BY s;
|?s
[889]
[890]
[899]
[900]

-- 389
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT (c>>1) as s FROM t ORDER BY s;
|?s
[50]
[50]
[55]
[55]

-- 390
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (100), (101), (110), (111);
COMMIT;
SELECT (c<<1) as s FROM t ORDER BY s;
|?s
[200]
[202]
[220]
[222]

-- 391
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE c&0x55555 == 0x11550;
|?c
[79856]

-- 392
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE c&or;0x55555 == 0x577f5;
|?c
[79856]

-- 393
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE c&^0x55555 == 0x022a0;
|?c
[79856]

-- 394
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE c^0x55555 == 0x462a5;
|?c
[79856]

-- 395
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE c%256 == 0xf0;
|?c
[79856]

-- 396
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE c*16 == 0x137f00;
|?c
[79856]

-- 397
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE ^c == -(0x137f0+1);
|?c
[79856]

-- 398
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE +c == 0x137f0;
|?c
[79856]

-- 399
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint);
	INSERT INTO t VALUES (0x137f0);
COMMIT;
SELECT * FROM t WHERE -c == -79856;
|?c
[79856]

-- 400
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat);
	INSERT INTO t VALUES (42), (114);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|?c
[42/1]
[114/1]

-- 401
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, i int);
	INSERT INTO t VALUES (100, 1), (101, 2), (100, 10), (101, 20);
COMMIT;
SELECT c, sum(i) FROM t GROUP BY c;
|?c, l
[100/1 11]
[101/1 22]

-- 402
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat);
	INSERT INTO t VALUES (42.24), (114e3);
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|?c
[5944751508129055/140737488355328]
[114000/1]

-- 403
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat);
	INSERT INTO t VALUES ('A'), ('B');
COMMIT;
SELECT * FROM t ORDER BY 15, c, 16;
|?c
[65/1]
[66/1]

-- 404
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat);
	INSERT INTO t VALUES (bigrat("2/3")+bigrat("5/7"));
COMMIT;
SELECT * FROM t;
|?c
[29/21]

-- 405
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat);
	INSERT INTO t VALUES (bigrat("2/3")-bigrat("5/7"));
COMMIT;
SELECT * FROM t;
|?c
[-1/21]

-- 406
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat);
	INSERT INTO t VALUES (bigrat("2/3")*bigrat("5/7"));
COMMIT;
SELECT * FROM t;
|?c
[10/21]

-- 407
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint, d bigint);
	INSERT INTO t VALUES (1, 0);
COMMIT;
SELECT c/d FROM t;
||division .* zero

-- 408
BEGIN TRANSACTION;
	CREATE TABLE t (c bigint, d bigint);
	INSERT INTO t VALUES (1, 0);
COMMIT;
SELECT c%d FROM t;
||division .* zero

-- 409
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT c == d FROM t;
|b
[false]

-- 410
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("4/6"));
COMMIT;
SELECT c == d FROM t;
|b
[true]

-- 411
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT c != d FROM t;
|b
[true]

-- 412
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("4/6"));
COMMIT;
SELECT c != d FROM t;
|b
[false]

-- 413
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT c < d FROM t;
|b
[true]

-- 414
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("4/6"));
COMMIT;
SELECT c < d FROM t;
|b
[false]

-- 415
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT c <= d FROM t;
|b
[true]

-- 416
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("4/6"));
COMMIT;
SELECT c <= d FROM t;
|b
[true]

-- 417
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT c > d FROM t;
|b
[false]

-- 418
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("4/6"));
COMMIT;
SELECT c > d FROM t;
|b
[false]

-- 419
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT c >= d FROM t;
|b
[false]

-- 420
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("4/6"));
COMMIT;
SELECT c >= d FROM t;
|b
[true]

-- 421
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT c / d FROM t;
|?
[14/15]

-- 422
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("0"));
COMMIT;
SELECT c / d FROM t;
||division .* zero

-- 423
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("0"));
COMMIT;
SELECT c / (6-2*3) FROM t;
||division .* zero

-- 424
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT +c, -d FROM t;
|?, ?
[2/3 -5/7]

-- 425
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat, d bigrat);
	INSERT INTO t VALUES (bigrat("2/3"), bigrat("5/7"));
COMMIT;
SELECT 1+c, d+1, 1.5+c, d+1.5 FROM t;
|?, ?, ?, ?
[5/3 12/7 13/6 31/14]

-- 426
BEGIN TRANSACTION;
	CREATE TABLE t (c bigrat);
	INSERT INTO t VALUES (bigrat("355/113"));
COMMIT;
SELECT float(c) FROM t;
|g
[3.1415929203539825]

-- 427
BEGIN TRANSACTION;
	CREATE TABLE t (c time);
	INSERT INTO t VALUES (date(2006, 1, 2, 15, 4, 5, 999999999, "CET"));
COMMIT;
SELECT formatTime(c, "2006-01-02 15:04:05.999999999 -0700") FROM t;
|s
[2006-01-02 15:04:05.999999999 +0100]

-- 428
BEGIN TRANSACTION;
	CREATE TABLE t (c duration);
	INSERT INTO t VALUES (duration("1s")), (duration("1m")), (duration("1h"));
COMMIT;
SELECT c, string(c) FROM t ORDER BY c;
|?c, s
[1s 1s]
[1m0s 1m0s]
[1h0m0s 1h0m0s]

-- 429
BEGIN TRANSACTION;
	CREATE TABLE t (c time);
	INSERT INTO t VALUES (date(2013, 11, 26, 10, 18, 5, 999999999, "CET"));
COMMIT;
SELECT since(c) > duration("24h") FROM t;
|b
[true]

-- 430
BEGIN TRANSACTION;
	CREATE TABLE t (c time);
	INSERT INTO t VALUES (date(2013, 11, 26, 10, 32, 5, 999999999, "CET"));
COMMIT;
SELECT !(since(c) < duration("24h")) FROM t;
|b
[true]

-- 431
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("42h21m11.999999994s"),
			duration("42h21m11.999999995s"),
			duration("42h21m11.999999996s"),
		),
	;
COMMIT;
SELECT a > a, a > b, a > c, b > a, b > b, b > c, c > a, c > b, c > c FROM t;
|b, b, b, b, b, b, b, b, b
[false false false true false false true true false]

-- 432
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("42h21m11.999999994s"),
			duration("42h21m11.999999995s"),
			duration("42h21m11.999999996s"),
		),
	;
COMMIT;
SELECT a < a, a < b, a < c, b < a, b < b, b < c, c < a, c < b, c < c FROM t;
|b, b, b, b, b, b, b, b, b
[false true true false false true false false false]

-- 433
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("42h21m11.999999994s"),
			duration("42h21m11.999999995s"),
			duration("42h21m11.999999996s"),
		),
	;
COMMIT;
SELECT a <= a, a <= b, a <= c, b <= a, b <= b, b <= c, c <= a, c <= b, c <= c FROM t;
|b, b, b, b, b, b, b, b, b
[true true true false true true false false true]

-- 434
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("42h21m11.999999994s"),
			duration("42h21m11.999999995s"),
			duration("42h21m11.999999996s"),
		),
	;
COMMIT;
SELECT a >= a, a >= b, a >= c, b >= a, b >= b, b >= c, c >= a, c >= b, c >= c FROM t;
|b, b, b, b, b, b, b, b, b
[true false false true true false true true true]

-- 435
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("42h21m11.999999994s"),
			duration("42h21m11.999999995s"),
			duration("42h21m11.999999996s"),
		),
	;
COMMIT;
SELECT a != a, a != b, a != c, b != a, b != b, b != c, c != a, c != b, c != c FROM t;
|b, b, b, b, b, b, b, b, b
[false true true true false true true true false]

-- 436
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("42h21m11.999999994s"),
			duration("42h21m11.999999995s"),
			duration("42h21m11.999999996s"),
		),
	;
COMMIT;
SELECT a == a, a == b, a == c, b == a, b == b, b == c, c == a, c == b, c == c FROM t;
|b, b, b, b, b, b, b, b, b
[true false false false true false false false true]

-- 437
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("5h"),
			duration("3m"),
			duration("2s"),
		),
	;
COMMIT;
SELECT b+c, a+c, a+b, a+b+c FROM t;
|?, ?, ?, ?
[3m2s 5h0m2s 5h3m0s 5h3m2s]

-- 438
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("5h"),
			duration("3m"),
			duration("2s"),
		),
	;
COMMIT;
SELECT b-c, a-c, a-b, a-b-c FROM t;
|?, ?, ?, ?
[2m58s 4h59m58s 4h57m0s 4h56m58s]

-- 439
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("5h"),
			duration("3m"),
			duration("2s"),
		),
	;
COMMIT;
SELECT a>>1, b>>1, c>>1 FROM t;
|?, ?, ?
[2h30m0s 1m30s 1s]

-- 440
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("5h"),
			duration("3m"),
			duration("2s"),
		),
	;
COMMIT;
SELECT a<<1, b<<1, c<<1 FROM t;
|?, ?, ?
[10h0m0s 6m0s 4s]

-- 441
BEGIN TRANSACTION;
	CREATE TABLE t (a duration);
	INSERT INTO t VALUES (
			duration("257ns"),
		),
	;
COMMIT;
SELECT a & 255 FROM t;
|?
[1ns]

-- 442
BEGIN TRANSACTION;
	CREATE TABLE t (a duration);
	INSERT INTO t VALUES (
			duration("1ns"),
		),
	;
COMMIT;
SELECT a &or; 256 FROM t;
|?
[257ns]

-- 443
BEGIN TRANSACTION;
	CREATE TABLE t (a duration);
	INSERT INTO t VALUES (
			duration(0x731),
		),
	;
COMMIT;
SELECT a &^ 0xd30 FROM t;
|?
[513ns]

-- 444
BEGIN TRANSACTION;
	CREATE TABLE t (a duration);
	INSERT INTO t VALUES (
			duration("3h2m1s"),
		),
	;
COMMIT;
SELECT a % duration("2h"), a % duration("1m") FROM t;
|?, ?
[1h2m1s 1s]

-- 445
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("5h"),
			duration("3m"),
			duration("2s"),
		),
	;
COMMIT;
SELECT a/2, b/2, c/2 FROM t;
|?, ?, ?
[2h30m0s 1m30s 1s]

-- 446
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("5h"),
			duration("3m"),
			duration("2s"),
		),
	;
COMMIT;
SELECT a*2, 2*b, c*2 FROM t;
|?, ?, ?
[10h0m0s 6m0s 4s]

-- 447
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("1ns"),
			duration("3ns"),
			duration("5ns"),
		),
	;
COMMIT;
SELECT ^a, ^b, ^c FROM t;
|?, ?, ?
[-2ns -4ns -6ns]

-- 448
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("1ns"),
			duration("3ns"),
			duration("5ns"),
		),
	;
COMMIT;
SELECT +a, +b, +c FROM t;
|?, ?, ?
[1ns 3ns 5ns]

-- 449
BEGIN TRANSACTION;
	CREATE TABLE t (a duration, b duration, c duration);
	INSERT INTO t VALUES (
			duration("1ns"),
			duration("3ns"),
			duration("5ns"),
		),
	;
COMMIT;
SELECT -a, -b, -c FROM t;
|?, ?, ?
[-1ns -3ns -5ns]

-- 450
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b time, c time);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999994, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999995, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999996, "CET"),
		),
	;
COMMIT;
SELECT a > a, a > b, a > c, b > a, b > b, b > c, c > a, c > b, c > c FROM t;
|b, b, b, b, b, b, b, b, b
[false false false true false false true true false]

-- 451
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b time, c time);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999994, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999995, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999996, "CET"),
		),
	;
COMMIT;
SELECT a < a, a < b, a < c, b < a, b < b, b < c, c < a, c < b, c < c FROM t;
|b, b, b, b, b, b, b, b, b
[false true true false false true false false false]

-- 452
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b time, c time);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999994, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999995, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999996, "CET"),
		),
	;
COMMIT;
SELECT a <= a, a <= b, a <= c, b <= a, b <= b, b <= c, c <= a, c <= b, c <= c FROM t;
|b, b, b, b, b, b, b, b, b
[true true true false true true false false true]

-- 453
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b time, c time);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999994, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999995, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999996, "CET"),
		),
	;
COMMIT;
SELECT a >= a, a >= b, a >= c, b >= a, b >= b, b >= c, c >= a, c >= b, c >= c FROM t;
|b, b, b, b, b, b, b, b, b
[true false false true true false true true true]

-- 454
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b time, c time);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999994, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999995, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999996, "CET"),
		),
	;
COMMIT;
SELECT a != a, a != b, a != c, b != a, b != b, b != c, c != a, c != b, c != c FROM t;
|b, b, b, b, b, b, b, b, b
[false true true true false true true true false]

-- 455
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b time, c time);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999994, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999995, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999996, "CET"),
		),
	;
COMMIT;
SELECT a == a, a == b, a == c, b == a, b == b, b == c, c == a, c == b, c == c FROM t;
|b, b, b, b, b, b, b, b, b
[true false false false true false false false true]

-- 456
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b duration);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999999, "CET"),
			duration("3h2m1s"),
		),
	;
COMMIT;
SELECT formatTime(a+b, "2006-01-02 15:04:05.999999999 -0700") FROM t;
|s
[2013-11-27 15:03:03.999999999 +0100]

-- 457
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b duration);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999999, "CET"),
			duration("3h2m1s"),
		),
	;
COMMIT;
SELECT formatTime(b+a, "2006-01-02 15:04:05.999999999 -0700") FROM t;
|s
[2013-11-27 15:03:03.999999999 +0100]

-- 458
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b duration);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999999, "CET"),
			duration("3h2m1s"),
		),
	;
COMMIT;
SELECT a+a FROM t;
||invalid operation

-- 459
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b time);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 13, 2, 3, 999999999, "CET"),
			date(2013, 11, 27, 12, 1, 2, 999999999, "CET"),
		),
	;
COMMIT;
SELECT a-b FROM t;
|?
[1h1m1s]

-- 460
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b duration);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999999, "CET"),
			duration("3h2m1s"),
		),
	;
COMMIT;
SELECT formatTime(a-b, "2006-01-02 15:04:05.999999999 -0700") FROM t;
|s
[2013-11-27 08:59:01.999999999 +0100]

-- 461
BEGIN TRANSACTION;
	CREATE TABLE t (a time, b duration);
	INSERT INTO t VALUES (
			date(2013, 11, 27, 12, 1, 2, 999999999, "CET"),
			duration("3h2m1s"),
		),
	;
COMMIT;
SELECT b-a FROM t;
||invalid operation

-- 462
BEGIN TRANSACTION;
	CREATE TABLE t (a duration);
	INSERT INTO t VALUES (
			duration("3h2m1.5s"),
		),
	;
COMMIT;
SELECT hours(a), minutes(a), seconds(a), nanoseconds(a) FROM t;
|g, g, g, l
[3.03375 182.025 10921.5 10921500000000]

-- 463
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES (now()-duration("1s"));
COMMIT;
SELECT a < now(), now() > a, a >= now(), now() <= a FROM t;
|b, b, b, b
[true true false false]

-- 464
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04pm (MST)", "Nov 27, 2013 at 2:07pm (CET)")),
		(parseTime("2006-Jan-02", "2013-Nov-27")),
	;
COMMIT;
SELECT formatTime(a, "2006-01-02 15:04:05.999999999 -0700") as a FROM t ORDER BY a;
|sa
[2013-11-27 00:00:00 +0000]
[2013-11-27 14:07:00 +0100]

-- 465
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04pm (MST)", "Nov 27, 2013 at 2:07pm (CET)")),
		(parseTime("2006-Jan-02", "2013-Nov-27")),
	;
COMMIT;
SELECT hour(a) AS y FROM t ORDER BY y;
|ly
[0]
[14]

-- 466
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04pm (MST)", "Nov 27, 2013 at 2:07pm (CET)")),
		(parseTime("2006-Jan-02", "2013-Nov-27")),
	;
COMMIT;
SELECT minute(a) AS y FROM t ORDER BY y;
|ly
[0]
[7]

-- 467
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04:05pm (MST)", "Nov 27, 2013 at 2:07:31pm (CET)")),
		(parseTime("2006-Jan-02", "2013-Nov-27")),
	;
COMMIT;
SELECT second(a) AS y FROM t ORDER BY y;
|ly
[0]
[31]

-- 468
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04:05pm (MST)", "Nov 27, 2013 at 2:07:31.123456789pm (CET)")),
		(parseTime("2006-Jan-02", "2013-Nov-27")),
	;
COMMIT;
SELECT nanosecond(a) AS y FROM t ORDER BY y;
|ly
[0]
[123456789]

-- 469
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04:05pm (MST)", "Nov 27, 2013 at 2:07:31.123456789pm (CET)")),
		(parseTime("2006-Jan-02", "2014-Nov-28")),
	;
COMMIT;
SELECT year(a) AS y FROM t ORDER BY y;
|ly
[2013]
[2014]

-- 470
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04:05pm (MST)", "Nov 27, 2013 at 2:07:31.123456789pm (CET)")),
		(parseTime("2006-Jan-02", "2014-Nov-28")),
	;
COMMIT;
SELECT day(a) AS y FROM t ORDER BY y;
|ly
[27]
[28]

-- 471
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04:05pm (MST)", "Nov 27, 2013 at 2:07:31.123456789pm (CET)")),
		(parseTime("2006-Jan-02", "2014-Dec-28")),
	;
COMMIT;
SELECT month(a) AS y FROM t ORDER BY y;
|ly
[11]
[12]

-- 472
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04:05pm (MST)", "Nov 27, 2013 at 2:07:31.123456789pm (CET)")),
		(parseTime("2006-Jan-02", "2013-Sep-08")),
	;
COMMIT;
SELECT weekday(a) AS y FROM t ORDER BY y;
|ly
[0]
[3]

-- 473
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04:05pm (MST)", "Feb 1, 2013 at 2:07:31.123456789pm (CET)")),
		(parseTime("2006-Jan-02", "2014-Feb-02")),
	;
COMMIT;
SELECT yearDay(a) AS y FROM t ORDER BY y;
|ly
[32]
[33]

-- 474
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04pm (MST)", "Feb 1, 2013 at 2:07pm (CET)")),
		(parseTime("2006-Jan-02", "2014-Feb-02")),
	;
COMMIT;
SELECT timeIn(a, ""), timeIn(a, "UTC") AS y FROM t ORDER BY y;
|?, ?y
[2013-02-01 13:07:00 +0000 UTC 2013-02-01 13:07:00 +0000 UTC]
[2014-02-02 00:00:00 +0000 UTC 2014-02-02 00:00:00 +0000 UTC]

-- 475
BEGIN TRANSACTION;
	CREATE TABLE t (a time);
	INSERT INTO t VALUES
		(parseTime("Jan 2, 2006 at 3:04pm (MST)", "Feb 1, 2013 at 2:07pm (CET)")),
		(parseTime("2006-Jan-02", "2014-Feb-02")),
	;
COMMIT;
SELECT formatTime(timeIn(a, "UTC"), "Jan 2, 2006 at 3:04pm (UTC)") AS y FROM t ORDER BY y;
|sy
[Feb 1, 2013 at 1:07pm (UTC)]
[Feb 2, 2014 at 12:00am (UTC)]

-- 476
BEGIN TRANSACTION;
	BEGIN TRANSACTION;
	COMMIT;
COMMIT;
SELECT * FROM t;
||does not exist

-- 477
BEGIN TRANSACTION;
	BEGIN TRANSACTION;
	ROLLBACK;
COMMIT;
SELECT * FROM t;
||does not exist

-- 478 // https://github.com/cznic/ql/issues/23
BEGIN TRANSACTION;
	CREATE TABLE t (a int, b string, t time);
	INSERT INTO t VALUES (1, "a", parseTime("Jan 2, 2006 at 3:04pm (MST)", "Jan 12, 2014 at 6:26pm (CET)"));
	INSERT INTO t VALUES (2, "b", parseTime("Jan 2, 2006 at 3:04pm (MST)", "Jan 12, 2014 at 6:27pm (CET)"));
	UPDATE t b = "hello" WHERE a == 1;
COMMIT;
SELECT a, b, formatTime(t, "2006-01-02 15:04:05.999999999 -0700") as t FROM t;
|la, sb, st
[2 b 2014-01-12 18:27:00 +0100]
[1 hello 2014-01-12 18:26:00 +0100]

-- 479 // https://github.com/cznic/ql/issues/23
BEGIN TRANSACTION;
	CREATE TABLE t (a int, b string, t time);
	INSERT INTO t VALUES (1, "a", parseTime("Jan 2, 2006 at 3:04pm (MST)", "Jan 12, 2014 at 6:26pm (CET)"));
	INSERT INTO t VALUES (2, "b", parseTime("Jan 2, 2006 at 3:04pm (MST)", "Jan 12, 2014 at 6:27pm (CET)"));
	UPDATE t
		b = "hello",
		t = parseTime("Jan 2, 2006 at 3:04pm (MST)", "Jan 12, 2014 at 6:28pm (CET)"),
	WHERE a == 1;
COMMIT;
SELECT a, b, formatTime(t, "2006-01-02 15:04:05.999999999 -0700") as t FROM t;
|la, sb, st
[2 b 2014-01-12 18:27:00 +0100]
[1 hello 2014-01-12 18:28:00 +0100]

-- 480 // https://github.com/cznic/ql/issues/23
BEGIN TRANSACTION;
	CREATE TABLE t (a int, b string, d duration);
	INSERT INTO t VALUES (1, "a", duration("1m"));
	INSERT INTO t VALUES (2, "b", duration("2m"));
	UPDATE t b = "hello" WHERE a == 1;
COMMIT;
SELECT * FROM t;
|la, sb, ?d
[2 b 2m0s]
[1 hello 1m0s]

-- 481 // https://github.com/cznic/ql/issues/23
BEGIN TRANSACTION;
	CREATE TABLE t (a int, b string, d duration);
	INSERT INTO t VALUES (1, "a", duration("1m"));
	INSERT INTO t VALUES (2, "b", duration("2m"));
	UPDATE t
		b = "hello",
		d = duration("3m"),
	WHERE a == 1;
COMMIT;
SELECT * FROM t;
|la, sb, ?d
[2 b 2m0s]
[1 hello 3m0s]

-- 482 // https://github.com/cznic/ql/issues/24
BEGIN TRANSACTION;
	CREATE TABLE t (c complex128);
	INSERT INTO t VALUES
		(2+complex128(1)),
		(22+complex(0, 1)),
	;
COMMIT;
SELECT * FROM t ORDER BY real(c);
|dc
[(3+0i)]
[(22+1i)]

-- 483
BEGIN TRANSACTION;
	CREATE TABLE t (s string, substr string);
	INSERT INTO t VALUES
		("seafood", "foo"),
		("seafood", "bar"),
		("seafood", ""),
		("", ""),
	;
COMMIT;
SELECT id() as i, contains(42, substr) FROM t ORDER BY i;
||invalid .* 42

-- 484
BEGIN TRANSACTION;
	CREATE TABLE t (s string, substr string);
	INSERT INTO t VALUES
		("seafood", "foo"),
		("seafood", "bar"),
		("seafood", ""),
		("", ""),
	;
COMMIT;
SELECT id() as i, contains(s, true) FROM t ORDER BY i;
||invalid .* true

-- 485
BEGIN TRANSACTION;
	CREATE TABLE t (s string, substr string);
	INSERT INTO t VALUES
		("seafood", "foo"),
		("seafood", "bar"),
		("seafood", ""),
		("", ""),
		("", NULL),
		("foo", NULL),
		(NULL, ""),
		(NULL, "foo"),
		(NULL, NULL),
	;
COMMIT;
SELECT id() as i, contains(s, substr) FROM t ORDER BY i;
|li, b
[1 true]
[2 false]
[3 true]
[4 true]
[5 <nil>]
[6 <nil>]
[7 <nil>]
[8 <nil>]
[9 <nil>]

-- 486
BEGIN TRANSACTION;
	CREATE TABLE t (s string, prefix string);
	INSERT INTO t VALUES
		("", ""),
		("f", ""),
		("", "foo"),
		("f", "foo"),
		("fo", "foo"),
		("foo", "foo"),
		("fooo", "foo"),
	;
COMMIT;
SELECT id() as i, hasPrefix(42, prefix) FROM t ORDER BY i;
||invalid .* 42

-- 487
BEGIN TRANSACTION;
	CREATE TABLE t (s string, prefix string);
	INSERT INTO t VALUES
		("", ""),
		("f", ""),
		("", "foo"),
		("f", "foo"),
		("fo", "foo"),
		("foo", "foo"),
		("fooo", "foo"),
	;
COMMIT;
SELECT id() as i, hasPrefix(s, false) FROM t ORDER BY i;
||invalid .* false

-- 488
BEGIN TRANSACTION;
	CREATE TABLE t (s string, prefix string);
	INSERT INTO t VALUES
		("", ""),
		("f", ""),
		("", "foo"),
		("f", "foo"),
		("fo", "foo"),
		("foo", "foo"),
		("fooo", "foo"),
		("", NULL),
		("foo", NULL),
		(NULL, ""),
		(NULL, "foo"),
		(NULL, NULL),
	;
COMMIT;
SELECT id() as i, hasPrefix(s, prefix) FROM t ORDER BY i;
|li, b
[1 true]
[2 true]
[3 false]
[4 false]
[5 false]
[6 true]
[7 true]
[8 <nil>]
[9 <nil>]
[10 <nil>]
[11 <nil>]
[12 <nil>]

-- 489
BEGIN TRANSACTION;
	CREATE TABLE t (s string, suffix string);
	INSERT INTO t VALUES
		("", ""),
		("f", ""),
		("x", "foo"),
		("xf", "foo"),
		("xfo", "foo"),
		("xfoo", "foo"),
		("xfooo", "foo"),
	;
COMMIT;
SELECT id() as i, hasSuffix(42, suffix) FROM t ORDER BY i;
||invalid .* 42

-- 490
BEGIN TRANSACTION;
	CREATE TABLE t (s string, suffix string);
	INSERT INTO t VALUES
		("", ""),
		("f", ""),
		("x", "foo"),
		("xf", "foo"),
		("xfo", "foo"),
		("xfoo", "foo"),
		("xfooo", "foo"),
	;
COMMIT;
SELECT id() as i, hasSuffix(s, true) FROM t ORDER BY i;
||invalid .* true

-- 491
BEGIN TRANSACTION;
	CREATE TABLE t (s string, suffix string);
	INSERT INTO t VALUES
		("", ""),
		("f", ""),
		("x", "foo"),
		("xf", "foo"),
		("xfo", "foo"),
		("xfoo", "foo"),
		("xfooo", "foo"),
		("", NULL),
		("foo", NULL),
		(NULL, ""),
		(NULL, "foo"),
		(NULL, NULL),
	;
COMMIT;
SELECT id() as i, hasSuffix(s, suffix) FROM t ORDER BY i;
|li, b
[1 true]
[2 true]
[3 false]
[4 false]
[5 false]
[6 true]
[7 false]
[8 <nil>]
[9 <nil>]
[10 <nil>]
[11 <nil>]
[12 <nil>]

-- 492 // issue #27
BEGIN TRANSACTION;
	DROP TABLE nonexistent;
COMMIT;
||does not exist

-- 493 // issue #27
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	DROP TABLE IF EXISTS nonexistent;
COMMIT;
SELECT * FROM t;
|?i

-- 494 // issue #27
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE TABLE t (i int);
COMMIT;
||exist

-- 495 // issue #27
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE TABLE IF NOT EXISTS t (s string);
COMMIT;
SELECT * FROM t;
|?i

-- 496 // issue #28
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
	ALTER TABLE t ADD s string;
COMMIT;
SELECT * FROM t;
|li, ?s
[42 <nil>]

-- 497
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
BEGIN TRANSACTION;
	INSERT INTO t VALUES(1000);
	BEGIN TRANSACTION;
		INSERT INTO t VALUES(2000);
	COMMIT;
	INSERT INTO t VALUES(3000);
COMMIT;
SELECT * FROM t;
|li
[3000]
[2000]
[1000]

-- 498
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
BEGIN TRANSACTION;
	INSERT INTO t VALUES(1000);
	BEGIN TRANSACTION;
		INSERT INTO t VALUES(2000);
	ROLLBACK;
	INSERT INTO t VALUES(3000);
COMMIT;
SELECT * FROM t;
|li
[3000]
[1000]

-- 499
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES(42, "foo");
	ALTER TABLE t DROP COLUMN i;
COMMIT;
SELECT * FROM t;
|ss
[foo]

-- 500
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES(42, "foo");
	ALTER TABLE t DROP COLUMN s;
COMMIT;
SELECT * FROM t;
|li
[42]

-- 501 // new spec rule: table must have at least 1 column
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t DROP COLUMN c;
COMMIT;
SELECT * FROM t;
||cannot drop.*column

-- 502 // fixed bug
BEGIN TRANSACTION;
	CREATE TABLE t (c int, s string);
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t DROP COLUMN s;
ROLLBACK;
SELECT * FROM t;
|?c, ?s

-- 503 // fixed bug
BEGIN TRANSACTION;
	CREATE TABLE t (c int, s string);
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t ADD b bool;
ROLLBACK;
SELECT * FROM t;
|?c, ?s

-- 504 // fixed bug
BEGIN TRANSACTION;
	CREATE TABLE t (c int, s string);
COMMIT;
BEGIN TRANSACTION;
	DROP TABLE t;
ROLLBACK;
SELECT * FROM t;
|?c, ?s

-- 505 // fixed bug
BEGIN TRANSACTION;
	CREATE INDEX x ON t (qty());
COMMIT;
||only .* id

-- 506
BEGIN TRANSACTION;
	CREATE INDEX x ON t (qty);
COMMIT;
||table.*not exist

-- 507
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX x ON t (qty);
COMMIT;
||column.*not exist

-- 508
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX x ON t (id());
COMMIT;
SELECT * FROM t;
|?c

-- 509
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX y ON t (c);
COMMIT;
SELECT * FROM t;
|?c

-- 510
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX x ON t (id());
	CREATE INDEX y ON t (id());
COMMIT;
SELECT * FROM t;
||already

-- 511
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX x ON t (id());
	CREATE INDEX x ON t (c);
COMMIT;
SELECT * FROM t;
||already

-- 512
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX x ON t (id());
	CREATE INDEX y ON t (c);
COMMIT;
SELECT * FROM t;
|?c

-- 513
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX y ON t (c);
	CREATE INDEX x ON t (id());
COMMIT;
SELECT * FROM t;
|?c

-- 514
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES(42);
COMMIT;
SELECT * FROM t;
|lc
[42]

-- 515
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	INSERT INTO t VALUES(42);
	CREATE INDEX x ON t (id());
COMMIT;
SELECT * FROM t;
|lc
[42]

-- 516
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	INSERT INTO t VALUES(42);
	INSERT INTO t VALUES(24);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES(1);
	CREATE INDEX i ON t (c);
	INSERT INTO t VALUES(999);
COMMIT;
SELECT * FROM t ORDER BY id();
|lc
[42]
[24]
[1]
[999]

-- 517
BEGIN TRANSACTION;
	CREATE TABLE t (c int);
	INSERT INTO t VALUES(42);
	INSERT INTO t VALUES(24);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES(1);
	CREATE INDEX i ON t (c);
	INSERT INTO t VALUES(999);
COMMIT;
SELECT * FROM t ORDER BY c;
|lc
[1]
[24]
[42]
[999]

-- 518
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX xid ON t (id());
	INSERT INTO t VALUES(42);
	INSERT INTO t VALUES(24);
	CREATE INDEX ii ON t (i);
	INSERT INTO t VALUES(1);
	INSERT INTO t VALUES(999);
	UPDATE t i = 240 WHERE i == 24;
	DELETE FROM t WHERE i == 240;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[42]
[999]

-- 519
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX i ON t (i);
COMMIT;
||collision: i

-- 520
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX t ON t (i);
COMMIT;
||collision: t

-- 521
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE TABLE u (s string);
	CREATE INDEX u ON t (i);
COMMIT;
||collision.*: u

-- 522
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE TABLE u (s string);
	CREATE INDEX z ON t (i);
	CREATE INDEX z ON u (s);
COMMIT;
||already

-- 523
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX u ON u (s);
COMMIT;
||collision: u

-- 524
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX v ON u (v);
COMMIT;
||collision: v

-- 525
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX s ON t (i);
COMMIT;
||collision.*: s

-- 526
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX id ON t (i);
COMMIT;
SELECT * FROM t;
|?i

-- 527
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	CREATE TABLE x (s string);
COMMIT;
||table t.*index x

-- 528
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
BEGIN TRANSACTION;
	INSERT INTO t VALUES(1000);
	BEGIN TRANSACTION;
		INSERT INTO t VALUES(2000);
	ROLLBACK;
	INSERT INTO t VALUES(3000);
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1000]
[3000]

-- 529
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
	TRUNCATE TABLE t;
COMMIT;
SELECT * FROM t;
|?i

-- 530
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
	DELETE FROM t;
COMMIT;
SELECT * FROM t;
|?i

-- 531
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX xi ON t (i);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN i;
	INSERT INTO t VALUES ("bar");
COMMIT;
SELECT * FROM t ORDER BY s;
|ss
[bar]
[foo]

-- 532
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX xs ON t (s);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN i;
	INSERT INTO t VALUES ("bar");
COMMIT;
SELECT * FROM t ORDER BY s;
|ss
[bar]
[foo]

-- 533
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX xi ON t (i);
	CREATE INDEX xs ON t (s);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN i;
	INSERT INTO t VALUES ("bar");
COMMIT;
SELECT * FROM t ORDER BY s;
|ss
[bar]
[foo]

-- 534
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX xi ON t (i);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN s;
	INSERT INTO t VALUES (24);
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[24]
[42]

-- 535
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX xs ON t (s);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN s;
	INSERT INTO t VALUES (24);
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[24]
[42]

-- 536
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX xi ON t (i);
	CREATE INDEX xs ON t (s);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN s;
	INSERT INTO t VALUES (24);
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[24]
[42]

-- 537
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN i;
COMMIT;
SELECT * FROM t;
|ss
[foo]

-- 538
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES (42, "foo");
	ALTER TABLE t DROP COLUMN s;
COMMIT;
SELECT * FROM t;
|li
[42]

-- 539
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES (42, "foo");
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t DROP COLUMN i;
COMMIT;
SELECT * FROM t;
|ss
[foo]

-- 540
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES (42, "foo");
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t DROP COLUMN s;
COMMIT;
SELECT * FROM t;
|li
[42]

-- 541
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42, "foo");
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t DROP COLUMN s;
COMMIT;
BEGIN TRANSACTION;
	INSERT INTO t VALUES (24);
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[24]
[42]

-- 542
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42, "foo");
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t DROP COLUMN i;
COMMIT;
BEGIN TRANSACTION;
	INSERT INTO t VALUES ("bar");
COMMIT;
SELECT * FROM t ORDER BY s;
|ss
[bar]
[foo]

-- 543
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
	INSERT INTO t SELECT 10*i FROM t;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[42]
[420]

-- 544
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
	INSERT INTO t SELECT 10*i FROM t;
	DROP INDEX none;
COMMIT;
||index none does not exist

-- 545
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
	INSERT INTO t SELECT 10*i FROM t;
	DROP INDEX x;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[42]
[420]

-- 546
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
	INSERT INTO t SELECT 10*i FROM t;
COMMIT;
BEGIN TRANSACTION;
	DROP INDEX x;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[42]
[420]

-- 547
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	ALTER TABLE t ADD s string;
COMMIT;
SELECT * FROM t;
|?i, ?s

-- 548
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	ALTER TABLE t ADD s string;
COMMIT;
SELECT * FROM t ORDER BY i;
|?i, ?s

-- 549
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t ADD s string;
COMMIT;
SELECT * FROM t ORDER BY s;
|?i, ?s

-- 550
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
COMMIT;
SELECT * FROM x;
|?x
[42]

-- 551
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (42);
	INSERT INTO t VALUES (420);
COMMIT;
SELECT * FROM x;
|?x
[42]
[420]

-- 552
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (420);
	INSERT INTO t VALUES (42);
COMMIT;
SELECT * FROM x;
|?x
[42]
[420]

-- 553
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (420);
	INSERT INTO t VALUES (42);
	INSERT INTO t VALUES (100);
COMMIT;
SELECT * FROM x;
|?x
[42]
[100]
[420]

-- 554
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (420);
	INSERT INTO t VALUES (42);
	INSERT INTO t VALUES (100);
	DELETE FROM t WHERE i == 100;
COMMIT;
SELECT * FROM x;
|?x
[42]
[420]

-- 555
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (420);
	INSERT INTO t VALUES (42);
	INSERT INTO t VALUES (100);
COMMIT;
BEGIN TRANSACTION;
	DELETE FROM t WHERE i == 100;
COMMIT;
SELECT * FROM x;
|?x
[42]
[420]

-- 556
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (1);
	INSERT INTO t VALUES (2);
	INSERT INTO t VALUES (3);
COMMIT;
SELECT * FROM x;
|?x
[1]
[2]
[3]

-- 557
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (1);
	INSERT INTO t VALUES (3);
	INSERT INTO t VALUES (2);
COMMIT;
SELECT * FROM x;
|?x
[1]
[2]
[3]

-- 558
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (2);
	INSERT INTO t VALUES (1);
	INSERT INTO t VALUES (3);
COMMIT;
SELECT * FROM x;
|?x
[1]
[2]
[3]

-- 559
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (2);
	INSERT INTO t VALUES (3);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * FROM x;
|?x
[1]
[2]
[3]

-- 560
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (3);
	INSERT INTO t VALUES (1);
	INSERT INTO t VALUES (2);
COMMIT;
SELECT * FROM x;
|?x
[1]
[2]
[3]

-- 561
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (3);
	INSERT INTO t VALUES (2);
	INSERT INTO t VALUES (1);
COMMIT;
SELECT * FROM x;
|?x
[1]
[2]
[3]

-- 562
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (bigint(1) << 100);
	INSERT INTO t VALUES (bigint(1) << 100 - 1);
	INSERT INTO t VALUES (bigint(1) << (256*8));
	INSERT INTO t VALUES (bigint(1) << 100 + 1);
	INSERT INTO t VALUES (bigint(1) << 10);
COMMIT;
SELECT * FROM x;
|?x
[1024]
[1267650600228229401496703205375]
[1267650600228229401496703205376]
[1267650600228229401496703205377]
[32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463688717960921898019494119559150490921095088152386448283120630877367300996091750197750389652106796057638384067568276792218642619756161838094338476170470581645852036305042887575891541065808607552399123930385521914333389668342420684974786564569494856176035326322058077805659331026192708460314150258592864177116725943603718461857357598351152301645904403697613233287231227125684710820209725157101726931323469678542580656697935045997268352998638215525166389437335543602135433229604645318478604952148193555853611059596230656]

-- 563
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (bigint(1) << 100);
	INSERT INTO t VALUES (bigint(1) << 100 - 1);
	INSERT INTO t VALUES (bigint(1) << (256*8));
	INSERT INTO t VALUES (bigint(1) << 100 + 1);
	INSERT INTO t VALUES (bigint(1) << 10);
COMMIT;
BEGIN TRANSACTION;
	DELETE FROM t WHERE i == (bigint(1) << (256*8));
COMMIT;
SELECT * FROM x;
|?x
[1024]
[1267650600228229401496703205375]
[1267650600228229401496703205376]
[1267650600228229401496703205377]

-- 564
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (bigint(1) << 100);
	INSERT INTO t VALUES (bigint(1) << 100 - 1);
	INSERT INTO t VALUES (bigint(1) << 100 + 1);
	INSERT INTO t VALUES (bigint(1) << 10);
COMMIT;
BEGIN TRANSACTION;
	UPDATE t
		i = i+10,
	WHERE i == bigint(1) << 100;
COMMIT;
SELECT * FROM x;
|?x
[1024]
[1267650600228229401496703205375]
[1267650600228229401496703205377]
[1267650600228229401496703205386]

-- 565
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (bigint(1) << 100);
	INSERT INTO t VALUES (bigint(1) << 100 - 1);
	INSERT INTO t VALUES (bigint(1) << (256*8));
	INSERT INTO t VALUES (bigint(1) << 100 + 1);
	INSERT INTO t VALUES (bigint(1) << 10);
COMMIT;
BEGIN TRANSACTION;
	UPDATE t
		i = 42,
	WHERE i == bigint(1) << (256*8);
COMMIT;
SELECT * FROM x;
|?x
[42]
[1024]
[1267650600228229401496703205375]
[1267650600228229401496703205376]
[1267650600228229401496703205377]

-- 566
BEGIN TRANSACTION;
	CREATE TABLE t (i bigint);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES (bigint(1) << 100);
	INSERT INTO t VALUES (bigint(1) << 100 - 1);
	INSERT INTO t VALUES (bigint(42));
	INSERT INTO t VALUES (bigint(1) << 100 + 1);
	INSERT INTO t VALUES (bigint(1) << 10);
COMMIT;
BEGIN TRANSACTION;
	UPDATE t
		i = bigint(1) << (256*8),
	WHERE i == 42;
COMMIT;
SELECT * FROM x;
|?x
[1024]
[1267650600228229401496703205375]
[1267650600228229401496703205376]
[1267650600228229401496703205377]
[32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463688717960921898019494119559150490921095088152386448283120630877367300996091750197750389652106796057638384067568276792218642619756161838094338476170470581645852036305042887575891541065808607552399123930385521914333389668342420684974786564569494856176035326322058077805659331026192708460314150258592864177116725943603718461857357598351152301645904403697613233287231227125684710820209725157101726931323469678542580656697935045997268352998638215525166389437335543602135433229604645318478604952148193555853611059596230656]

-- 567
BEGIN TRANSACTION;
	CREATE TABLE t (b blob);
	CREATE INDEX x ON t (b);
	INSERT INTO t VALUES (blob(
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" // > shortBlob
	));
	DROP TABLE t;
COMMIT;
SELECT * FROM t;
||does not exist

-- 568
BEGIN TRANSACTION;
	CREATE TABLE t (b blob);
	CREATE INDEX x ON t (b);
	INSERT INTO t VALUES (blob(
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" // > shortBlob
	));
	DROP TABLE t;
COMMIT;
BEGIN TRANSACTION;
	DROP TABLE t;
COMMIT;
SELECT * FROM t;
||does not exist

-- 569
BEGIN TRANSACTION;
	CREATE TABLE t (b blob);
	CREATE INDEX x ON t (b);
	INSERT INTO t VALUES (blob(
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" // > shortBlob
	));
	DROP INDEX x;
COMMIT;
SELECT len(string(b)) AS n FROM t;
|ln
[320]

-- 570
BEGIN TRANSACTION;
	CREATE TABLE t (b blob);
	CREATE INDEX x ON t (b);
	INSERT INTO t VALUES (blob(
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"+
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" // > shortBlob
	));
COMMIT;
BEGIN TRANSACTION;
	DROP INDEX x;
COMMIT;
SELECT len(string(b)) AS n FROM t;
|ln
[320]

-- 571
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
	ALTER TABLE t ADD s string;
COMMIT;
SELECT * FROM t;
|li, ?s
[42 <nil>]

-- 572
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
COMMIT;
BEGIN TRANSACTION;
	ALTER TABLE t ADD s string;
COMMIT;
SELECT * FROM t;
|li, ?s
[42 <nil>]

-- 573
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
SELECT * FROM q.t;
||syntax error

-- 574
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
COMMIT;
SELECT * FROM t AS u;
|li
[42]

-- 575
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
COMMIT;
SELECT u.x FROM t AS u;
||unknown field u.x

-- 576
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
COMMIT;
SELECT u.i FROM t AS u;
||unknown field u.i

-- 577
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
COMMIT;
SELECT i FROM t AS u;
|li
[42]

-- 578
BEGIN TRANSACTION;
	CREATE TABLE t (i int, b bool);
	CREATE INDEX x ON t (b);
	INSERT INTO t VALUES(24, false);
	INSERT INTO t VALUES(333, NULL);
	INSERT INTO t VALUES(42, true);
	INSERT INTO t VALUES(240, false);
	INSERT INTO t VALUES(420, true);
COMMIT;
SELECT i FROM t WHERE b ORDER BY i;
|li
[42]
[420]

-- 579
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(10, "foo");
COMMIT;
SELECT * FROM t WHERE i < "30";
||cannot.*compar

-- 580
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM x;
|?x
[<nil>]
[<nil>]
[10]
[10]
[20]
[20]
[30]
[30]
[40]
[40]
[50]
[50]

-- 581
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE i < 30;
|li
[10]
[20]
[20]
[10]

-- 582 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE i < 30;
|li
[10]
[10]
[20]
[20]

-- 583
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i <= 30;
|li
[10]
[20]
[30]
[30]
[20]
[10]

-- 584 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i <= 30;
|li
[10]
[10]
[20]
[20]
[30]
[30]

-- 585
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE i == 30;
|li
[30]
[30]

-- 586 // index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE i == 30;
|li
[30]
[30]

-- 587
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i >= 30;
|li
[50]
[40]
[30]
[30]
[40]
[50]

-- 588 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i >= 30;
|li
[30]
[30]
[40]
[40]
[50]
[50]

-- 589
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i > 30;
|li
[50]
[40]
[40]
[50]

-- 590 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i > 30;
|li
[50]
[50]
[40]
[40]

-- 591
BEGIN TRANSACTION;
	CREATE TABLE t (i int, b bool);
	INSERT INTO t VALUES(24, false);
	INSERT INTO t VALUES(333, NULL);
	INSERT INTO t VALUES(42, true);
	INSERT INTO t VALUES(240, false);
	INSERT INTO t VALUES(420, true);
COMMIT;
SELECT i FROM t WHERE !b ORDER BY i;
|li
[24]
[240]

-- 592
BEGIN TRANSACTION;
	CREATE TABLE t (i int, b bool);
	CREATE INDEX x ON t (b);
	INSERT INTO t VALUES(24, false);
	INSERT INTO t VALUES(333, NULL);
	INSERT INTO t VALUES(42, true);
	INSERT INTO t VALUES(240, false);
	INSERT INTO t VALUES(420, true);
COMMIT;
SELECT i FROM t WHERE !b ORDER BY i;
|li
[24]
[240]

-- 593 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE i < $1; // 30
|li
[10]
[10]
[20]
[20]

-- 594 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i <= $1; // 30
|li
[10]
[10]
[20]
[20]
[30]
[30]

-- 595 // index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE i == $1; // 30
|li
[30]
[30]

-- 596 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i >= $1; // 30
|li
[30]
[30]
[40]
[40]
[50]
[50]

-- 597 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i > $1; // 30
|li
[50]
[50]
[40]
[40]

-- 598 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE $1 > i; // 30
|li
[10]
[10]
[20]
[20]

-- 599 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE $1 >= i; // 30
|li
[10]
[10]
[20]
[20]
[30]
[30]

-- 600 // index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE $1 == i; // 30
|li
[30]
[30]

-- 601 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE $1 <= i; // 30
|li
[30]
[30]
[40]
[40]
[50]
[50]

-- 602 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE $1 < i; // 30
|li
[50]
[50]
[40]
[40]

-- 603 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE 30 > i;
|li
[10]
[10]
[20]
[20]

-- 604 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE 30 >= i;
|li
[10]
[10]
[20]
[20]
[30]
[30]

-- 605 // index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT i FROM t WHERE 30 == i;
|li
[30]
[30]

-- 606 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE 30 <= i;
|li
[30]
[30]
[40]
[40]
[50]
[50]

-- 607 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE 30 < i;
|li
[50]
[50]
[40]
[40]

-- 608
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i < 30;
|li
[20]
[10]

-- 609 // ordered -> index is used
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE UNIQUE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(20);
	INSERT INTO t VALUES(30);
	INSERT INTO t VALUES(40);
	INSERT INTO t VALUES(50);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i < 30;
|li
[10]
[20]

-- 610
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE UNIQUE INDEX x ON t (i);
	INSERT INTO t VALUES(NULL);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(10);
	INSERT INTO t VALUES(NULL);
COMMIT;
SELECT * FROM t WHERE i < 30;
||duplicate

-- 611 // Issue #34
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42);
COMMIT;
SELECT * FROM t WHERE i == $0;
||parameter.*non zero

-- 612
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE TABLE __Table (i int);
COMMIT;
SELECT * FROM t;
||system table

-- 613
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE TABLE __Column (i int);
COMMIT;
SELECT * FROM t;
||system table

-- 614
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE TABLE __Index (i int);
COMMIT;
SELECT * FROM t;
||system table

-- 615
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	DROP TABLE __Table;
COMMIT;
SELECT * FROM t;
||system table

-- 616
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	DROP TABLE __Column;
COMMIT;
SELECT * FROM t;
||system table

-- 617
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	DROP TABLE __Index;
COMMIT;
SELECT * FROM t;
||system table

-- 618
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX __Table ON t (i);
COMMIT;
SELECT * FROM t;
||system table

-- 619
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX __Column ON t (i);
COMMIT;
SELECT * FROM t;
||system table

-- 620
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX __Index ON t (i);
COMMIT;
SELECT * FROM t;
||system table

-- 621
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON __Table (Name);
COMMIT;
SELECT * FROM t;
||system table

-- 622
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON __Column (Name);
COMMIT;
SELECT * FROM t;
||system table

-- 623
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON __Index (Name);
COMMIT;
SELECT * FROM t;
||system table

-- 624
SELECT * FROM __Table;
|?Name, ?Schema

-- 625
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
COMMIT;
SELECT * FROM __Table ORDER BY Name;
|sName, sSchema
[t CREATE TABLE t (i int64, s string);]

-- 626
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE TABLE u (b bool, i bigint, t time, d duration);
COMMIT;
SELECT * FROM __Table ORDER BY Name;
|sName, sSchema
[t CREATE TABLE t (i int64, s string);]
[u CREATE TABLE u (b bool, i bigint, t time, d duration);]

-- 627
SELECT * FROM __Column;
|?TableName, ?Ordinal, ?Name, ?Type

-- 628
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
COMMIT;
SELECT * FROM __Column ORDER BY TableName, Name;
|sTableName, lOrdinal, sName, sType
[t 1 i int64]
[t 2 s string]

-- 629
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE TABLE u (b bool, i bigint, t time, d duration);
COMMIT;
SELECT * FROM __Column ORDER BY TableName, Ordinal;
|sTableName, lOrdinal, sName, sType
[t 1 i int64]
[t 2 s string]
[u 1 b bool]
[u 2 i bigint]
[u 3 t time]
[u 4 d duration]

-- 630
SELECT * FROM __Index;
|?TableName, ?ColumnName, ?Name, ?IsUnique

-- 631
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
COMMIT;
SELECT * FROM __Index ORDER BY TableName, Name;
|?TableName, ?ColumnName, ?Name, ?IsUnique

-- 632
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (i);
COMMIT;
SELECT * FROM __Index ORDER BY TableName, ColumnName, Name;
|sTableName, sColumnName, sName, bIsUnique
[t i x false]

-- 633
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (i);
	CREATE INDEX id ON t (id());
	CREATE TABLE u (b bool, i bigint, t time, d duration);
COMMIT;
SELECT * FROM __Index ORDER BY TableName, ColumnName, Name;
|sTableName, sColumnName, sName, bIsUnique
[t i x false]
[t id() id false]

-- 634
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (i);
	CREATE INDEX id ON t (id());
	CREATE TABLE u (b bool, i bigint, t time, d duration);
	CREATE INDEX z ON u (t);
	CREATE UNIQUE INDEX y ON u (i);
COMMIT;
SELECT * FROM __Index ORDER BY TableName, ColumnName, Name;
|sTableName, sColumnName, sName, bIsUnique
[t i x false]
[t id() id false]
[u i y true]
[u t z false]

-- 635
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (i);
	CREATE INDEX id ON t (id());
	CREATE TABLE u (b bool, i bigint, t time, d duration);
	CREATE INDEX z ON u (t);
	CREATE UNIQUE INDEX y ON u (i);
COMMIT;
SELECT c.TableName, c.Ordinal, c.Name
FROM __Table AS t, __Column AS c
WHERE t.Name == "u" && t.Name == c.TableName
ORDER BY c.Ordinal;
|sc.TableName, lc.Ordinal, sc.Name
[u 1 b]
[u 2 i]
[u 3 t]
[u 4 d]

-- 636 // https://github.com/cznic/ql/issues/36
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES (1, "test");
COMMIT;
SELECT * FROM t WHERE s == "test";
|li, ss
[1 test]

-- 637 // https://github.com/cznic/ql/issues/36
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	INSERT INTO t VALUES (1, "test");
	CREATE INDEX idx_s ON t (s);
COMMIT;
SELECT * FROM t WHERE s == "test";
|li, ss
[1 test]

-- 638 // https://github.com/cznic/ql/issues/37
BEGIN TRANSACTION;
	CREATE TABLE artist (id int64, name string);
	CREATE TABLE data_types (id int64, _uint int64, _uint8 int64, _uint16
		int64, _uint32 int64, _uint64 int64, _int int64, _int8 int64,
		_int16 int64, _int32 int64, _int64 int64, _float32 float32,
		_float64 float64, _bool bool, _string string, _date time, _time
		time);
COMMIT;
SELECT * FROM __Table ORDER BY Name; // Must sort, map range is not deterministic.
|sName, sSchema
[artist CREATE TABLE artist (id int64, name string);]
[data_types CREATE TABLE data_types (id int64, _uint int64, _uint8 int64, _uint16 int64, _uint32 int64, _uint64 int64, _int int64, _int8 int64, _int16 int64, _int32 int64, _int64 int64, _float32 float32, _float64 float64, _bool bool, _string string, _date time, _time time);]

-- 639 // https://github.com/cznic/ql/issues/37
BEGIN TRANSACTION;
	CREATE TABLE artist (id int64, name string);
	CREATE TABLE data_types (id int64, _uint int64, _uint8 int64, _uint16
		int64, _uint32 int64, _uint64 int64, _int int64, _int8 int64,
		_int16 int64, _int32 int64, _int64 int64, _float32 float32,
		_float64 float64, _bool bool, _string string, _date time, _time
		time);
COMMIT;
SELECT * FROM __Table WHERE Name == "artist";
|sName, sSchema
[artist CREATE TABLE artist (id int64, name string);]

-- 640 // https://github.com/cznic/ql/issues/37
BEGIN TRANSACTION;
	CREATE TABLE artist (id int64, name string);
	CREATE TABLE data_types (id int64, _uint int64, _uint8 int64, _uint16
		int64, _uint32 int64, _uint64 int64, _int int64, _int8 int64,
		_int16 int64, _int32 int64, _int64 int64, _float32 float32,
		_float64 float64, _bool bool, _string string, _date time, _time
		time);
COMMIT;
SELECT * FROM __Column ORDER BY TableName, Ordinal;
|sTableName, lOrdinal, sName, sType
[artist 1 id int64]
[artist 2 name string]
[data_types 1 id int64]
[data_types 2 _uint int64]
[data_types 3 _uint8 int64]
[data_types 4 _uint16 int64]
[data_types 5 _uint32 int64]
[data_types 6 _uint64 int64]
[data_types 7 _int int64]
[data_types 8 _int8 int64]
[data_types 9 _int16 int64]
[data_types 10 _int32 int64]
[data_types 11 _int64 int64]
[data_types 12 _float32 float32]
[data_types 13 _float64 float64]
[data_types 14 _bool bool]
[data_types 15 _string string]
[data_types 16 _date time]
[data_types 17 _time time]

-- 641 // https://github.com/cznic/ql/issues/37
BEGIN TRANSACTION;
	CREATE TABLE artist (id int64, name string);
	CREATE TABLE data_types (id int64, _uint int64, _uint8 int64, _uint16
		int64, _uint32 int64, _uint64 int64, _int int64, _int8 int64,
		_int16 int64, _int32 int64, _int64 int64, _float32 float32,
		_float64 float64, _bool bool, _string string, _date time, _time
		time);
COMMIT;
SELECT * FROM __Column WHERE TableName == "artist" ORDER BY TableName, Ordinal;
|sTableName, lOrdinal, sName, sType
[artist 1 id int64]
[artist 2 name string]

-- 642
BEGIN TRANSACTION;
	CREATE TABLE t (i int, j int, k int);
	INSERT INTO t VALUES
		(1, 2, 3),
		(4, 5, 6);
	CREATE TABLE u (x int, y int, z int);
	INSERT INTO u VALUES
		(10, 20, 30),
		(40, 50, 60);
COMMIT;
SELECT * FROM t, u WHERE u.y < 60 && t.k < 7;
|lt.i, lt.j, lt.k, lu.x, lu.y, lu.z
[4 5 6 40 50 60]
[4 5 6 10 20 30]
[1 2 3 40 50 60]
[1 2 3 10 20 30]

-- 643 // order -> xk used
BEGIN TRANSACTION;
	CREATE TABLE t (i int, j int, k int);
	CREATE INDEX xk ON t (k);
	INSERT INTO t VALUES
		(1, 2, 3),
		(4, 5, 6);
	CREATE TABLE u (x int, y int, z int);
	INSERT INTO u VALUES
		(10, 20, 30),
		(40, 50, 60);
COMMIT;
SELECT * FROM t, u WHERE u.y < 60 && t.k < 7;
|lt.i, lt.j, lt.k, lu.x, lu.y, lu.z
[1 2 3 40 50 60]
[1 2 3 10 20 30]
[4 5 6 40 50 60]
[4 5 6 10 20 30]

-- 644 // order -> xy used
BEGIN TRANSACTION;
	CREATE TABLE t (i int, j int, k int);
	INSERT INTO t VALUES
		(1, 2, 3),
		(4, 5, 6);
	CREATE TABLE u (x int, y int, z int);
	CREATE INDEX xy ON u (y);
	INSERT INTO u VALUES
		(10, 20, 30),
		(40, 50, 60);
COMMIT;
SELECT * FROM t, u WHERE u.y < 60 && t.k < 7;
|lt.i, lt.j, lt.k, lu.x, lu.y, lu.z
[4 5 6 10 20 30]
[4 5 6 40 50 60]
[1 2 3 10 20 30]
[1 2 3 40 50 60]

-- 645 // order -> both xk and xy used
BEGIN TRANSACTION;
	CREATE TABLE t (i int, j int, k int);
	CREATE INDEX xk ON t (k);
	INSERT INTO t VALUES
		(1, 2, 3),
		(4, 5, 6);
	CREATE TABLE u (x int, y int, z int);
	CREATE INDEX xy ON u (y);
	INSERT INTO u VALUES
		(10, 20, 30),
		(40, 50, 60);
COMMIT;
SELECT * FROM t, u WHERE u.y < 60 && t.k < 7;
|lt.i, lt.j, lt.k, lu.x, lu.y, lu.z
[1 2 3 10 20 30]
[1 2 3 40 50 60]
[4 5 6 10 20 30]
[4 5 6 40 50 60]

-- 646 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
SELECT * FROM t OFFSET -1; // no rows -> not evaluated
|?i

-- 647 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
SELECT * FROM t OFFSET 0; // no rows -> not evaluated
|?i

-- 648 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
SELECT * FROM t OFFSET 1; // no rows -> not evaluated
|?i

-- 649 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() OFFSET -1;
||invalid .* -1 .*must.* non-negative

-- 650 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() OFFSET 0;
|li
[42]
[24]

-- 651 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() OFFSET 1;
|li
[24]

-- 652 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() OFFSET 2;
|?i

-- 653 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT -1; // no rows -> not evaluated
|?i

-- 654 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 0; // no rows -> not evaluated
|?i

-- 655 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 1; // no rows -> not evaluated
|?i

-- 656 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT -1;
||invalid .* -1 .*must.* non-negative

-- 657 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 0;
|?i

-- 658 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 1;
|li
[42]

-- 659 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 2;
|li
[42]
[24]

-- 660 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 3;
|li
[42]
[24]

-- 661 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 0 OFFSET 0;
|?i

-- 662 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 0 OFFSET 1;
|?i

-- 663 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 0 OFFSET 2;
|?i

-- 664 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 0 OFFSET 3;
|?i

-- 665 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 1 OFFSET 0;
|li
[42]

-- 666 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 1 OFFSET 1;
|li
[24]

-- 667 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 1 OFFSET 2;
|?i

-- 668 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 1 OFFSET 3;
|?i

-- 669 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 2 OFFSET 0;
|li
[42]
[24]

-- 670 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 2 OFFSET 1;
|li
[24]

-- 671 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 2 OFFSET 2;
|?i

-- 672 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 2 OFFSET 3;
|?i

-- 673 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 3 OFFSET 0;
|li
[42]
[24]

-- 674 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 3 OFFSET 1;
|li
[24]

-- 675 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 3 OFFSET 2;
|?i

-- 676 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(42), (24);
COMMIT;
SELECT * FROM t ORDER BY id() LIMIT 3 OFFSET 3;
|?i

-- 677 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	CREATE TABLE u (i int);
	INSERT INTO u VALUES(10), (20), (30);
COMMIT;
SELECT * FROM
	(SELECT * FROM t ORDER BY i LIMIT 2 OFFSET 1;) AS a,
	(SELECT * FROM u ORDER BY i) AS b,
ORDER BY a.i, b.i;
|la.i, lb.i
[2 10]
[2 20]
[2 30]
[3 10]
[3 20]
[3 30]

-- 678 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	CREATE TABLE u (i int);
	INSERT INTO u VALUES(10), (20), (30);
COMMIT;
SELECT * FROM
	(SELECT * FROM t ORDER BY i LIMIT 2 OFFSET 1;) AS a,
	(SELECT * FROM u ORDER BY i OFFSET 1) AS b,
ORDER BY a.i, b.i;
|la.i, lb.i
[2 20]
[2 30]
[3 20]
[3 30]

-- 679 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	CREATE TABLE u (i int);
	INSERT INTO u VALUES(10), (20), (30);
COMMIT;
SELECT * FROM
	(SELECT * FROM t ORDER BY i LIMIT 2 OFFSET 1;) AS a,
	(SELECT * FROM u ORDER BY i LIMIT 1) AS b,
ORDER BY a.i, b.i;
|la.i, lb.i
[2 10]
[3 10]

-- 680 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	CREATE TABLE u (i int);
	INSERT INTO u VALUES(10), (20), (30);
COMMIT;
SELECT * FROM
	(SELECT * FROM t ORDER BY i LIMIT 2 OFFSET 1;) AS a,
	(SELECT * FROM u ORDER BY i LIMIT 1 OFFSET 1) AS b,
ORDER BY a.i, b.i;
|la.i, lb.i
[2 20]
[3 20]

-- 681 // https://github.com/cznic/ql/issues/41
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	CREATE TABLE u (i int);
	INSERT INTO u VALUES(10), (20), (30);
COMMIT;
SELECT * FROM
	(SELECT * FROM t ORDER BY i LIMIT 2 OFFSET 1;) AS a,
	(SELECT * FROM u ORDER BY i LIMIT 1 OFFSET 1) AS b,
ORDER BY a.i, b.i
LIMIT 1;
|la.i, lb.i
[2 20]

-- 682 // https://github.com/cznic/ql/issues/42
BEGIN TRANSACTION;
  DROP TABLE IF EXISTS fibonacci;
  CREATE TABLE fibonacci(
    input int,
    output int
  );
COMMIT;

BEGIN TRANSACTION;
  INSERT INTO fibonacci (input, output) VALUES (0, 0);
  INSERT INTO fibonacci (input, output) VALUES (1, 1);
  INSERT INTO fibonacci (input, output) VALUES (2, 1);
  INSERT INTO fibonacci (input, output) VALUES (3, 2);
  INSERT INTO fibonacci (input, output) VALUES (4, 3);
  INSERT INTO fibonacci (input, output) VALUES (5, 5);
  INSERT INTO fibonacci (input, output) VALUES (6, 8);
  INSERT INTO fibonacci (input, output) VALUES (7, 13);
  INSERT INTO fibonacci (input, output) VALUES (8, 21);
  INSERT INTO fibonacci (input, output) VALUES (9, 34);
COMMIT;

--' Should print 4.
SELECT count(1) AS total FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3;
|ltotal
[4]

-- 683 // https://github.com/cznic/ql/issues/42
BEGIN TRANSACTION;
  DROP TABLE IF EXISTS fibonacci;
  CREATE TABLE fibonacci(
    input int,
    output int
  );
COMMIT;

BEGIN TRANSACTION;
  INSERT INTO fibonacci (input, output) VALUES (0, 0);
  INSERT INTO fibonacci (input, output) VALUES (1, 1);
  INSERT INTO fibonacci (input, output) VALUES (2, 1);
  INSERT INTO fibonacci (input, output) VALUES (3, 2);
  INSERT INTO fibonacci (input, output) VALUES (4, 3);
  INSERT INTO fibonacci (input, output) VALUES (5, 5);
  INSERT INTO fibonacci (input, output) VALUES (6, 8);
  INSERT INTO fibonacci (input, output) VALUES (7, 13);
  INSERT INTO fibonacci (input, output) VALUES (8, 21);
  INSERT INTO fibonacci (input, output) VALUES (9, 34);
COMMIT;

--' Should output (6, 8) (5, 5).
SELECT * FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3 ORDER BY input DESC LIMIT 2 OFFSET 1;
|linput, loutput
[6 8]
[5 5]

-- 684 // https://github.com/cznic/ql/issues/42
BEGIN TRANSACTION;
  DROP TABLE IF EXISTS fibonacci;
  CREATE TABLE fibonacci(
    input int,
    output int
  );
COMMIT;

BEGIN TRANSACTION;
  INSERT INTO fibonacci (input, output) VALUES (0, 0);
  INSERT INTO fibonacci (input, output) VALUES (1, 1);
  INSERT INTO fibonacci (input, output) VALUES (2, 1);
  INSERT INTO fibonacci (input, output) VALUES (3, 2);
  INSERT INTO fibonacci (input, output) VALUES (4, 3);
  INSERT INTO fibonacci (input, output) VALUES (5, 5);
  INSERT INTO fibonacci (input, output) VALUES (6, 8);
  INSERT INTO fibonacci (input, output) VALUES (7, 13);
  INSERT INTO fibonacci (input, output) VALUES (8, 21);
  INSERT INTO fibonacci (input, output) VALUES (9, 34);
  --' Let's delete 4 rows.
  // Delete where input == 5, input == 6, input == 7 or input == 3
  DELETE FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3;
COMMIT;
SELECT * FROM fibonacci ORDER BY input;
|linput, loutput
[0 0]
[1 1]
[2 1]
[4 3]
[8 21]
[9 34]

-- 685 // https://github.com/cznic/ql/issues/42
BEGIN TRANSACTION;
  DROP TABLE IF EXISTS fibonacci;
  CREATE TABLE fibonacci(
    input int,
    output int
  );
COMMIT;

BEGIN TRANSACTION;
  INSERT INTO fibonacci (input, output) VALUES (0, 0);
  INSERT INTO fibonacci (input, output) VALUES (1, 1);
  INSERT INTO fibonacci (input, output) VALUES (2, 1);
  INSERT INTO fibonacci (input, output) VALUES (3, 2);
  INSERT INTO fibonacci (input, output) VALUES (4, 3);
  INSERT INTO fibonacci (input, output) VALUES (5, 5);
  INSERT INTO fibonacci (input, output) VALUES (6, 8);
  INSERT INTO fibonacci (input, output) VALUES (7, 13);
  INSERT INTO fibonacci (input, output) VALUES (8, 21);
  INSERT INTO fibonacci (input, output) VALUES (9, 34);
COMMIT;
--' Let's delete 4 rows.
BEGIN TRANSACTION;
  // Delete where input == 5, input == 6, input == 7 or input == 3
  DELETE FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3;
COMMIT;
SELECT * FROM fibonacci ORDER BY input;
|linput, loutput
[0 0]
[1 1]
[2 1]
[4 3]
[8 21]
[9 34]

-- 686 // https://github.com/cznic/ql/issues/42
BEGIN TRANSACTION;
  DROP TABLE IF EXISTS fibonacci;
  CREATE TABLE fibonacci(
    input int,
    output int
  );
COMMIT;

BEGIN TRANSACTION;
  INSERT INTO fibonacci (input, output) VALUES (0, 0);
  INSERT INTO fibonacci (input, output) VALUES (1, 1);
  INSERT INTO fibonacci (input, output) VALUES (2, 1);
  INSERT INTO fibonacci (input, output) VALUES (3, 2);
  INSERT INTO fibonacci (input, output) VALUES (4, 3);
  INSERT INTO fibonacci (input, output) VALUES (5, 5);
  INSERT INTO fibonacci (input, output) VALUES (6, 8);
  INSERT INTO fibonacci (input, output) VALUES (7, 13);
  INSERT INTO fibonacci (input, output) VALUES (8, 21);
  INSERT INTO fibonacci (input, output) VALUES (9, 34);
  --' Let's delete 4 rows.
  // Delete where input == 5, input == 6, input == 7 or input == 3
  DELETE FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3;
COMMIT;
--' Try to count the rows we've just deleted, using the very same condition. Result is 1, should be 0.
SELECT count() AS total FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3;
|ltotal
[0]

-- 687 // https://github.com/cznic/ql/issues/42
BEGIN TRANSACTION;
  DROP TABLE IF EXISTS fibonacci;
  CREATE TABLE fibonacci(
    input int,
    output int
  );
COMMIT;

BEGIN TRANSACTION;
  INSERT INTO fibonacci (input, output) VALUES (0, 0);
  INSERT INTO fibonacci (input, output) VALUES (1, 1);
  INSERT INTO fibonacci (input, output) VALUES (2, 1);
  INSERT INTO fibonacci (input, output) VALUES (3, 2);
  INSERT INTO fibonacci (input, output) VALUES (4, 3);
  INSERT INTO fibonacci (input, output) VALUES (5, 5);
  INSERT INTO fibonacci (input, output) VALUES (6, 8);
  INSERT INTO fibonacci (input, output) VALUES (7, 13);
  INSERT INTO fibonacci (input, output) VALUES (8, 21);
  INSERT INTO fibonacci (input, output) VALUES (9, 34);
COMMIT;
BEGIN TRANSACTION;
  --' Let's delete 4 rows.
  // Delete where input == 5, input == 6, input == 7 or input == 3
  DELETE FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3;
COMMIT;
--' Try to count the rows we've just deleted, using the very same condition. Result is 1, should be 0.
SELECT count() AS total FROM fibonacci WHERE input >= 5 && input <= 7 &oror; input == 3;
|ltotal
[0]

-- 688
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1);
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t ORDER BY i;
|?i

-- 689
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2);
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[2]

-- 690
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2);
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]

-- 691
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[2]
[3]

-- 692
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[3]

-- 693
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3);
	DELETE FROM t WHERE i == 3;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[2]

-- 694
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[2]
[3]
[4]

-- 695
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[3]
[4]

-- 696
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 3;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[2]
[4]

-- 697
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 4;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[2]
[3]

-- 698
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 1;
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[3]
[4]

-- 699
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 2;
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[3]
[4]

-- 700
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 2;
	DELETE FROM t WHERE i == 3;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[4]

-- 701
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 3;
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[4]

-- 702
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 3;
	DELETE FROM t WHERE i == 4;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[2]

-- 703
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 4;
	DELETE FROM t WHERE i == 3;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[2]

-- 704
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 1;
	DELETE FROM t WHERE i == 3;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[2]
[4]

-- 705
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 3;
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[2]
[4]

-- 706
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 1;
	DELETE FROM t WHERE i == 4;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[2]
[3]

-- 707
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 4;
	DELETE FROM t WHERE i == 1;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[2]
[3]

-- 708
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 2;
	DELETE FROM t WHERE i == 4;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[3]

-- 709
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES(1), (2), (3), (4);
	DELETE FROM t WHERE i == 4;
	DELETE FROM t WHERE i == 2;
COMMIT;
SELECT * FROM t ORDER BY i;
|li
[1]
[3]

-- 710 // https://github.com/cznic/ql/issues/43
SELECT Name, Unique FROM __Index;
||syntax error

-- 711
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
COMMIT;
SELECT s FROM t;
|ss
[foo]
[bar]

-- 712
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
COMMIT;
SELECT * FROM x;
|sx
[bar]
[foo]

-- 713
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (s);
	CREATE INDEX x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
COMMIT;
SELECT * FROM x;
||already

-- 714
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
COMMIT;
SELECT * FROM x;
|sx
[bar]
[foo]

-- 715
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX x ON t (s);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
COMMIT;
SELECT * FROM x;
|sx
[bar]
[foo]

-- 716
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
COMMIT;
SELECT s FROM t WHERE s != "z";
|ss
[foo]
[bar]

-- 717
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
COMMIT;
SELECT s FROM t WHERE s < "z";
|ss
[bar]
[foo]

-- 718
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
	DROP INDEX x;
COMMIT;
SELECT s FROM t WHERE s < "z";
|ss
[foo]
[bar]

-- 719
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
	DROP INDEX x;
	DROP INDEX x;
COMMIT;
SELECT s FROM t WHERE s < "z";
||does not exist

-- 720
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
	DROP INDEX IF EXISTS x;
	DROP INDEX x;
COMMIT;
SELECT s FROM t WHERE s < "z";
||does not exist

-- 721
BEGIN TRANSACTION;
	CREATE TABLE t (i int, s string);
	CREATE INDEX IF NOT EXISTS x ON t (s);
	INSERT INTO t VALUES (1, "bar"), (2, "foo");
	DROP INDEX x;
	DROP INDEX IF EXISTS x;
COMMIT;
SELECT s FROM t WHERE s < "z";
|ss
[foo]
[bar]

-- 722
BEGIN TRANSACTION;
	CREATE TABLE t (p string, c blob);
	CREATE UNIQUE INDEX x ON t (p);
	INSERT INTO t VALUES
		("empty", blob("")),
	;
COMMIT;
SELECT p, string(c) FROM t;
|sp, s
[empty ]

-- 723
BEGIN TRANSACTION;
	CREATE TABLE t (p string, c blob);
	CREATE INDEX x ON t (p);
	INSERT INTO t VALUES
		("empty", blob("")),
	;
COMMIT;
BEGIN TRANSACTION;
	DELETE FROM t WHERE p == "empty";
COMMIT;
SELECT p, string(c) FROM t;
|?p, ?

-- 724
BEGIN TRANSACTION;
	CREATE TABLE t (p string, c blob);
	CREATE UNIQUE INDEX x ON t (p);
	INSERT INTO t VALUES
		("empty", blob("")),
	;
COMMIT;
BEGIN TRANSACTION;
	DELETE FROM t WHERE p == "empty";
COMMIT;
SELECT p, string(c) FROM t;
|?p, ?

-- S 725
BEGIN TRANSACTION;
	UPDATE none SET
		DepartmentID = DepartmentID+1000,
	WHERE DepartmentID == 33;
COMMIT;
SELECT * FROM employee;
||table.*not.*exist

-- S 726
BEGIN TRANSACTION;
	UPDATE employee SET
		FirstName = "John"
	WHERE DepartmentID == 33;
COMMIT;
SELECT * FROM employee;
||unknown.*FirstName

-- S 727
BEGIN TRANSACTION;
	UPDATE employee SET
		DepartmentID = DepartmentID+1000,
	WHERE DepartmentID == 33;
COMMIT;
SELECT * FROM employee
ORDER BY LastName;
|sLastName, lDepartmentID
[Heisenberg 1033]
[John <nil>]
[Jones 1033]
[Rafferty 31]
[Robinson 34]
[Smith 34]

-- 728 // https://github.com/cznic/ql/issues/49
BEGIN TRANSACTION;
	CREATE TABLE IF NOT EXISTS t (username string, departname string, created time, detail_id int, height float64, avatar blob, is_man bool);
	CREATE UNIQUE INDEX UQE_userinfo_username ON t (username);
	INSERT INTO t (username, departname, created, detail_id, height, avatar, is_man) VALUES ( 
		"xiaolunwen",
		"dev",
		now(),
		1,
		1.78,
		blob("012"),
		true,
	);
	DELETE FROM t WHERE id() == 1;
COMMIT;
SELECT * FROM t;
|?username, ?departname, ?created, ?detail_id, ?height, ?avatar, ?is_man

-- 729 // https://github.com/cznic/ql/issues/49
BEGIN TRANSACTION;
	CREATE TABLE IF NOT EXISTS t (username string, departname string, created time, detail_id int, height float64, avatar blob, is_man bool);
	CREATE UNIQUE INDEX UQE_userinfo_username ON t (username);
	INSERT INTO t (username, departname, created, detail_id, height, avatar, is_man) VALUES ( 
		"xiaolunwen",
		"dev",
		now(),
		1,
		1.78,
		__testBlob(256),
		true,
	);
	DELETE FROM t WHERE id() == 1;
COMMIT;
SELECT * FROM t;
|?username, ?departname, ?created, ?detail_id, ?height, ?avatar, ?is_man

-- 730 // https://github.com/cznic/ql/issues/49
BEGIN TRANSACTION;
	CREATE TABLE IF NOT EXISTS t (username string, departname string, created time, detail_id int, height float64, avatar blob, is_man bool);
	CREATE UNIQUE INDEX UQE_userinfo_username ON t (username);
	INSERT INTO t (username, departname, created, detail_id, height, avatar, is_man) VALUES ( 
		"xiaolunwen",
		"dev",
		now(),
		1,
		1.78,
		__testBlob(1<<16),
		true,
	);
	DELETE FROM t WHERE id() == 1;
COMMIT;
SELECT * FROM t;
|?username, ?departname, ?created, ?detail_id, ?height, ?avatar, ?is_man

-- 731 // https://github.com/cznic/ql/issues/49
BEGIN TRANSACTION;
	CREATE TABLE IF NOT EXISTS t (username string, departname string, created time, detail_id int, height float64, avatar blob, is_man bool);
	CREATE UNIQUE INDEX UQE_userinfo_username ON t (username);
	INSERT INTO t (username, departname, created, detail_id, height, avatar, is_man) VALUES ( 
		"xiaolunwen",
		"dev",
		now(),
		1,
		1.78,
		__testBlob(1<<20),
		true,
	);
	DELETE FROM t WHERE id() == 1;
COMMIT;
SELECT * FROM t;
|?username, ?departname, ?created, ?detail_id, ?height, ?avatar, ?is_man

-- 732 // https://github.com/cznic/ql/issues/49
BEGIN TRANSACTION;
	CREATE TABLE IF NOT EXISTS t (username string, departname string, created time, detail_id int, height float64, avatar blob, is_man bool);
	CREATE UNIQUE INDEX UQE_userinfo_username ON t (username);
	INSERT INTO t (username, departname, created, detail_id, height, avatar, is_man) VALUES ( 
		"xiaolunwen",
		"dev",
		now(),
		1,
		1.78,
		__testBlob(1<<20),
		true,
	), (
		"2xiaolunwen",
		"2dev",
		now(),
		2,
		2.78,
		__testBlob(1<<21),
		true,
	);
	DELETE FROM t WHERE id() == 1;
COMMIT;
SELECT id() == 2, username == "2xiaolunwen", len(string(avatar)) == 1<<21 FROM t;
|b, b, b
[true true true]

-- 733 // https://github.com/cznic/ql/issues/49
BEGIN TRANSACTION;
	CREATE TABLE IF NOT EXISTS t (username string, departname string, created time, detail_id int, height float64, avatar blob, is_man bool);
	CREATE UNIQUE INDEX UQE_userinfo_username ON t (username);
	INSERT INTO t (username, departname, created, detail_id, height, avatar, is_man) VALUES ( 
		"xiaolunwen",
		"dev",
		now(),
		1,
		1.78,
		__testBlob(1<<20),
		true,
	), (
		"2xiaolunwen",
		"2dev",
		now(),
		2,
		2.78,
		__testBlob(1<<21),
		true,
	);
	DELETE FROM t WHERE id() == 2;
COMMIT;
SELECT id() == 1, username == "xiaolunwen", len(string(avatar)) == 1<<20 FROM t;
|b, b, b
[true true true]

-- 734 // https://github.com/cznic/ql/issues/51
BEGIN TRANSACTION;
	CREATE TABLE IF NOT EXISTS no_id_user (user string, remain int, total int);
	CREATE UNIQUE INDEX UQE_no_id_user_user ON no_id_user (user);
	DELETE FROM no_id_user WHERE user == "xlw";
	INSERT INTO no_id_user (user, remain, total) VALUES ("xlw", 20, 100);
COMMIT;
SELECT user, remain, total FROM no_id_user WHERE user == "xlw" LIMIT 1;
|suser, lremain, ltotal
[xlw 20 100]

-- 735
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	INSERT INTO t VALUES (1), (2), (3), (4), (5), (6);
COMMIT;
SELECT * FROM t WHERE id() < 4; // reverse order -> no index used
|li
[3]
[2]
[1]

-- 736
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES (1), (2), (3), (4), (5), (6);
COMMIT;
SELECT * FROM t WHERE id() < 4; // ordered -> index is used
|li
[1]
[2]
[3]

-- 737
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES (1), (2), (3), (4), (5), (6);
COMMIT;
SELECT * FROM t WHERE id() <= 4; // ordered -> index is used
|li
[1]
[2]
[3]
[4]

-- 738
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES (1), (2), (3), (4), (5), (6);
COMMIT;
SELECT * FROM t WHERE id() == 4;
|li
[4]

-- 739
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES (1), (2), (3), (4), (5), (6);
COMMIT;
SELECT * FROM t WHERE id() >= 4; // ordered -> index is used
|li
[4]
[5]
[6]

-- 740
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (id());
	INSERT INTO t VALUES (1), (2), (3), (4), (5), (6);
COMMIT;
SELECT * FROM t WHERE id() > 4;
|li
[6]
[5]

-- 741
BEGIN TRANSACTION;
	CREATE TABLE t (i int);
	CREATE INDEX x ON t (id());
	CREATE TABLE u (i int);
	CREATE INDEX y ON u (i);
	INSERT INTO t VALUES (1), (2), (3), (4), (5), (6);
	INSERT INTO u VALUES (10), (20), (30), (40), (50), (60);
COMMIT;
SELECT t.ID, t.i, u.i FROM
	(SELECT id() as ID, i FROM t WHERE id() < 4) AS t,
	(SELECT * FROM u WHERE i < 40) AS u; // ordered -> both indices are used
|lt.ID, lt.i, lu.i
[1 1 10]
[1 1 20]
[1 1 30]
[2 2 10]
[2 2 20]
[2 2 30]
[3 3 10]
[3 3 20]
[3 3 30]

-- 742 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(NULL),
	;
COMMIT;
SELECT max(t) as T FROM t;
|?T
[<nil>]

-- 743 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(NULL),
		(NULL),
	;
COMMIT;
SELECT max(t) as T FROM t;
|?T
[<nil>]

-- 744 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(NULL),
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
	;
COMMIT;
SELECT max(t) as T FROM t;
|?T
[2014-08-08 14:05:11 +0000 UTC]

-- 745 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
		(NULL),
	;
COMMIT;
SELECT max(t) as T FROM t;
|?T
[2014-08-08 14:05:11 +0000 UTC]

-- 746 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:12")),
	;
COMMIT;
SELECT max(t) as T FROM t;
|?T
[2014-08-08 14:05:12 +0000 UTC]

-- 747 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:12")),
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
	;
COMMIT;
SELECT max(t) as T FROM t;
|?T
[2014-08-08 14:05:12 +0000 UTC]

-- 748 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(NULL),
	;
COMMIT;
SELECT min(t) as T FROM t;
|?T
[<nil>]

-- 749 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(NULL),
		(NULL),
	;
COMMIT;
SELECT min(t) as T FROM t;
|?T
[<nil>]

-- 750 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(NULL),
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
	;
COMMIT;
SELECT min(t) as T FROM t;
|?T
[2014-08-08 14:05:11 +0000 UTC]

-- 751 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
		(NULL),
	;
COMMIT;
SELECT min(t) as T FROM t;
|?T
[2014-08-08 14:05:11 +0000 UTC]

-- 752 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:12")),
	;
COMMIT;
SELECT min(t) as T FROM t;
|?T
[2014-08-08 14:05:11 +0000 UTC]

-- 753 // https://github.com/cznic/ql/pull/65
BEGIN TRANSACTION;
	CREATE TABLE t (t time);
	INSERT INTO t VALUES
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:12")),
		(parseTime("2006-01-02 15:04:05", "2014-08-08 14:05:11")),
	;
COMMIT;
SELECT min(t) as T FROM t;
|?T
[2014-08-08 14:05:11 +0000 UTC]

-- 754 // https://github.com/cznic/ql/issues/68
BEGIN TRANSACTION;
	CREATE TABLE department (Name string);
	INSERT INTO department (Name) VALUES ("small"), ("large"), ("medium");
	SELECT * FROM department;
	ALTER TABLE department ADD score float;
	SELECT * from department;
	UPDATE department SET score=0 WHERE Name=="small";
COMMIT;
SELECT * FROM department ORDER BY Name;
|sName, ?score
[large <nil>]
[medium <nil>]
[small 0]

-- 755
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s LIKE "foo"
ORDER BY id();
|l, ss
[1 seafood]
[2 A fool on the hill]
[5 foobar]

-- 756
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE !(s LIKE "foo")
ORDER BY id();
|l, ss
[4 barbaz]

-- 757
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s LIKE "foo" IS NULL
ORDER BY id();
|l, ?s
[3 <nil>]

-- 758
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s LIKE "foo" IS NOT NULL
ORDER BY id();
|l, ss
[1 seafood]
[2 A fool on the hill]
[4 barbaz]
[5 foobar]

-- 759
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s LIKE "bar"
ORDER BY id();
|l, ss
[4 barbaz]
[5 foobar]

-- 760
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s LIKE "^bar"
ORDER BY id();
|l, ss
[4 barbaz]

-- 761
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s LIKE "bar$"
ORDER BY id();
|l, ss
[5 foobar]

-- 762
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s LIKE "bar"+"$"
ORDER BY id();
|l, ss
[5 foobar]

-- 763
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s+"qux" LIKE "qux"+"$"
ORDER BY id();
|l, ss
[1 seafood]
[2 A fool on the hill]
[4 barbaz]
[5 foobar]

-- 764
BEGIN TRANSACTION;
	CREATE TABLE t (s string);
	INSERT INTO t VALUES
		("seafood"),
		("A fool on the hill"),
		(NULL),
		("barbaz"),
		("foobar"),
	;
COMMIT;
SELECT id(), s
FROM t
WHERE s+"quxx" LIKE "qux"+"$"
ORDER BY id();
|?, ?s

-- 765 // https://github.com/cznic/ql/issues/75
BEGIN TRANSACTION;
	CREATE TABLE foo (i int);
	INSERT INTO foo VALUES (10), (20);
	CREATE TABLE bar (fooID int, s string);
	INSERT INTO bar SELECT id(), "ten" FROM foo WHERE i == 10;
	INSERT INTO bar SELECT id(), "twenty" FROM foo WHERE i == 20;
COMMIT;
SELECT *
FROM
	(SELECT id() AS ID, i FROM foo) AS foo,
	bar
WHERE bar.fooID == foo.ID
ORDER BY foo.ID;
|lfoo.ID, lfoo.i, lbar.fooID, sbar.s
[1 10 1 ten]
[2 20 2 twenty]

-- 766 // https://github.com/cznic/ql/issues/75
BEGIN TRANSACTION;
	CREATE TABLE foo (i int);
	INSERT INTO foo VALUES (10), (20);
	CREATE TABLE bar (fooID int, s string);
	INSERT INTO bar SELECT id(), "ten" FROM foo WHERE i == 10;
	INSERT INTO bar SELECT id(), "twenty" FROM foo WHERE i == 20;
COMMIT;
SELECT *
FROM foo, bar
WHERE bar.fooID == id(foo)
ORDER BY id(foo);
|lfoo.i, lbar.fooID, sbar.s
[10 1 ten]
[20 2 twenty]

-- 767 // https://github.com/cznic/ql/issues/81
BEGIN TRANSACTION;
	CREATE TABLE t (name string, mail string);
	INSERT INTO t VALUES
		("a", "foo@example.com"),
		("b", "bar@example.com"),
		("c", "baz@example.com"),
		("d", "foo@example.com"),
		("e", "bar@example.com"),
		("f", "baz@example.com"),
	;
COMMIT;
SELECT *
FROM t
WHERE name == "b" AND mail == "bar@example.com";
|sname, smail
[b bar@example.com]

-- 768 // https://github.com/cznic/ql/issues/81
BEGIN TRANSACTION;
	CREATE TABLE t (name string, mail string);
	INSERT INTO t VALUES
		("a", "foo@example.com"),
		("b", "bar@example.com"),
		("c", "baz@example.com"),
		("d", "foo@example.com"),
		("e", "bar@example.com"),
		("f", "baz@example.com"),
	;
COMMIT;
SELECT *
FROM t
WHERE name == "b" and mail == "bar@example.com";
|sname, smail
[b bar@example.com]

-- 769 // https://github.com/cznic/ql/issues/81
BEGIN TRANSACTION;
	CREATE TABLE t (name string, mail string);
	INSERT INTO t VALUES
		("a", "foo@example.com"),
		("b", "bar@example.com"),
		("c", "baz@example.com"),
		("d", "foo@example.com"),
		("e", "bar@example.com"),
		("f", "baz@example.com"),
	;
COMMIT;
SELECT *
FROM t
WHERE name == "b" OR mail == "bar@example.com"
ORDER BY name;
|sname, smail
[b bar@example.com]
[e bar@example.com]

-- 770 // https://github.com/cznic/ql/issues/81
BEGIN TRANSACTION;
	CREATE TABLE t (name string, mail string);
	INSERT INTO t VALUES
		("a", "foo@example.com"),
		("b", "bar@example.com"),
		("c", "baz@example.com"),
		("d", "foo@example.com"),
		("e", "bar@example.com"),
		("f", "baz@example.com"),
	;
COMMIT;
SELECT *
FROM t
WHERE name == "b" or mail == "bar@example.com"
ORDER BY name;
|sname, smail
[b bar@example.com]
[e bar@example.com]

-- 771 // https://github.com/cznic/ql/issues/72
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
COMMIT;
SELECT id(), i FROM tableA WHERE id() IN (SELECT idA FROM tableB) ORDER BY id();
|l, li
[2 12]
[4 14]
[6 16]

-- 772 // https://github.com/cznic/ql/issues/72
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
COMMIT;
SELECT id(), i FROM tableA WHERE id() NOT IN (SELECT idA FROM tableB) ORDER BY id();
|l, li
[1 11]
[3 13]
[5 15]

-- 773 // https://github.com/cznic/ql/issues/72
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE id() IN (SELECT idA FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
|l, li
[1 11]
[3 13]
[5 15]

-- 774 // https://github.com/cznic/ql/issues/72
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE id() NOT IN (SELECT idA FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
|l, li
[2 12]
[4 14]
[6 16]

-- 775 // https://github.com/cznic/ql/issues/72, coerce
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE 2 IN (SELECT idA FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
|?, ?i

-- 776 // https://github.com/cznic/ql/issues/72, coerce
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE 2 NOT IN (SELECT idA FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
|l, li
[1 11]
[2 12]
[3 13]
[4 14]
[5 15]
[6 16]

-- 777 // https://github.com/cznic/ql/issues/72, different types have zero set intersetion.
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE 3.14 IN (SELECT idA FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
|l, li
[1 11]
[2 12]
[3 13]
[4 14]
[5 15]
[6 16]

-- 778 // https://github.com/cznic/ql/issues/72, different have zero set intersection but NOT makes the result true.
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE 3.14 NOT IN (SELECT idA FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
|?, ?i

-- 779 // https://github.com/cznic/ql/issues/72, invalid field type
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA time);
	INSERT INTO tableB
		SELECT now() FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE 3.14 NOT IN (SELECT idA FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
||invalid field type

-- 780 // https://github.com/cznic/ql/issues/72, too many fields
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int, name string);
	INSERT INTO tableB
		SELECT id(), "foo" FROM tableA WHERE i&1 == 0;
	DELETE FROM tableA WHERE id() NOT IN (SELECT * FROM tableB);
COMMIT;
SELECT id(), i FROM tableA ORDER BY id();
||mismatched field count

-- 781 // https://github.com/cznic/ql/issues/72, some NULL
BEGIN TRANSACTION;
	DROP TABLE IF EXISTS tableA;
	DROP TABLE IF EXISTS tableB;
COMMIT;
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB VALUES(NULL);
	INSERT INTO tableB
		SELECT id() FROM tableA WHERE i&1 == 0;
COMMIT;
SELECT i FROM tableA WHERE id() IN (SELECT idA from tableB) ORDER BY id();
|li
[12]
[14]
[16]

-- 782 // https://github.com/cznic/ql/issues/72, all NULL
BEGIN TRANSACTION;
	DROP TABLE IF EXISTS tableA;
	DROP TABLE IF EXISTS tableB;
COMMIT;
BEGIN TRANSACTION;
	CREATE TABLE tableA (i int);
	INSERT INTO tableA VALUES
		(11),
		(12),
		(13),
		(14),
		(15),
		(16),
	;
	CREATE TABLE tableB (idA int);
	INSERT INTO tableB VALUES(NULL);
COMMIT;
SELECT i FROM tableA WHERE id() IN (SELECT idA from tableB) ORDER BY id();
|?i
