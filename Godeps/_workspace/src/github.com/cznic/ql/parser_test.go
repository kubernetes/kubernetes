// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"testing"
)

func TestParser0(t *testing.T) {
	table := []struct {
		src string
		ok  bool
	}{
		{"", true},
		{";", true},
		{"CREATE", false},
		{"CREATE TABLE", false},
		{"CREATE TABLE foo (", false},
		// 5
		{"CREATE TABLE foo ()", false},
		{"CREATE TABLE foo ();", false},
		{"CREATE TABLE foo (a byte)", true},
		{"CREATE TABLE foo (a uint8);", true},
		{"CREATE TABLE foo (a uint16, b uint32)", true},
		// 10
		{"CREATE TABLE foo (a uint64, b bool);", true},
		{"CREATE TABLE foo (a int8, b int16) CREATE TABLE bar (x int32, y int64)", false},
		{"CREATE TABLE foo (a int, b float32); CREATE TABLE bar (x float64, y float)", true},
		{"INSERT INTO foo VALUES (1234)", true},
		{"INSERT INTO foo VALUES (1234, 5678)", true},
		// 15
		{"INSERT INTO foo VALUES (1 || 2)", false},
		{"INSERT INTO foo VALUES (1 | 2)", true},
		{"INSERT INTO foo VALUES (false || true)", true},
		{"INSERT INTO foo VALUES (id())", true},
		{"INSERT INTO foo VALUES (bar(5678))", false},
		// 20
		{"INSERT INTO foo VALUES ()", false},
		{"CREATE TABLE foo (a.b, b);", false},
		{"CREATE TABLE foo (a, b.c);", false},
		{"SELECT * FROM t", true},
		{"SELECT * FROM t AS u", true},
		// 25
		{"SELECT * FROM t, v", true},
		{"SELECT * FROM t AS u, v", true},
		{"SELECT * FROM t, v AS w", true},
		{"SELECT * FROM t AS u, v AS w", true},
		{"SELECT * FROM foo, bar, foo", true},
		// 30
		{"CREATE TABLE foo (a bytes)", false},
		{"SELECT DISTINCTS * FROM t", false},
		{"SELECT DISTINCT * FROM t", true},
		{"INSERT INTO foo (a) VALUES (42)", true},
		{"INSERT INTO foo (a,) VALUES (42,)", true},
		// 35
		{"INSERT INTO foo (a,b) VALUES (42,314)", true},
		{"INSERT INTO foo (a,b,) VALUES (42,314)", true},
		{"INSERT INTO foo (a,b,) VALUES (42,314,)", true},
		{"CREATE TABLE foo (a uint16, b uint32,)", true},
		{"CREATE TABLE foo (a uint16, b uint32,) -- foo", true},
		// 40
		{"CREATE TABLE foo (a uint16, b uint32,) // foo", true},
		{"CREATE TABLE foo (a uint16, b uint32,) /* foo */", true},
		{"CREATE TABLE foo /* foo */ (a uint16, b uint32,) /* foo */", true},
		{`-- Examples
		ALTER TABLE Stock ADD Qty int;
	
		ALTER TABLE Income DROP COLUMN Taxes;
	
		CREATE TABLE department
		(
			DepartmentID   int,
			DepartmentName string,	// optional comma
		);
	
		CREATE TABLE employee
		(
			LastName	string,
			DepartmentID	int	// optional comma
		);
	
		DROP TABLE Inventory;
			
		INSERT INTO department (DepartmentID) VALUES (42);
	
		INSERT INTO department (
			DepartmentName,
			DepartmentID,
		)
		VALUES (
			"R&D",
			42,
		);
	
		INSERT INTO department VALUES (
			42,
			"R&D",
		);
	
		SELECT * FROM Stock;
	
		SELECT DepartmentID
		FROM department
		WHERE DepartmentID == 42
		ORDER BY DepartmentName;
	
		SELECT employee.LastName
		FROM department, employee
		WHERE department.DepartmentID == employee.DepartmentID
		ORDER BY DepartmentID;
	
		SELECT a.b, c.d
		FROM
			x AS a,
			(
				SELECT * FROM y; // optional semicolon
			) AS c
		WHERE a.e > c.e;
	
		SELECT a.b, c.d
		FROM
			x AS a,
			(
				SELECT * FROM y // no semicolon
			) AS c
		WHERE a.e > c.e;
	
		TRUNCATE TABLE department;
	
	 	SELECT DepartmentID
	 	FROM department
	 	WHERE DepartmentID == ?1
	 	ORDER BY DepartmentName;
	
	 	SELECT employee.LastName
	 	FROM department, employee
	 	WHERE department.DepartmentID == $1 && employee.LastName > $2
	 	ORDER BY DepartmentID;

		`, true},
		{"BEGIN TRANSACTION", true},
		// 45
		{"COMMIT", true},
		{"ROLLBACK", true},
		{`
		BEGIN TRANSACTION;
			INSERT INTO foo VALUES (42, 3.14);
			INSERT INTO foo VALUES (-1, 2.78);
		COMMIT;`, true},
		{`
		BEGIN TRANSACTION;
			INSERT INTO AccountA (Amount) VALUES ($1);
			INSERT INTO AccountB (Amount) VALUES (-$1);
		COMMIT;`, true},
		{` // A
		BEGIN TRANSACTION;
			INSERT INTO tmp SELECT * from bar;
		SELECT * from tmp;

		// B
		ROLLBACK;`, true},
		// 50
		{`-- 6
			ALTER TABLE none DROP COLUMN c1;
		`, true},
	}

	for i, test := range table {
		//dbg("%d ----\n%q\n----\n", i, test.src)
		l := newLexer(test.src)
		ok := yyParse(l) == 0
		if g, e := ok, test.ok; g != e {
			if !ok {
				t.Log(l.errs[0])
			}
			t.Error(i, test.src, g, e)
			return
		}

		switch ok {
		case true:
			if len(l.errs) != 0 {
				t.Fatal(l.errs)
			}
		case false:
			if len(l.errs) == 0 {
				t.Fatal(l.errs)
			}
		}
	}
}
