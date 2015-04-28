// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package driver

import (
	"database/sql"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

func Example_testFile() {
	dir, err := ioutil.TempDir("", "ql-driver-test")
	if err != nil {
		return
	}

	defer func() {
		os.RemoveAll(dir)
	}()

	db, err := sql.Open("ql", filepath.Join(dir, "ql.db"))
	if err != nil {
		return
	}

	defer func() {
		if err := db.Close(); err != nil {
			return
		}

		fmt.Println("OK")
	}()

	tx, err := db.Begin()
	if err != nil {
		return
	}

	if _, err := tx.Exec("CREATE TABLE t (Qty int, Name string);"); err != nil {
		return
	}

	result, err := tx.Exec(`
	INSERT INTO t VALUES
		($1, $2),
		($3, $4),
	;
	`,
		42, "foo",
		314, "bar",
	)
	if err != nil {
		return
	}

	if err = tx.Commit(); err != nil {
		return
	}

	id, err := result.LastInsertId()
	if err != nil {
		return
	}

	aff, err := result.RowsAffected()
	if err != nil {
		return
	}

	fmt.Printf("LastInsertId %d, RowsAffected %d\n", id, aff)

	rows, err := db.Query("SELECT * FROM t;")
	if err != nil {
		return
	}

	cols, err := rows.Columns()
	if err != nil {
		return
	}

	fmt.Printf("Columns: %v\n", cols)

	var data struct {
		Qty  int
		Name string
	}

	for rows.Next() {
		if err = rows.Scan(&data.Qty, &data.Name); err != nil {
			rows.Close()
			break
		}

		fmt.Printf("%+v\n", data)
	}

	if err = rows.Err(); err != nil {
		return
	}

	// Output:
	// LastInsertId 2, RowsAffected 2
	// Columns: [Qty Name]
	// {Qty:314 Name:bar}
	// {Qty:42 Name:foo}
	// OK
}

func Example_testMem() {
	db, err := sql.Open("ql-mem", "mem.db")
	if err != nil {
		return
	}

	defer func() {
		if err := db.Close(); err != nil {
			return
		}

		fmt.Println("OK")
	}()

	tx, err := db.Begin()
	if err != nil {
		return
	}

	if _, err := tx.Exec("CREATE TABLE t (Qty int, Name string);"); err != nil {
		return
	}

	result, err := tx.Exec(`
	INSERT INTO t VALUES
		($1, $2),
		($3, $4),
	;
	`,
		1042, "foo-mem",
		1314, "bar-mem",
	)
	if err != nil {
		return
	}

	if err = tx.Commit(); err != nil {
		return
	}

	id, err := result.LastInsertId()
	if err != nil {
		return
	}

	aff, err := result.RowsAffected()
	if err != nil {
		return
	}

	fmt.Printf("LastInsertId %d, RowsAffected %d\n", id, aff)

	rows, err := db.Query("SELECT * FROM t;")
	if err != nil {
		return
	}

	cols, err := rows.Columns()
	if err != nil {
		return
	}

	fmt.Printf("Columns: %v\n", cols)

	var data struct {
		Qty  int
		Name string
	}

	for rows.Next() {
		if err = rows.Scan(&data.Qty, &data.Name); err != nil {
			rows.Close()
			break
		}

		fmt.Printf("%+v\n", data)
	}

	if err = rows.Err(); err != nil {
		return
	}

	// Output:
	// LastInsertId 2, RowsAffected 2
	// Columns: [Qty Name]
	// {Qty:1314 Name:bar-mem}
	// {Qty:1042 Name:foo-mem}
	// OK
}
