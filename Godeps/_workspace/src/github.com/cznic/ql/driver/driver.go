// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package driver registers a QL sql/driver named "ql" and a memory driver named "ql-mem".

See also [0], [1] and [3].

Usage

A skeleton program using ql/driver.

	package main

	import (
		"database/sql"

		_ "github.com/cznic/ql/driver"
	)

	func main() {
		...
		// Disk file DB
		db, err := sql.Open("ql", "ql.db")  // [2]
		// alternatively
		db, err := sql.Open("ql", "file://ql.db")

		// and/or

		// RAM DB
		mdb, err := sql.Open("ql-mem", "mem.db")
		// alternatively
		mdb, err := sql.Open("ql", "memory://mem.db")
		if err != nil {
			log.Fatal(err)
		}

		// Use db/mdb here
		...
	}

This package exports nothing.

Links

Referenced from above:

  [0]: http://godoc.org/github.com/cznic/ql
  [1]: http://golang.org/pkg/database/sql/
  [2]: http://golang.org/pkg/database/sql/#Open
  [3]: http://golang.org/pkg/database/sql/driver
*/
package driver

import "github.com/cznic/ql"

func init() {
	ql.RegisterDriver()
	ql.RegisterMemDriver()
}
