// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package cloudsql exposes access to Google Cloud SQL databases.

This package does not work in App Engine "flexible environment".

This package is intended for MySQL drivers to make App Engine-specific
connections. Applications should use this package through database/sql:
Select a pure Go MySQL driver that supports this package, and use sql.Open
with protocol "cloudsql" and an address of the Cloud SQL instance.

A Go MySQL driver that has been tested to work well with Cloud SQL
is the go-sql-driver:
	import "database/sql"
	import _ "github.com/go-sql-driver/mysql"

	db, err := sql.Open("mysql", "user@cloudsql(project-id:instance-name)/dbname")


Another driver that works well with Cloud SQL is the mymysql driver:
	import "database/sql"
	import _ "github.com/ziutek/mymysql/godrv"

	db, err := sql.Open("mymysql", "cloudsql:instance-name*dbname/user/password")


Using either of these drivers, you can perform a standard SQL query.
This example assumes there is a table named 'users' with
columns 'first_name' and 'last_name':

	rows, err := db.Query("SELECT first_name, last_name FROM users")
	if err != nil {
		log.Errorf(ctx, "db.Query: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var firstName string
		var lastName string
		if err := rows.Scan(&firstName, &lastName); err != nil {
			log.Errorf(ctx, "rows.Scan: %v", err)
			continue
		}
		log.Infof(ctx, "First: %v - Last: %v", firstName, lastName)
	}
	if err := rows.Err(); err != nil {
		log.Errorf(ctx, "Row error: %v", err)
	}
*/
package cloudsql

import (
	"net"
)

// Dial connects to the named Cloud SQL instance.
func Dial(instance string) (net.Conn, error) {
	return connect(instance)
}
