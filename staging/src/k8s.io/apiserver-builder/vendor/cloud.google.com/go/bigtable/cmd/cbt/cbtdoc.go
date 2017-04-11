// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// DO NOT EDIT. THIS IS AUTOMATICALLY GENERATED.
// Run "go generate" to regenerate.
//go:generate go run cbt.go -o cbtdoc.go doc

/*
Cbt is a tool for doing basic interactions with Cloud Bigtable.

Usage:

	cbt [options] command [arguments]

The commands are:

	count                     Count rows in a table
	createfamily              Create a column family
	createtable               Create a table
	deletefamily              Delete a column family
	deleterow                 Delete a row
	deletetable               Delete a table
	doc                       Print godoc-suitable documentation for cbt
	help                      Print help text
	listinstances             List instances in a project
	lookup                    Read from a single row
	ls                        List tables and column families
	mddoc                     Print documentation for cbt in Markdown format
	read                      Read rows
	set                       Set value of a cell
	setgcpolicy               Set the GC policy for a column family

Use "cbt help <command>" for more information about a command.

The options are:

	-project string
		project ID
	-instance string
		Cloud Bigtable instance
	-creds string
		if set, use application credentials in this file


Count rows in a table

Usage:
	cbt count <table>




Create a column family

Usage:
	cbt createfamily <table> <family>




Create a table

Usage:
	cbt createtable <table>




Delete a column family

Usage:
	cbt deletefamily <table> <family>




Delete a row

Usage:
	cbt deleterow <table> <row>




Delete a table

Usage:
	cbt deletetable <table>




Print godoc-suitable documentation for cbt

Usage:
	cbt doc




Print help text

Usage:
	cbt help [command]




List instances in a project

Usage:
	cbt listinstances




Read from a single row

Usage:
	cbt lookup <table> <row>




List tables and column families

Usage:
	cbt ls			List tables
	cbt ls <table>		List column families in <table>




Print documentation for cbt in Markdown format

Usage:
	cbt mddoc




Read rows

Usage:
	cbt read <table> [start=<row>] [end=<row>] [prefix=<prefix>] [count=<n>]
	  start=<row>		Start reading at this row
	  end=<row>		Stop reading before this row
	  prefix=<prefix>	Read rows with this prefix
	  count=<n>		Read only this many rows





Set value of a cell

Usage:
	cbt set <table> <row> family:column=val[@ts] ...
	  family:column=val[@ts] may be repeated to set multiple cells.

	  ts is an optional integer timestamp.
	  If it cannot be parsed, the `@ts` part will be
	  interpreted as part of the value.




Set the GC policy for a column family

Usage:
	cbt setgcpolicy <table> <family> ( maxage=<d> | maxversions=<n> )

	  maxage=<d>		Maximum timestamp age to preserve (e.g. "1h", "4d")
	  maxversions=<n>	Maximum number of versions to preserve




*/
package main
