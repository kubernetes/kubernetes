go-sqlite3
==========

[![GoDoc Reference](https://godoc.org/github.com/mattn/go-sqlite3?status.svg)](http://godoc.org/github.com/mattn/go-sqlite3)
[![Build Status](https://travis-ci.org/mattn/go-sqlite3.svg?branch=master)](https://travis-ci.org/mattn/go-sqlite3)
[![Coverage Status](https://coveralls.io/repos/mattn/go-sqlite3/badge.svg?branch=master)](https://coveralls.io/r/mattn/go-sqlite3?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/mattn/go-sqlite3)](https://goreportcard.com/report/github.com/mattn/go-sqlite3)

Description
-----------

sqlite3 driver conforming to the built-in database/sql interface

Installation
------------

This package can be installed with the go get command:

    go get github.com/mattn/go-sqlite3

_go-sqlite3_ is *cgo* package.
If you want to build your app using go-sqlite3, you need gcc.
However, if you install _go-sqlite3_ with `go install github.com/mattn/go-sqlite3`, you don't need gcc to build your app anymore.

Documentation
-------------

API documentation can be found here: http://godoc.org/github.com/mattn/go-sqlite3

Examples can be found under the `./_example` directory

FAQ
---

* Want to build go-sqlite3 with libsqlite3 on my linux.

    Use `go build --tags "libsqlite3 linux"`

* Want to build go-sqlite3 with libsqlite3 on OS X.

    Install sqlite3 from homebrew: `brew install sqlite3`

    Use `go build --tags "libsqlite3 darwin"`

* Want to build go-sqlite3 with icu extension.

   Use `go build --tags "icu"`

   Available extensions: `json1`, `fts5`, `icu`

* Can't build go-sqlite3 on windows 64bit.

    > Probably, you are using go 1.0, go1.0 has a problem when it comes to compiling/linking on windows 64bit.
    > See: [#27](https://github.com/mattn/go-sqlite3/issues/27)

* Getting insert error while query is opened.

    > You can pass some arguments into the connection string, for example, a URI.
    > See: [#39](https://github.com/mattn/go-sqlite3/issues/39)

* Do you want to cross compile? mingw on Linux or Mac?

    > See: [#106](https://github.com/mattn/go-sqlite3/issues/106)
    > See also: http://www.limitlessfx.com/cross-compile-golang-app-for-windows-from-linux.html

* Want to get time.Time with current locale

    Use `_loc=auto` in SQLite3 filename schema like `file:foo.db?_loc=auto`.

* Can I use this in multiple routines concurrently?

    Yes for readonly. But, No for writable. See [#50](https://github.com/mattn/go-sqlite3/issues/50), [#51](https://github.com/mattn/go-sqlite3/issues/51), [#209](https://github.com/mattn/go-sqlite3/issues/209).

* Why is it racy if I use a `sql.Open("sqlite3", ":memory:")` database?

    Each connection to :memory: opens a brand new in-memory sql database, so if
    the stdlib's sql engine happens to open another connection and you've only
    specified ":memory:", that connection will see a brand new database. A
    workaround is to use "file::memory:?mode=memory&cache=shared". Every
    connection to this string will point to the same in-memory database. See
    [#204](https://github.com/mattn/go-sqlite3/issues/204) for more info.

License
-------

MIT: http://mattn.mit-license.org/2012

sqlite3-binding.c, sqlite3-binding.h, sqlite3ext.h

The -binding suffix was added to avoid build failures under gccgo.

In this repository, those files are an amalgamation of code that was copied from SQLite3. The license of that code is the same as the license of SQLite3.

Author
------

Yasuhiro Matsumoto (a.k.a mattn)
