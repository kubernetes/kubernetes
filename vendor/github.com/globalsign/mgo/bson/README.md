[![GoDoc](https://godoc.org/github.com/globalsign/mgo/bson?status.svg)](https://godoc.org/github.com/globalsign/mgo/bson)

An Implementation of BSON for Go
--------------------------------

Package bson is an implementation of the [BSON specification](http://bsonspec.org) for Go.

While the BSON package implements the BSON spec as faithfully as possible, there
is some MongoDB specific behaviour (such as map keys `$in`, `$all`, etc) in the
`bson` package. The priority is for backwards compatibility for the `mgo`
driver, though fixes for obviously buggy behaviour is welcome (and features, etc
behind feature flags).
