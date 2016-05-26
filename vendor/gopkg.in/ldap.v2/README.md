[![GoDoc](https://godoc.org/gopkg.in/ldap.v2?status.svg)](https://godoc.org/gopkg.in/ldap.v2)
[![Build Status](https://travis-ci.org/go-ldap/ldap.svg)](https://travis-ci.org/go-ldap/ldap)

# Basic LDAP v3 functionality for the GO programming language.

## Install

For the latest version use:

    go get gopkg.in/ldap.v2

Import the latest version with:

    import "gopkg.in/ldap.v2"


## Required Libraries:

 - gopkg.in/asn1-ber.v1

## Working:

 - Connecting to LDAP server
 - Binding to LDAP server
 - Searching for entries
 - Compiling string filters to LDAP filters
 - Paging Search Results
 - Modify Requests / Responses
 - Add Requests / Responses
 - Delete Requests / Responses
 - Better Unicode support

## Examples:

 - search
 - modify

## Tests Implemented:

 - Filter Compile / Decompile

## TODO:

 - [x] Add Requests / Responses
 - [x] Delete Requests / Responses
 - [x] Modify DN Requests / Responses
 - [ ] Compare Requests / Responses
 - [ ] Implement Tests / Benchmarks



---
The Go gopher was designed by Renee French. (http://reneefrench.blogspot.com/)
The design is licensed under the Creative Commons 3.0 Attributions license.
Read this article for more details: http://blog.golang.org/gopher
