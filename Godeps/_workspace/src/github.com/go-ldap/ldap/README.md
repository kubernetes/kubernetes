[![GoDoc](https://godoc.org/gopkg.in/ldap.v1?status.svg)](https://godoc.org/gopkg.in/ldap.v1) [![Build Status](https://travis-ci.org/go-ldap/ldap.svg)](https://travis-ci.org/go-ldap/ldap)

# Basic LDAP v3 functionality for the GO programming language.

## Required Librarys: 

 - gopkg.in/asn1-ber.v1

## Working:

 - Connecting to LDAP server
 - Binding to LDAP server
 - Searching for entries
 - Compiling string filters to LDAP filters
 - Paging Search Results
 - Modify Requests / Responses

## Examples:

 - search
 - modify

## Tests Implemented:

 - Filter Compile / Decompile

## TODO:

 - Add Requests / Responses
 - Delete Requests / Responses
 - Modify DN Requests / Responses
 - Compare Requests / Responses
 - Implement Tests / Benchmarks

---
This feature is disabled at the moment, because in some cases the "Search Request Done" packet will be handled before the last "Search Request Entry":

 - Mulitple internal goroutines to handle network traffic
        Makes library goroutine safe
        Can perform multiple search requests at the same time and return
        the results to the proper goroutine. All requests are blocking requests,
        so the goroutine does not need special handling

---

The Go gopher was designed by Renee French. (http://reneefrench.blogspot.com/)
The design is licensed under the Creative Commons 3.0 Attributions license.
Read this article for more details: http://blog.golang.org/gopher
