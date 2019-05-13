package internal

// MainPath stores the file path of the main package. On App Engine Standard
// using Go version 1.9 and below, this will be unset. On App Engine Flex and
// App Engine Standard second-gen (Go 1.11 and above), this will be the
// filepath to package main.
var MainPath string
