// Package jsoniter implements encoding and decoding of JSON as defined in
// RFC 4627 and provides interfaces with identical syntax of standard lib encoding/json.
// Converting from encoding/json to jsoniter is no more than replacing the package with jsoniter
// and variable type declarations (if any).
// jsoniter interfaces gives 100% compatibility with code using standard lib.
//
// "JSON and Go"
// (https://golang.org/doc/articles/json_and_go.html)
// gives a description of how Marshal/Unmarshal operate
// between arbitrary or predefined json objects and bytes,
// and it applies to jsoniter.Marshal/Unmarshal as well.
//
// Besides, jsoniter.Iterator provides a different set of interfaces
// iterating given bytes/string/reader
// and yielding parsed elements one by one.
// This set of interfaces reads input as required and gives
// better performance.
package jsoniter
