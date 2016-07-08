// Package hcl decodes HCL into usable Go structures.
//
// hcl input can come in either pure HCL format or JSON format.
// It can be parsed into an AST, and then decoded into a structure,
// or it can be decoded directly from a string into a structure.
//
// If you choose to parse HCL into a raw AST, the benefit is that you
// can write custom visitor implementations to implement custom
// semantic checks. By default, HCL does not perform any semantic
// checks.
package hcl
