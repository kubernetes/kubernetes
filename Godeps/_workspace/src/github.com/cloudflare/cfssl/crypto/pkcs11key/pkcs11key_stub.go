// Package pkcs11key exists to satisfy Go build tools.
// Some Go tools will complain "no buildable Go source files in ..." because
// pkcs11key.go only builds when the pkcs11 tag is supplied. This empty file
// exists only to suppress that error, which blocks completion in some tools
// (specifically godep).
package pkcs11key
