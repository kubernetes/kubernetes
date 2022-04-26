// Package astfmt implements `ast.Node` formatting with fmt-like API.
package astfmt

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"io"
)

// Println calls fmt.Println with additional support of %s format
// for ast.Node arguments.
//
// Uses empty file set for AST printing.
func Println(args ...interface{}) error {
	return defaultPrinter.Println(args...)
}

// Fprintf calls fmt.Fprintf with additional support of %s format
// for ast.Node arguments.
//
// Uses empty file set for AST printing.
func Fprintf(w io.Writer, format string, args ...interface{}) error {
	return defaultPrinter.Fprintf(w, format, args...)
}

// Sprintf calls fmt.Sprintf with additional support of %s format
// for ast.Node arguments.
//
// Uses empty file set for AST printing.
func Sprintf(format string, args ...interface{}) string {
	return defaultPrinter.Sprintf(format, args...)
}

// Sprint calls fmt.Sprint with additional support of %s format
// for ast.Node arguments.
//
// Uses empty file set for AST printing.
func Sprint(args ...interface{}) string {
	return defaultPrinter.Sprint(args...)
}

// NewPrinter returns printer that uses bound file set when printing AST nodes.
func NewPrinter(fset *token.FileSet) *Printer {
	return &Printer{fset: fset}
}

// Printer provides API close to fmt package for printing AST nodes.
// Unlike freestanding functions from this package, it makes it possible
// to associate appropriate file set for better output.
type Printer struct {
	fset *token.FileSet
}

// Println printer method is like Println function, but uses bound file set when printing.
func (p *Printer) Println(args ...interface{}) error {
	_, err := fmt.Println(wrapArgs(p.fset, args)...)
	return err
}

// Fprintf printer method is like Fprintf function, but uses bound file set when printing.
func (p *Printer) Fprintf(w io.Writer, format string, args ...interface{}) error {
	_, err := fmt.Fprintf(w, format, wrapArgs(p.fset, args)...)
	return err
}

// Sprintf printer method is like Sprintf function, but uses bound file set when printing.
func (p *Printer) Sprintf(format string, args ...interface{}) string {
	return fmt.Sprintf(format, wrapArgs(p.fset, args)...)
}

// Sprint printer method is like Sprint function, but uses bound file set when printing.
func (p *Printer) Sprint(args ...interface{}) string {
	return fmt.Sprint(wrapArgs(p.fset, args)...)
}

// defaultPrinter is used in printing functions like Println.
// Uses empty file set.
var defaultPrinter = NewPrinter(token.NewFileSet())

// wrapArgs returns arguments slice with every ast.Node element
// replaced with fmtNode wrapper that supports additional formatting.
func wrapArgs(fset *token.FileSet, args []interface{}) []interface{} {
	for i := range args {
		if x, ok := args[i].(ast.Node); ok {
			args[i] = fmtNode{fset: fset, node: x}
		}
	}
	return args
}

type fmtNode struct {
	fset *token.FileSet
	node ast.Node
}

func (n fmtNode) String() string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, n.fset, n.node); err != nil {
		return fmt.Sprintf("%%!s(ast.Node=%s)", err)
	}
	return buf.String()
}

func (n fmtNode) GoString() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%#v", n.node)
	return buf.String()
}
