package astwalk

import (
	"go/ast"
)

// LocalDefVisitor visits every name definitions inside a function.
//
// Next elements are considered as name definitions:
//	- Function parameters (input, output, receiver)
//	- Every LHS of ":=" assignment that defines a new name
//	- Every local var/const declaration.
//
// NOTE: this visitor is experimental.
// This is also why it lives in a separate file.
type LocalDefVisitor interface {
	walkerEvents
	VisitLocalDef(Name, ast.Expr)
}

type (
	// NameKind describes what kind of name Name object holds.
	NameKind int

	// Name holds ver/const/param definition symbol info.
	Name struct {
		ID   *ast.Ident
		Kind NameKind

		// Index is NameVar-specific field that is used to
		// specify nth tuple element being assigned to the name.
		Index int
	}
)

// NOTE: set of name kinds is not stable and may change over time.
//
// TODO(quasilyte): is NameRecv/NameParam/NameResult granularity desired?
// TODO(quasilyte): is NameVar/NameBind (var vs :=) granularity desired?
const (
	// NameParam is function/method receiver/input/output name.
	// Initializing expression is always nil.
	NameParam NameKind = iota
	// NameVar is var or ":=" declared name.
	// Initializing expression may be nil for var-declared names
	// without explicit initializing expression.
	NameVar
	// NameConst is const-declared name.
	// Initializing expression is never nil.
	NameConst
)
