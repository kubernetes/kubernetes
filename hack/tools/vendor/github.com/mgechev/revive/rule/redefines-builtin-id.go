package rule

import (
	"fmt"
	"github.com/mgechev/revive/lint"
	"go/ast"
	"go/token"
)

// RedefinesBuiltinIDRule warns when a builtin identifier is shadowed.
type RedefinesBuiltinIDRule struct{}

// Apply applies the rule to given file.
func (r *RedefinesBuiltinIDRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	var builtInConstAndVars = map[string]bool{
		"true":  true,
		"false": true,
		"iota":  true,
		"nil":   true,
	}

	var builtFunctions = map[string]bool{
		"append":  true,
		"cap":     true,
		"close":   true,
		"complex": true,
		"copy":    true,
		"delete":  true,
		"imag":    true,
		"len":     true,
		"make":    true,
		"new":     true,
		"panic":   true,
		"print":   true,
		"println": true,
		"real":    true,
		"recover": true,
	}

	var builtInTypes = map[string]bool{
		"ComplexType": true,
		"FloatType":   true,
		"IntegerType": true,
		"Type":        true,
		"Type1":       true,
		"bool":        true,
		"byte":        true,
		"complex128":  true,
		"complex64":   true,
		"error":       true,
		"float32":     true,
		"float64":     true,
		"int":         true,
		"int16":       true,
		"int32":       true,
		"int64":       true,
		"int8":        true,
		"rune":        true,
		"string":      true,
		"uint":        true,
		"uint16":      true,
		"uint32":      true,
		"uint64":      true,
		"uint8":       true,
		"uintptr":     true,
	}

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	astFile := file.AST
	w := &lintRedefinesBuiltinID{builtInConstAndVars, builtFunctions, builtInTypes, onFailure}
	ast.Walk(w, astFile)

	return failures
}

// Name returns the rule name.
func (r *RedefinesBuiltinIDRule) Name() string {
	return "redefines-builtin-id"
}

type lintRedefinesBuiltinID struct {
	constsAndVars map[string]bool
	funcs         map[string]bool
	types         map[string]bool
	onFailure     func(lint.Failure)
}

func (w *lintRedefinesBuiltinID) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.GenDecl:
		if n.Tok != token.TYPE {
			return nil // skip if not type declaration
		}
		typeSpec, ok := n.Specs[0].(*ast.TypeSpec)
		if !ok {
			return nil
		}
		id := typeSpec.Name.Name
		if w.types[id] {
			w.addFailure(n, fmt.Sprintf("redefinition of the built-in type %s", id))
		}
	case *ast.FuncDecl:
		if n.Recv != nil {
			return w // skip methods
		}

		id := n.Name.Name
		if w.funcs[id] {
			w.addFailure(n, fmt.Sprintf("redefinition of the built-in function %s", id))
		}
	case *ast.AssignStmt:
		for _, e := range n.Lhs {
			id, ok := e.(*ast.Ident)
			if !ok {
				continue
			}

			if w.constsAndVars[id.Name] {
				var msg string
				if n.Tok == token.DEFINE {
					msg = fmt.Sprintf("assignment creates a shadow of built-in identifier %s", id.Name)
				} else {
					msg = fmt.Sprintf("assignment modifies built-in identifier %s", id.Name)
				}
				w.addFailure(n, msg)
			}
		}
	}

	return w
}

func (w lintRedefinesBuiltinID) addFailure(node ast.Node, msg string) {
	w.onFailure(lint.Failure{
		Confidence: 1,
		Node:       node,
		Category:   "logic",
		Failure:    msg,
	})
}
