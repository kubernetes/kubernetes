package rule

import (
	"fmt"
	"go/ast"
	"go/types"

	"github.com/mgechev/revive/lint"
)

// UnexportedReturnRule lints given else constructs.
type UnexportedReturnRule struct{}

// Apply applies the rule to given file.
func (r *UnexportedReturnRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintUnexportedReturn{
		file:    file,
		fileAst: fileAst,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	file.Pkg.TypeCheck()
	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *UnexportedReturnRule) Name() string {
	return "unexported-return"
}

type lintUnexportedReturn struct {
	file      *lint.File
	fileAst   *ast.File
	onFailure func(lint.Failure)
}

func (w lintUnexportedReturn) Visit(n ast.Node) ast.Visitor {
	fn, ok := n.(*ast.FuncDecl)
	if !ok {
		return w
	}
	if fn.Type.Results == nil {
		return nil
	}
	if !fn.Name.IsExported() {
		return nil
	}
	thing := "func"
	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		thing = "method"
		if !ast.IsExported(receiverType(fn)) {
			// Don't report exported methods of unexported types,
			// such as private implementations of sort.Interface.
			return nil
		}
	}
	for _, ret := range fn.Type.Results.List {
		typ := w.file.Pkg.TypeOf(ret.Type)
		if exportedType(typ) {
			continue
		}
		w.onFailure(lint.Failure{
			Category:   "unexported-type-in-api",
			Node:       ret.Type,
			Confidence: 0.8,
			Failure: fmt.Sprintf("exported %s %s returns unexported type %s, which can be annoying to use",
				thing, fn.Name.Name, typ),
		})
		break // only flag one
	}
	return nil
}

// exportedType reports whether typ is an exported type.
// It is imprecise, and will err on the side of returning true,
// such as for composite types.
func exportedType(typ types.Type) bool {
	switch T := typ.(type) {
	case *types.Named:
		obj := T.Obj()
		switch {
		// Builtin types have no package.
		case obj.Pkg() == nil:
		case obj.Exported():
		default:
			_, ok := T.Underlying().(*types.Interface)
			return ok
		}
		return true
	case *types.Map:
		return exportedType(T.Key()) && exportedType(T.Elem())
	case interface {
		Elem() types.Type
	}: // array, slice, pointer, chan
		return exportedType(T.Elem())
	}
	// Be conservative about other types, such as struct, interface, etc.
	return true
}
