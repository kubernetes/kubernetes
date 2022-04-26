package analyzer

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const (
	name = "nilnil"
	doc  = "Checks that there is no simultaneous return of `nil` error and an invalid value."

	reportMsg = "return both the `nil` error and invalid value: use a sentinel error instead"
)

// New returns new nilnil analyzer.
func New() *analysis.Analyzer {
	n := newNilNil()

	a := &analysis.Analyzer{
		Name:     name,
		Doc:      doc,
		Run:      n.run,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	}
	a.Flags.Var(&n.checkedTypes, "checked-types", "coma separated list")

	return a
}

type nilNil struct {
	checkedTypes checkedTypes
}

func newNilNil() *nilNil {
	return &nilNil{
		checkedTypes: newDefaultCheckedTypes(),
	}
}

var (
	types = []ast.Node{(*ast.TypeSpec)(nil)}

	funcAndReturns = []ast.Node{
		(*ast.FuncDecl)(nil),
		(*ast.FuncLit)(nil),
		(*ast.ReturnStmt)(nil),
	}
)

type typeSpecByName map[string]*ast.TypeSpec

func (n *nilNil) run(pass *analysis.Pass) (interface{}, error) {
	insp := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	typeSpecs := typeSpecByName{}
	insp.Preorder(types, func(node ast.Node) {
		t := node.(*ast.TypeSpec)
		typeSpecs[t.Name.Name] = t
	})

	var fs funcTypeStack
	insp.Nodes(funcAndReturns, func(node ast.Node, push bool) (proceed bool) {
		switch v := node.(type) {
		case *ast.FuncLit:
			if push {
				fs.Push(v.Type)
			} else {
				fs.Pop()
			}

		case *ast.FuncDecl:
			if push {
				fs.Push(v.Type)
			} else {
				fs.Pop()
			}

		case *ast.ReturnStmt:
			ft := fs.Top() // Current function.

			if !push || len(v.Results) != 2 || ft == nil || ft.Results == nil || len(ft.Results.List) != 2 {
				return false
			}

			fRes1, fRes2 := ft.Results.List[0], ft.Results.List[1]
			if !(n.isDangerNilField(fRes1, typeSpecs) && n.isErrorField(fRes2)) {
				return
			}

			rRes1, rRes2 := v.Results[0], v.Results[1]
			if isNil(rRes1) && isNil(rRes2) {
				pass.Reportf(v.Pos(), reportMsg)
			}
		}

		return true
	})

	return nil, nil
}

func (n *nilNil) isDangerNilField(f *ast.Field, typeSpecs typeSpecByName) bool {
	return n.isDangerNilType(f.Type, typeSpecs)
}

func (n *nilNil) isDangerNilType(t ast.Expr, typeSpecs typeSpecByName) bool {
	switch v := t.(type) {
	case *ast.StarExpr:
		return n.checkedTypes.Contains(ptrType)

	case *ast.FuncType:
		return n.checkedTypes.Contains(funcType)

	case *ast.InterfaceType:
		return n.checkedTypes.Contains(ifaceType)

	case *ast.MapType:
		return n.checkedTypes.Contains(mapType)

	case *ast.ChanType:
		return n.checkedTypes.Contains(chanType)

	case *ast.Ident:
		if t, ok := typeSpecs[v.Name]; ok {
			return n.isDangerNilType(t.Type, nil)
		}
	}
	return false
}

func (n *nilNil) isErrorField(f *ast.Field) bool {
	return isIdent(f.Type, "error")
}

func isNil(e ast.Expr) bool {
	return isIdent(e, "nil")
}

func isIdent(n ast.Node, name string) bool {
	i, ok := n.(*ast.Ident)
	if !ok {
		return false
	}
	return i.Name == name
}
