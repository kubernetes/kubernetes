package goconst

import (
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

// treeVisitor carries the package name and file name
// for passing it to the imports map, and the fileSet for
// retrieving the token.Position.
type treeVisitor struct {
	p                     *Parser
	fileSet               *token.FileSet
	packageName, fileName string
}

// Visit browses the AST tree for strings that could be potentially
// replaced by constants.
// A map of existing constants is built as well (-match-constant).
func (v *treeVisitor) Visit(node ast.Node) ast.Visitor {
	if node == nil {
		return v
	}

	// A single case with "ast.BasicLit" would be much easier
	// but then we wouldn't be able to tell in which context
	// the string is defined (could be a constant definition).
	switch t := node.(type) {
	// Scan for constants in an attempt to match strings with existing constants
	case *ast.GenDecl:
		if !v.p.matchConstant {
			return v
		}
		if t.Tok != token.CONST {
			return v
		}

		for _, spec := range t.Specs {
			val := spec.(*ast.ValueSpec)
			for i, str := range val.Values {
				lit, ok := str.(*ast.BasicLit)
				if !ok || !v.isSupported(lit.Kind) {
					continue
				}

				v.addConst(val.Names[i].Name, lit.Value, val.Names[i].Pos())
			}
		}

	// foo := "moo"
	case *ast.AssignStmt:
		for _, rhs := range t.Rhs {
			lit, ok := rhs.(*ast.BasicLit)
			if !ok || !v.isSupported(lit.Kind) {
				continue
			}

			v.addString(lit.Value, rhs.(*ast.BasicLit).Pos(), Assignment)
		}

	// if foo == "moo"
	case *ast.BinaryExpr:
		if t.Op != token.EQL && t.Op != token.NEQ {
			return v
		}

		var lit *ast.BasicLit
		var ok bool

		lit, ok = t.X.(*ast.BasicLit)
		if ok && v.isSupported(lit.Kind) {
			v.addString(lit.Value, lit.Pos(), Binary)
		}

		lit, ok = t.Y.(*ast.BasicLit)
		if ok && v.isSupported(lit.Kind) {
			v.addString(lit.Value, lit.Pos(), Binary)
		}

	// case "foo":
	case *ast.CaseClause:
		for _, item := range t.List {
			lit, ok := item.(*ast.BasicLit)
			if ok && v.isSupported(lit.Kind) {
				v.addString(lit.Value, lit.Pos(), Case)
			}
		}

	// return "boo"
	case *ast.ReturnStmt:
		for _, item := range t.Results {
			lit, ok := item.(*ast.BasicLit)
			if ok && v.isSupported(lit.Kind) {
				v.addString(lit.Value, lit.Pos(), Return)
			}
		}

	// fn("http://")
	case *ast.CallExpr:
		for _, item := range t.Args {
			lit, ok := item.(*ast.BasicLit)
			if ok && v.isSupported(lit.Kind) {
				v.addString(lit.Value, lit.Pos(), Call)
			}
		}
	}

	return v
}

// addString adds a string in the map along with its position in the tree.
func (v *treeVisitor) addString(str string, pos token.Pos, typ Type) {
	ok, excluded := v.p.excludeTypes[typ]
	if ok && excluded {
		return
	}
	// Drop quotes if any
	if strings.HasPrefix(str, `"`) || strings.HasPrefix(str, "`") {
		str, _ = strconv.Unquote(str)
	}

	// Ignore empty strings
	if len(str) == 0 {
		return
	}

	if len(str) < v.p.minLength {
		return
	}

	_, ok = v.p.strs[str]
	if !ok {
		v.p.strs[str] = make([]ExtendedPos, 0)
	}
	v.p.strs[str] = append(v.p.strs[str], ExtendedPos{
		packageName: v.packageName,
		Position:    v.fileSet.Position(pos),
	})
}

// addConst adds a const in the map along with its position in the tree.
func (v *treeVisitor) addConst(name string, val string, pos token.Pos) {
	val = strings.Replace(val, `"`, "", 2)
	v.p.consts[val] = ConstType{
		Name:        name,
		packageName: v.packageName,
		Position:    v.fileSet.Position(pos),
	}
}

func (v *treeVisitor) isSupported(tk token.Token) bool {
	for _, s := range v.p.supportedTokens {
		if tk == s {
			return true
		}
	}
	return false
}
