package edit

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/token"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/pattern"
)

type Ranger interface {
	Pos() token.Pos
	End() token.Pos
}

type Range [2]token.Pos

func (r Range) Pos() token.Pos { return r[0] }
func (r Range) End() token.Pos { return r[1] }

func ReplaceWithString(fset *token.FileSet, old Ranger, new string) analysis.TextEdit {
	return analysis.TextEdit{
		Pos:     old.Pos(),
		End:     old.End(),
		NewText: []byte(new),
	}
}

func ReplaceWithNode(fset *token.FileSet, old Ranger, new ast.Node) analysis.TextEdit {
	buf := &bytes.Buffer{}
	if err := format.Node(buf, fset, new); err != nil {
		panic("internal error: " + err.Error())
	}
	return analysis.TextEdit{
		Pos:     old.Pos(),
		End:     old.End(),
		NewText: buf.Bytes(),
	}
}

func ReplaceWithPattern(pass *analysis.Pass, after pattern.Pattern, state pattern.State, node Ranger) analysis.TextEdit {
	r := pattern.NodeToAST(after.Root, state)
	buf := &bytes.Buffer{}
	format.Node(buf, pass.Fset, r)
	return analysis.TextEdit{
		Pos:     node.Pos(),
		End:     node.End(),
		NewText: buf.Bytes(),
	}
}

func Delete(old Ranger) analysis.TextEdit {
	return analysis.TextEdit{
		Pos:     old.Pos(),
		End:     old.End(),
		NewText: nil,
	}
}

func Fix(msg string, edits ...analysis.TextEdit) analysis.SuggestedFix {
	return analysis.SuggestedFix{
		Message:   msg,
		TextEdits: edits,
	}
}

func Selector(x, sel string) *ast.SelectorExpr {
	return &ast.SelectorExpr{
		X:   &ast.Ident{Name: x},
		Sel: &ast.Ident{Name: sel},
	}
}
