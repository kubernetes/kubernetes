package ruleguard

import (
	"go/ast"
	"go/constant"

	"github.com/quasilyte/gogrep/nodetag"
)

type astWalker struct {
	nodePath *nodePath

	filterParams *filterParams

	visit func(ast.Node, nodetag.Value)
}

func (w *astWalker) Walk(root ast.Node, visit func(ast.Node, nodetag.Value)) {
	w.visit = visit
	w.walk(root)
}

func (w *astWalker) walkIdentList(list []*ast.Ident) {
	for _, x := range list {
		w.walk(x)
	}
}

func (w *astWalker) walkExprList(list []ast.Expr) {
	for _, x := range list {
		w.walk(x)
	}
}

func (w *astWalker) walkStmtList(list []ast.Stmt) {
	for _, x := range list {
		w.walk(x)
	}
}

func (w *astWalker) walkDeclList(list []ast.Decl) {
	for _, x := range list {
		w.walk(x)
	}
}

func (w *astWalker) walk(n ast.Node) {
	w.nodePath.Push(n)
	defer w.nodePath.Pop()

	switch n := n.(type) {
	case *ast.Field:
		// TODO: handle field types.
		// See #252
		w.walkIdentList(n.Names)
		w.walk(n.Type)

	case *ast.FieldList:
		for _, f := range n.List {
			w.walk(f)
		}

	case *ast.Ellipsis:
		w.visit(n, nodetag.Ellipsis)
		if n.Elt != nil {
			w.walk(n.Elt)
		}

	case *ast.FuncLit:
		w.visit(n, nodetag.FuncLit)
		w.walk(n.Type)
		w.walk(n.Body)

	case *ast.CompositeLit:
		w.visit(n, nodetag.CompositeLit)
		if n.Type != nil {
			w.walk(n.Type)
		}
		w.walkExprList(n.Elts)

	case *ast.ParenExpr:
		w.visit(n, nodetag.ParenExpr)
		w.walk(n.X)

	case *ast.SelectorExpr:
		w.visit(n, nodetag.SelectorExpr)
		w.walk(n.X)
		w.walk(n.Sel)

	case *ast.IndexExpr:
		w.visit(n, nodetag.IndexExpr)
		w.walk(n.X)
		w.walk(n.Index)

	case *ast.SliceExpr:
		w.visit(n, nodetag.SliceExpr)
		w.walk(n.X)
		if n.Low != nil {
			w.walk(n.Low)
		}
		if n.High != nil {
			w.walk(n.High)
		}
		if n.Max != nil {
			w.walk(n.Max)
		}

	case *ast.TypeAssertExpr:
		w.visit(n, nodetag.TypeAssertExpr)
		w.walk(n.X)
		if n.Type != nil {
			w.walk(n.Type)
		}

	case *ast.CallExpr:
		w.visit(n, nodetag.CallExpr)
		w.walk(n.Fun)
		w.walkExprList(n.Args)

	case *ast.StarExpr:
		w.visit(n, nodetag.StarExpr)
		w.walk(n.X)

	case *ast.UnaryExpr:
		w.visit(n, nodetag.UnaryExpr)
		w.walk(n.X)

	case *ast.BinaryExpr:
		w.visit(n, nodetag.BinaryExpr)
		w.walk(n.X)
		w.walk(n.Y)

	case *ast.KeyValueExpr:
		w.visit(n, nodetag.KeyValueExpr)
		w.walk(n.Key)
		w.walk(n.Value)

	case *ast.ArrayType:
		w.visit(n, nodetag.ArrayType)
		if n.Len != nil {
			w.walk(n.Len)
		}
		w.walk(n.Elt)

	case *ast.StructType:
		w.visit(n, nodetag.StructType)
		w.walk(n.Fields)

	case *ast.FuncType:
		w.visit(n, nodetag.FuncType)
		if n.Params != nil {
			w.walk(n.Params)
		}
		if n.Results != nil {
			w.walk(n.Results)
		}

	case *ast.InterfaceType:
		w.visit(n, nodetag.InterfaceType)
		w.walk(n.Methods)

	case *ast.MapType:
		w.visit(n, nodetag.MapType)
		w.walk(n.Key)
		w.walk(n.Value)

	case *ast.ChanType:
		w.visit(n, nodetag.ChanType)
		w.walk(n.Value)

	case *ast.DeclStmt:
		w.visit(n, nodetag.DeclStmt)
		w.walk(n.Decl)

	case *ast.LabeledStmt:
		w.visit(n, nodetag.LabeledStmt)
		w.walk(n.Label)
		w.walk(n.Stmt)

	case *ast.ExprStmt:
		w.visit(n, nodetag.ExprStmt)
		w.walk(n.X)

	case *ast.SendStmt:
		w.visit(n, nodetag.SendStmt)
		w.walk(n.Chan)
		w.walk(n.Value)

	case *ast.IncDecStmt:
		w.visit(n, nodetag.IncDecStmt)
		w.walk(n.X)

	case *ast.AssignStmt:
		w.visit(n, nodetag.AssignStmt)
		w.walkExprList(n.Lhs)
		w.walkExprList(n.Rhs)

	case *ast.GoStmt:
		w.visit(n, nodetag.GoStmt)
		w.walk(n.Call)

	case *ast.DeferStmt:
		w.visit(n, nodetag.DeferStmt)
		w.walk(n.Call)

	case *ast.ReturnStmt:
		w.visit(n, nodetag.ReturnStmt)
		w.walkExprList(n.Results)

	case *ast.BranchStmt:
		w.visit(n, nodetag.BranchStmt)
		if n.Label != nil {
			w.walk(n.Label)
		}

	case *ast.BlockStmt:
		w.visit(n, nodetag.BlockStmt)
		w.walkStmtList(n.List)

	case *ast.IfStmt:
		w.visit(n, nodetag.IfStmt)
		if n.Init != nil {
			w.walk(n.Init)
		}
		w.walk(n.Cond)
		deadcode := w.filterParams.deadcode
		if !deadcode {
			cv := w.filterParams.ctx.Types.Types[n.Cond].Value
			if cv != nil {
				w.filterParams.deadcode = !deadcode && !constant.BoolVal(cv)
				w.walk(n.Body)
				w.filterParams.deadcode = !w.filterParams.deadcode
				if n.Else != nil {
					w.walk(n.Else)
				}
				w.filterParams.deadcode = deadcode
				return
			}
		}
		w.walk(n.Body)
		if n.Else != nil {
			w.walk(n.Else)
		}

	case *ast.CaseClause:
		w.visit(n, nodetag.CaseClause)
		w.walkExprList(n.List)
		w.walkStmtList(n.Body)

	case *ast.SwitchStmt:
		w.visit(n, nodetag.SwitchStmt)
		if n.Init != nil {
			w.walk(n.Init)
		}
		if n.Tag != nil {
			w.walk(n.Tag)
		}
		w.walk(n.Body)

	case *ast.TypeSwitchStmt:
		w.visit(n, nodetag.TypeSwitchStmt)
		if n.Init != nil {
			w.walk(n.Init)
		}
		w.walk(n.Assign)
		w.walk(n.Body)

	case *ast.CommClause:
		w.visit(n, nodetag.CommClause)
		if n.Comm != nil {
			w.walk(n.Comm)
		}
		w.walkStmtList(n.Body)

	case *ast.SelectStmt:
		w.visit(n, nodetag.SelectStmt)
		w.walk(n.Body)

	case *ast.ForStmt:
		w.visit(n, nodetag.ForStmt)
		if n.Init != nil {
			w.walk(n.Init)
		}
		if n.Cond != nil {
			w.walk(n.Cond)
		}
		if n.Post != nil {
			w.walk(n.Post)
		}
		w.walk(n.Body)

	case *ast.RangeStmt:
		w.visit(n, nodetag.RangeStmt)
		if n.Key != nil {
			w.walk(n.Key)
		}
		if n.Value != nil {
			w.walk(n.Value)
		}
		w.walk(n.X)
		w.walk(n.Body)

	case *ast.ImportSpec:
		w.visit(n, nodetag.ImportSpec)
		if n.Name != nil {
			w.walk(n.Name)
		}
		w.walk(n.Path)
		if n.Comment != nil {
			w.walk(n.Comment)
		}

	case *ast.ValueSpec:
		w.visit(n, nodetag.ValueSpec)
		if n.Doc != nil {
			w.walk(n.Doc)
		}
		w.walkIdentList(n.Names)
		if n.Type != nil {
			w.walk(n.Type)
		}
		w.walkExprList(n.Values)
		if n.Comment != nil {
			w.walk(n.Comment)
		}

	case *ast.TypeSpec:
		w.visit(n, nodetag.TypeSpec)
		if n.Doc != nil {
			w.walk(n.Doc)
		}
		w.walk(n.Name)
		w.walk(n.Type)
		if n.Comment != nil {
			w.walk(n.Comment)
		}

	case *ast.GenDecl:
		w.visit(n, nodetag.GenDecl)
		if n.Doc != nil {
			w.walk(n.Doc)
		}
		for _, s := range n.Specs {
			w.walk(s)
		}

	case *ast.FuncDecl:
		w.visit(n, nodetag.FuncDecl)
		prevFunc := w.filterParams.currentFunc
		w.filterParams.currentFunc = n
		if n.Doc != nil {
			w.walk(n.Doc)
		}
		if n.Recv != nil {
			w.walk(n.Recv)
		}
		w.walk(n.Name)
		w.walk(n.Type)
		if n.Body != nil {
			w.walk(n.Body)
		}
		w.filterParams.currentFunc = prevFunc

	case *ast.File:
		w.visit(n, nodetag.File)
		w.walk(n.Name)
		w.walkDeclList(n.Decls)
	}
}
