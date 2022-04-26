package golang

import (
	"go/ast"
	"go/parser"
	"go/token"

	"github.com/golangci/dupl/syntax"
)

const (
	BadNode = iota
	File
	ArrayType
	AssignStmt
	BasicLit
	BinaryExpr
	BlockStmt
	BranchStmt
	CallExpr
	CaseClause
	ChanType
	CommClause
	CompositeLit
	DeclStmt
	DeferStmt
	Ellipsis
	EmptyStmt
	ExprStmt
	Field
	FieldList
	ForStmt
	FuncDecl
	FuncLit
	FuncType
	GenDecl
	GoStmt
	Ident
	IfStmt
	IncDecStmt
	IndexExpr
	InterfaceType
	KeyValueExpr
	LabeledStmt
	MapType
	ParenExpr
	RangeStmt
	ReturnStmt
	SelectStmt
	SelectorExpr
	SendStmt
	SliceExpr
	StarExpr
	StructType
	SwitchStmt
	TypeAssertExpr
	TypeSpec
	TypeSwitchStmt
	UnaryExpr
	ValueSpec
)

// Parse the given file and return uniform syntax tree.
func Parse(filename string) (*syntax.Node, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, 0)
	if err != nil {
		return nil, err
	}
	t := &transformer{
		fileset:  fset,
		filename: filename,
	}
	return t.trans(file), nil
}

type transformer struct {
	fileset  *token.FileSet
	filename string
}

// trans transforms given golang AST to uniform tree structure.
func (t *transformer) trans(node ast.Node) (o *syntax.Node) {
	o = syntax.NewNode()
	o.Filename = t.filename
	st, end := node.Pos(), node.End()
	o.Pos, o.End = t.fileset.File(st).Offset(st), t.fileset.File(end).Offset(end)

	switch n := node.(type) {
	case *ast.ArrayType:
		o.Type = ArrayType
		if n.Len != nil {
			o.AddChildren(t.trans(n.Len))
		}
		o.AddChildren(t.trans(n.Elt))

	case *ast.AssignStmt:
		o.Type = AssignStmt
		for _, e := range n.Rhs {
			o.AddChildren(t.trans(e))
		}

		for _, e := range n.Lhs {
			o.AddChildren(t.trans(e))
		}

	case *ast.BasicLit:
		o.Type = BasicLit

	case *ast.BinaryExpr:
		o.Type = BinaryExpr
		o.AddChildren(t.trans(n.X), t.trans(n.Y))

	case *ast.BlockStmt:
		o.Type = BlockStmt
		for _, stmt := range n.List {
			o.AddChildren(t.trans(stmt))
		}

	case *ast.BranchStmt:
		o.Type = BranchStmt
		if n.Label != nil {
			o.AddChildren(t.trans(n.Label))
		}

	case *ast.CallExpr:
		o.Type = CallExpr
		o.AddChildren(t.trans(n.Fun))
		for _, arg := range n.Args {
			o.AddChildren(t.trans(arg))
		}

	case *ast.CaseClause:
		o.Type = CaseClause
		for _, e := range n.List {
			o.AddChildren(t.trans(e))
		}
		for _, stmt := range n.Body {
			o.AddChildren(t.trans(stmt))
		}

	case *ast.ChanType:
		o.Type = ChanType
		o.AddChildren(t.trans(n.Value))

	case *ast.CommClause:
		o.Type = CommClause
		if n.Comm != nil {
			o.AddChildren(t.trans(n.Comm))
		}
		for _, stmt := range n.Body {
			o.AddChildren(t.trans(stmt))
		}

	case *ast.CompositeLit:
		o.Type = CompositeLit
		if n.Type != nil {
			o.AddChildren(t.trans(n.Type))
		}
		for _, e := range n.Elts {
			o.AddChildren(t.trans(e))
		}

	case *ast.DeclStmt:
		o.Type = DeclStmt
		o.AddChildren(t.trans(n.Decl))

	case *ast.DeferStmt:
		o.Type = DeferStmt
		o.AddChildren(t.trans(n.Call))

	case *ast.Ellipsis:
		o.Type = Ellipsis
		if n.Elt != nil {
			o.AddChildren(t.trans(n.Elt))
		}

	case *ast.EmptyStmt:
		o.Type = EmptyStmt

	case *ast.ExprStmt:
		o.Type = ExprStmt
		o.AddChildren(t.trans(n.X))

	case *ast.Field:
		o.Type = Field
		for _, name := range n.Names {
			o.AddChildren(t.trans(name))
		}
		o.AddChildren(t.trans(n.Type))

	case *ast.FieldList:
		o.Type = FieldList
		for _, field := range n.List {
			o.AddChildren(t.trans(field))
		}

	case *ast.File:
		o.Type = File
		for _, decl := range n.Decls {
			if genDecl, ok := decl.(*ast.GenDecl); ok && genDecl.Tok == token.IMPORT {
				// skip import declarations
				continue
			}
			o.AddChildren(t.trans(decl))
		}

	case *ast.ForStmt:
		o.Type = ForStmt
		if n.Init != nil {
			o.AddChildren(t.trans(n.Init))
		}
		if n.Cond != nil {
			o.AddChildren(t.trans(n.Cond))
		}
		if n.Post != nil {
			o.AddChildren(t.trans(n.Post))
		}
		o.AddChildren(t.trans(n.Body))

	case *ast.FuncDecl:
		o.Type = FuncDecl
		if n.Recv != nil {
			o.AddChildren(t.trans(n.Recv))
		}
		o.AddChildren(t.trans(n.Name), t.trans(n.Type))
		if n.Body != nil {
			o.AddChildren(t.trans(n.Body))
		}

	case *ast.FuncLit:
		o.Type = FuncLit
		o.AddChildren(t.trans(n.Type), t.trans(n.Body))

	case *ast.FuncType:
		o.Type = FuncType
		o.AddChildren(t.trans(n.Params))
		if n.Results != nil {
			o.AddChildren(t.trans(n.Results))
		}

	case *ast.GenDecl:
		o.Type = GenDecl
		for _, spec := range n.Specs {
			o.AddChildren(t.trans(spec))
		}

	case *ast.GoStmt:
		o.Type = GoStmt
		o.AddChildren(t.trans(n.Call))

	case *ast.Ident:
		o.Type = Ident

	case *ast.IfStmt:
		o.Type = IfStmt
		if n.Init != nil {
			o.AddChildren(t.trans(n.Init))
		}
		o.AddChildren(t.trans(n.Cond), t.trans(n.Body))
		if n.Else != nil {
			o.AddChildren(t.trans(n.Else))
		}

	case *ast.IncDecStmt:
		o.Type = IncDecStmt
		o.AddChildren(t.trans(n.X))

	case *ast.IndexExpr:
		o.Type = IndexExpr
		o.AddChildren(t.trans(n.X), t.trans(n.Index))

	case *ast.InterfaceType:
		o.Type = InterfaceType
		o.AddChildren(t.trans(n.Methods))

	case *ast.KeyValueExpr:
		o.Type = KeyValueExpr
		o.AddChildren(t.trans(n.Key), t.trans(n.Value))

	case *ast.LabeledStmt:
		o.Type = LabeledStmt
		o.AddChildren(t.trans(n.Label), t.trans(n.Stmt))

	case *ast.MapType:
		o.Type = MapType
		o.AddChildren(t.trans(n.Key), t.trans(n.Value))

	case *ast.ParenExpr:
		o.Type = ParenExpr
		o.AddChildren(t.trans(n.X))

	case *ast.RangeStmt:
		o.Type = RangeStmt
		if n.Key != nil {
			o.AddChildren(t.trans(n.Key))
		}
		if n.Value != nil {
			o.AddChildren(t.trans(n.Value))
		}
		o.AddChildren(t.trans(n.X), t.trans(n.Body))

	case *ast.ReturnStmt:
		o.Type = ReturnStmt
		for _, e := range n.Results {
			o.AddChildren(t.trans(e))
		}

	case *ast.SelectStmt:
		o.Type = SelectStmt
		o.AddChildren(t.trans(n.Body))

	case *ast.SelectorExpr:
		o.Type = SelectorExpr
		o.AddChildren(t.trans(n.X), t.trans(n.Sel))

	case *ast.SendStmt:
		o.Type = SendStmt
		o.AddChildren(t.trans(n.Chan), t.trans(n.Value))

	case *ast.SliceExpr:
		o.Type = SliceExpr
		o.AddChildren(t.trans(n.X))
		if n.Low != nil {
			o.AddChildren(t.trans(n.Low))
		}
		if n.High != nil {
			o.AddChildren(t.trans(n.High))
		}
		if n.Max != nil {
			o.AddChildren(t.trans(n.Max))
		}

	case *ast.StarExpr:
		o.Type = StarExpr
		o.AddChildren(t.trans(n.X))

	case *ast.StructType:
		o.Type = StructType
		o.AddChildren(t.trans(n.Fields))

	case *ast.SwitchStmt:
		o.Type = SwitchStmt
		if n.Init != nil {
			o.AddChildren(t.trans(n.Init))
		}
		if n.Tag != nil {
			o.AddChildren(t.trans(n.Tag))
		}
		o.AddChildren(t.trans(n.Body))

	case *ast.TypeAssertExpr:
		o.Type = TypeAssertExpr
		o.AddChildren(t.trans(n.X))
		if n.Type != nil {
			o.AddChildren(t.trans(n.Type))
		}

	case *ast.TypeSpec:
		o.Type = TypeSpec
		o.AddChildren(t.trans(n.Name), t.trans(n.Type))

	case *ast.TypeSwitchStmt:
		o.Type = TypeSwitchStmt
		if n.Init != nil {
			o.AddChildren(t.trans(n.Init))
		}
		o.AddChildren(t.trans(n.Assign), t.trans(n.Body))

	case *ast.UnaryExpr:
		o.Type = UnaryExpr
		o.AddChildren(t.trans(n.X))

	case *ast.ValueSpec:
		o.Type = ValueSpec
		for _, name := range n.Names {
			o.AddChildren(t.trans(name))
		}
		if n.Type != nil {
			o.AddChildren(t.trans(n.Type))
		}
		for _, val := range n.Values {
			o.AddChildren(t.trans(val))
		}

	default:
		o.Type = BadNode

	}

	return o
}
