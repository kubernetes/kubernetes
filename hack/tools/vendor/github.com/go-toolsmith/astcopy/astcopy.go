// Package astcopy implements Go AST reflection-free deep copy operations.
package astcopy

import (
	"go/ast"
)

// Node returns x node deep copy.
// Copy of nil argument is nil.
func Node(x ast.Node) ast.Node {
	return copyNode(x)
}

// NodeList returns xs node slice deep copy.
// Copy of nil argument is nil.
func NodeList(xs []ast.Node) []ast.Node {
	if xs == nil {
		return nil
	}
	cp := make([]ast.Node, len(xs))
	for i := range xs {
		cp[i] = copyNode(xs[i])
	}
	return cp
}

// Expr returns x expression deep copy.
// Copy of nil argument is nil.
func Expr(x ast.Expr) ast.Expr {
	return copyExpr(x)
}

// ExprList returns xs expression slice deep copy.
// Copy of nil argument is nil.
func ExprList(xs []ast.Expr) []ast.Expr {
	if xs == nil {
		return nil
	}
	cp := make([]ast.Expr, len(xs))
	for i := range xs {
		cp[i] = copyExpr(xs[i])
	}
	return cp
}

// Stmt returns x statement deep copy.
// Copy of nil argument is nil.
func Stmt(x ast.Stmt) ast.Stmt {
	return copyStmt(x)
}

// StmtList returns xs statement slice deep copy.
// Copy of nil argument is nil.
func StmtList(xs []ast.Stmt) []ast.Stmt {
	if xs == nil {
		return nil
	}
	cp := make([]ast.Stmt, len(xs))
	for i := range xs {
		cp[i] = copyStmt(xs[i])
	}
	return cp
}

// Decl returns x declaration deep copy.
// Copy of nil argument is nil.
func Decl(x ast.Decl) ast.Decl {
	return copyDecl(x)
}

// DeclList returns xs declaration slice deep copy.
// Copy of nil argument is nil.
func DeclList(xs []ast.Decl) []ast.Decl {
	if xs == nil {
		return nil
	}
	cp := make([]ast.Decl, len(xs))
	for i := range xs {
		cp[i] = copyDecl(xs[i])
	}
	return cp
}

// BadExpr returns x deep copy.
// Copy of nil argument is nil.
func BadExpr(x *ast.BadExpr) *ast.BadExpr {
	if x == nil {
		return nil
	}
	cp := *x
	return &cp
}

// Ident returns x deep copy.
// Copy of nil argument is nil.
func Ident(x *ast.Ident) *ast.Ident {
	if x == nil {
		return nil
	}
	cp := *x
	return &cp
}

// IdentList returns xs identifier slice deep copy.
// Copy of nil argument is nil.
func IdentList(xs []*ast.Ident) []*ast.Ident {
	if xs == nil {
		return nil
	}
	cp := make([]*ast.Ident, len(xs))
	for i := range xs {
		cp[i] = Ident(xs[i])
	}
	return cp
}

// Ellipsis returns x deep copy.
// Copy of nil argument is nil.
func Ellipsis(x *ast.Ellipsis) *ast.Ellipsis {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Elt = copyExpr(x.Elt)
	return &cp
}

// BasicLit returns x deep copy.
// Copy of nil argument is nil.
func BasicLit(x *ast.BasicLit) *ast.BasicLit {
	if x == nil {
		return nil
	}
	cp := *x
	return &cp
}

// FuncLit returns x deep copy.
// Copy of nil argument is nil.
func FuncLit(x *ast.FuncLit) *ast.FuncLit {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Type = FuncType(x.Type)
	cp.Body = BlockStmt(x.Body)
	return &cp
}

// CompositeLit returns x deep copy.
// Copy of nil argument is nil.
func CompositeLit(x *ast.CompositeLit) *ast.CompositeLit {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Type = copyExpr(x.Type)
	cp.Elts = ExprList(x.Elts)
	return &cp
}

// ParenExpr returns x deep copy.
// Copy of nil argument is nil.
func ParenExpr(x *ast.ParenExpr) *ast.ParenExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	return &cp
}

// SelectorExpr returns x deep copy.
// Copy of nil argument is nil.
func SelectorExpr(x *ast.SelectorExpr) *ast.SelectorExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	cp.Sel = Ident(x.Sel)
	return &cp
}

// IndexExpr returns x deep copy.
// Copy of nil argument is nil.
func IndexExpr(x *ast.IndexExpr) *ast.IndexExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	cp.Index = copyExpr(x.Index)
	return &cp
}

// SliceExpr returns x deep copy.
// Copy of nil argument is nil.
func SliceExpr(x *ast.SliceExpr) *ast.SliceExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	cp.Low = copyExpr(x.Low)
	cp.High = copyExpr(x.High)
	cp.Max = copyExpr(x.Max)
	return &cp
}

// TypeAssertExpr returns x deep copy.
// Copy of nil argument is nil.
func TypeAssertExpr(x *ast.TypeAssertExpr) *ast.TypeAssertExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	cp.Type = copyExpr(x.Type)
	return &cp
}

// CallExpr returns x deep copy.
// Copy of nil argument is nil.
func CallExpr(x *ast.CallExpr) *ast.CallExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Fun = copyExpr(x.Fun)
	cp.Args = ExprList(x.Args)
	return &cp
}

// StarExpr returns x deep copy.
// Copy of nil argument is nil.
func StarExpr(x *ast.StarExpr) *ast.StarExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	return &cp
}

// UnaryExpr returns x deep copy.
// Copy of nil argument is nil.
func UnaryExpr(x *ast.UnaryExpr) *ast.UnaryExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	return &cp
}

// BinaryExpr returns x deep copy.
// Copy of nil argument is nil.
func BinaryExpr(x *ast.BinaryExpr) *ast.BinaryExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	cp.Y = copyExpr(x.Y)
	return &cp
}

// KeyValueExpr returns x deep copy.
// Copy of nil argument is nil.
func KeyValueExpr(x *ast.KeyValueExpr) *ast.KeyValueExpr {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Key = copyExpr(x.Key)
	cp.Value = copyExpr(x.Value)
	return &cp
}

// ArrayType returns x deep copy.
// Copy of nil argument is nil.
func ArrayType(x *ast.ArrayType) *ast.ArrayType {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Len = copyExpr(x.Len)
	cp.Elt = copyExpr(x.Elt)
	return &cp
}

// StructType returns x deep copy.
// Copy of nil argument is nil.
func StructType(x *ast.StructType) *ast.StructType {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Fields = FieldList(x.Fields)
	return &cp
}

// Field returns x deep copy.
// Copy of nil argument is nil.
func Field(x *ast.Field) *ast.Field {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Names = IdentList(x.Names)
	cp.Type = copyExpr(x.Type)
	cp.Tag = BasicLit(x.Tag)
	cp.Doc = CommentGroup(x.Doc)
	cp.Comment = CommentGroup(x.Comment)
	return &cp
}

// FieldList returns x deep copy.
// Copy of nil argument is nil.
func FieldList(x *ast.FieldList) *ast.FieldList {
	if x == nil {
		return nil
	}
	cp := *x
	if x.List != nil {
		cp.List = make([]*ast.Field, len(x.List))
		for i := range x.List {
			cp.List[i] = Field(x.List[i])
		}
	}
	return &cp
}

// FuncType returns x deep copy.
// Copy of nil argument is nil.
func FuncType(x *ast.FuncType) *ast.FuncType {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Params = FieldList(x.Params)
	cp.Results = FieldList(x.Results)
	return &cp
}

// InterfaceType returns x deep copy.
// Copy of nil argument is nil.
func InterfaceType(x *ast.InterfaceType) *ast.InterfaceType {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Methods = FieldList(x.Methods)
	return &cp
}

// MapType returns x deep copy.
// Copy of nil argument is nil.
func MapType(x *ast.MapType) *ast.MapType {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Key = copyExpr(x.Key)
	cp.Value = copyExpr(x.Value)
	return &cp
}

// ChanType returns x deep copy.
// Copy of nil argument is nil.
func ChanType(x *ast.ChanType) *ast.ChanType {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Value = copyExpr(x.Value)
	return &cp
}

// BlockStmt returns x deep copy.
// Copy of nil argument is nil.
func BlockStmt(x *ast.BlockStmt) *ast.BlockStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.List = StmtList(x.List)
	return &cp
}

// ImportSpec returns x deep copy.
// Copy of nil argument is nil.
func ImportSpec(x *ast.ImportSpec) *ast.ImportSpec {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Name = Ident(x.Name)
	cp.Path = BasicLit(x.Path)
	cp.Doc = CommentGroup(x.Doc)
	cp.Comment = CommentGroup(x.Comment)
	return &cp
}

// ValueSpec returns x deep copy.
// Copy of nil argument is nil.
func ValueSpec(x *ast.ValueSpec) *ast.ValueSpec {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Names = IdentList(x.Names)
	cp.Values = ExprList(x.Values)
	cp.Type = copyExpr(x.Type)
	cp.Doc = CommentGroup(x.Doc)
	cp.Comment = CommentGroup(x.Comment)
	return &cp
}

// TypeSpec returns x deep copy.
// Copy of nil argument is nil.
func TypeSpec(x *ast.TypeSpec) *ast.TypeSpec {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Name = Ident(x.Name)
	cp.Type = copyExpr(x.Type)
	cp.Doc = CommentGroup(x.Doc)
	cp.Comment = CommentGroup(x.Comment)
	return &cp
}

// Spec returns x deep copy.
// Copy of nil argument is nil.
func Spec(x ast.Spec) ast.Spec {
	if x == nil {
		return nil
	}

	switch x := x.(type) {
	case *ast.ImportSpec:
		return ImportSpec(x)
	case *ast.ValueSpec:
		return ValueSpec(x)
	case *ast.TypeSpec:
		return TypeSpec(x)
	default:
		panic("unhandled spec")
	}
}

// SpecList returns xs spec slice deep copy.
// Copy of nil argument is nil.
func SpecList(xs []ast.Spec) []ast.Spec {
	if xs == nil {
		return nil
	}
	cp := make([]ast.Spec, len(xs))
	for i := range xs {
		cp[i] = Spec(xs[i])
	}
	return cp
}

// BadStmt returns x deep copy.
// Copy of nil argument is nil.
func BadStmt(x *ast.BadStmt) *ast.BadStmt {
	if x == nil {
		return nil
	}
	cp := *x
	return &cp
}

// DeclStmt returns x deep copy.
// Copy of nil argument is nil.
func DeclStmt(x *ast.DeclStmt) *ast.DeclStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Decl = copyDecl(x.Decl)
	return &cp
}

// EmptyStmt returns x deep copy.
// Copy of nil argument is nil.
func EmptyStmt(x *ast.EmptyStmt) *ast.EmptyStmt {
	if x == nil {
		return nil
	}
	cp := *x
	return &cp
}

// LabeledStmt returns x deep copy.
// Copy of nil argument is nil.
func LabeledStmt(x *ast.LabeledStmt) *ast.LabeledStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Label = Ident(x.Label)
	cp.Stmt = copyStmt(x.Stmt)
	return &cp
}

// ExprStmt returns x deep copy.
// Copy of nil argument is nil.
func ExprStmt(x *ast.ExprStmt) *ast.ExprStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	return &cp
}

// SendStmt returns x deep copy.
// Copy of nil argument is nil.
func SendStmt(x *ast.SendStmt) *ast.SendStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Chan = copyExpr(x.Chan)
	cp.Value = copyExpr(x.Value)
	return &cp
}

// IncDecStmt returns x deep copy.
// Copy of nil argument is nil.
func IncDecStmt(x *ast.IncDecStmt) *ast.IncDecStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.X = copyExpr(x.X)
	return &cp
}

// AssignStmt returns x deep copy.
// Copy of nil argument is nil.
func AssignStmt(x *ast.AssignStmt) *ast.AssignStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Lhs = ExprList(x.Lhs)
	cp.Rhs = ExprList(x.Rhs)
	return &cp
}

// GoStmt returns x deep copy.
// Copy of nil argument is nil.
func GoStmt(x *ast.GoStmt) *ast.GoStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Call = CallExpr(x.Call)
	return &cp
}

// DeferStmt returns x deep copy.
// Copy of nil argument is nil.
func DeferStmt(x *ast.DeferStmt) *ast.DeferStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Call = CallExpr(x.Call)
	return &cp
}

// ReturnStmt returns x deep copy.
// Copy of nil argument is nil.
func ReturnStmt(x *ast.ReturnStmt) *ast.ReturnStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Results = ExprList(x.Results)
	return &cp
}

// BranchStmt returns x deep copy.
// Copy of nil argument is nil.
func BranchStmt(x *ast.BranchStmt) *ast.BranchStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Label = Ident(x.Label)
	return &cp
}

// IfStmt returns x deep copy.
// Copy of nil argument is nil.
func IfStmt(x *ast.IfStmt) *ast.IfStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Init = copyStmt(x.Init)
	cp.Cond = copyExpr(x.Cond)
	cp.Body = BlockStmt(x.Body)
	cp.Else = copyStmt(x.Else)
	return &cp
}

// CaseClause returns x deep copy.
// Copy of nil argument is nil.
func CaseClause(x *ast.CaseClause) *ast.CaseClause {
	if x == nil {
		return nil
	}
	cp := *x
	cp.List = ExprList(x.List)
	cp.Body = StmtList(x.Body)
	return &cp
}

// SwitchStmt returns x deep copy.
// Copy of nil argument is nil.
func SwitchStmt(x *ast.SwitchStmt) *ast.SwitchStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Init = copyStmt(x.Init)
	cp.Tag = copyExpr(x.Tag)
	cp.Body = BlockStmt(x.Body)
	return &cp
}

// TypeSwitchStmt returns x deep copy.
// Copy of nil argument is nil.
func TypeSwitchStmt(x *ast.TypeSwitchStmt) *ast.TypeSwitchStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Init = copyStmt(x.Init)
	cp.Assign = copyStmt(x.Assign)
	cp.Body = BlockStmt(x.Body)
	return &cp
}

// CommClause returns x deep copy.
// Copy of nil argument is nil.
func CommClause(x *ast.CommClause) *ast.CommClause {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Comm = copyStmt(x.Comm)
	cp.Body = StmtList(x.Body)
	return &cp
}

// SelectStmt returns x deep copy.
// Copy of nil argument is nil.
func SelectStmt(x *ast.SelectStmt) *ast.SelectStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Body = BlockStmt(x.Body)
	return &cp
}

// ForStmt returns x deep copy.
// Copy of nil argument is nil.
func ForStmt(x *ast.ForStmt) *ast.ForStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Init = copyStmt(x.Init)
	cp.Cond = copyExpr(x.Cond)
	cp.Post = copyStmt(x.Post)
	cp.Body = BlockStmt(x.Body)
	return &cp
}

// RangeStmt returns x deep copy.
// Copy of nil argument is nil.
func RangeStmt(x *ast.RangeStmt) *ast.RangeStmt {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Key = copyExpr(x.Key)
	cp.Value = copyExpr(x.Value)
	cp.X = copyExpr(x.X)
	cp.Body = BlockStmt(x.Body)
	return &cp
}

// Comment returns x deep copy.
// Copy of nil argument is nil.
func Comment(x *ast.Comment) *ast.Comment {
	if x == nil {
		return nil
	}
	cp := *x
	return &cp
}

// CommentGroup returns x deep copy.
// Copy of nil argument is nil.
func CommentGroup(x *ast.CommentGroup) *ast.CommentGroup {
	if x == nil {
		return nil
	}
	cp := *x
	if x.List != nil {
		cp.List = make([]*ast.Comment, len(x.List))
		for i := range x.List {
			cp.List[i] = Comment(x.List[i])
		}
	}
	return &cp
}

// File returns x deep copy.
// Copy of nil argument is nil.
func File(x *ast.File) *ast.File {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Doc = CommentGroup(x.Doc)
	cp.Name = Ident(x.Name)
	cp.Decls = DeclList(x.Decls)
	cp.Imports = make([]*ast.ImportSpec, len(x.Imports))
	for i := range x.Imports {
		cp.Imports[i] = ImportSpec(x.Imports[i])
	}
	cp.Unresolved = IdentList(x.Unresolved)
	cp.Comments = make([]*ast.CommentGroup, len(x.Comments))
	for i := range x.Comments {
		cp.Comments[i] = CommentGroup(x.Comments[i])
	}
	return &cp
}

// Package returns x deep copy.
// Copy of nil argument is nil.
func Package(x *ast.Package) *ast.Package {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Files = make(map[string]*ast.File)
	for filename, f := range x.Files {
		cp.Files[filename] = f
	}
	return &cp
}

// BadDecl returns x deep copy.
// Copy of nil argument is nil.
func BadDecl(x *ast.BadDecl) *ast.BadDecl {
	if x == nil {
		return nil
	}
	cp := *x
	return &cp
}

// GenDecl returns x deep copy.
// Copy of nil argument is nil.
func GenDecl(x *ast.GenDecl) *ast.GenDecl {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Specs = SpecList(x.Specs)
	cp.Doc = CommentGroup(x.Doc)
	return &cp
}

// FuncDecl returns x deep copy.
// Copy of nil argument is nil.
func FuncDecl(x *ast.FuncDecl) *ast.FuncDecl {
	if x == nil {
		return nil
	}
	cp := *x
	cp.Recv = FieldList(x.Recv)
	cp.Name = Ident(x.Name)
	cp.Type = FuncType(x.Type)
	cp.Body = BlockStmt(x.Body)
	cp.Doc = CommentGroup(x.Doc)
	return &cp
}

func copyNode(x ast.Node) ast.Node {
	switch x := x.(type) {
	case ast.Expr:
		return copyExpr(x)
	case ast.Stmt:
		return copyStmt(x)
	case ast.Decl:
		return copyDecl(x)

	case ast.Spec:
		return Spec(x)
	case *ast.FieldList:
		return FieldList(x)
	case *ast.Comment:
		return Comment(x)
	case *ast.CommentGroup:
		return CommentGroup(x)
	case *ast.File:
		return File(x)
	case *ast.Package:
		return Package(x)

	default:
		panic("unhandled node")
	}
}

func copyExpr(x ast.Expr) ast.Expr {
	if x == nil {
		return nil
	}

	switch x := x.(type) {
	case *ast.BadExpr:
		return BadExpr(x)
	case *ast.Ident:
		return Ident(x)
	case *ast.Ellipsis:
		return Ellipsis(x)
	case *ast.BasicLit:
		return BasicLit(x)
	case *ast.FuncLit:
		return FuncLit(x)
	case *ast.CompositeLit:
		return CompositeLit(x)
	case *ast.ParenExpr:
		return ParenExpr(x)
	case *ast.SelectorExpr:
		return SelectorExpr(x)
	case *ast.IndexExpr:
		return IndexExpr(x)
	case *ast.SliceExpr:
		return SliceExpr(x)
	case *ast.TypeAssertExpr:
		return TypeAssertExpr(x)
	case *ast.CallExpr:
		return CallExpr(x)
	case *ast.StarExpr:
		return StarExpr(x)
	case *ast.UnaryExpr:
		return UnaryExpr(x)
	case *ast.BinaryExpr:
		return BinaryExpr(x)
	case *ast.KeyValueExpr:
		return KeyValueExpr(x)
	case *ast.ArrayType:
		return ArrayType(x)
	case *ast.StructType:
		return StructType(x)
	case *ast.FuncType:
		return FuncType(x)
	case *ast.InterfaceType:
		return InterfaceType(x)
	case *ast.MapType:
		return MapType(x)
	case *ast.ChanType:
		return ChanType(x)

	default:
		panic("unhandled expr")
	}
}

func copyStmt(x ast.Stmt) ast.Stmt {
	if x == nil {
		return nil
	}

	switch x := x.(type) {
	case *ast.BadStmt:
		return BadStmt(x)
	case *ast.DeclStmt:
		return DeclStmt(x)
	case *ast.EmptyStmt:
		return EmptyStmt(x)
	case *ast.LabeledStmt:
		return LabeledStmt(x)
	case *ast.ExprStmt:
		return ExprStmt(x)
	case *ast.SendStmt:
		return SendStmt(x)
	case *ast.IncDecStmt:
		return IncDecStmt(x)
	case *ast.AssignStmt:
		return AssignStmt(x)
	case *ast.GoStmt:
		return GoStmt(x)
	case *ast.DeferStmt:
		return DeferStmt(x)
	case *ast.ReturnStmt:
		return ReturnStmt(x)
	case *ast.BranchStmt:
		return BranchStmt(x)
	case *ast.BlockStmt:
		return BlockStmt(x)
	case *ast.IfStmt:
		return IfStmt(x)
	case *ast.CaseClause:
		return CaseClause(x)
	case *ast.SwitchStmt:
		return SwitchStmt(x)
	case *ast.TypeSwitchStmt:
		return TypeSwitchStmt(x)
	case *ast.CommClause:
		return CommClause(x)
	case *ast.SelectStmt:
		return SelectStmt(x)
	case *ast.ForStmt:
		return ForStmt(x)
	case *ast.RangeStmt:
		return RangeStmt(x)

	default:
		panic("unhandled stmt")
	}
}

func copyDecl(x ast.Decl) ast.Decl {
	if x == nil {
		return nil
	}

	switch x := x.(type) {
	case *ast.BadDecl:
		return BadDecl(x)
	case *ast.GenDecl:
		return GenDecl(x)
	case *ast.FuncDecl:
		return FuncDecl(x)

	default:
		panic("unhandled decl")
	}
}
