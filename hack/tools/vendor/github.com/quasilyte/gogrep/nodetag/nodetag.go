package nodetag

import (
	"go/ast"
)

type Value int

const (
	Unknown Value = iota

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
	File
	ForStmt
	FuncDecl
	FuncLit
	FuncType
	GenDecl
	GoStmt
	Ident
	IfStmt
	ImportSpec
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

	NumBuckets

	StmtList // gogrep stmt list
	ExprList // gogrep expr list
	DeclList // gogrep decl list

	Node // ast.Node
	Expr // ast.Expr
	Stmt // ast.Stmt
)

func FromNode(n ast.Node) Value {
	switch n.(type) {
	case *ast.ArrayType:
		return ArrayType
	case *ast.AssignStmt:
		return AssignStmt
	case *ast.BasicLit:
		return BasicLit
	case *ast.BinaryExpr:
		return BinaryExpr
	case *ast.BlockStmt:
		return BlockStmt
	case *ast.BranchStmt:
		return BranchStmt
	case *ast.CallExpr:
		return CallExpr
	case *ast.CaseClause:
		return CaseClause
	case *ast.ChanType:
		return ChanType
	case *ast.CommClause:
		return CommClause
	case *ast.CompositeLit:
		return CompositeLit
	case *ast.DeclStmt:
		return DeclStmt
	case *ast.DeferStmt:
		return DeferStmt
	case *ast.Ellipsis:
		return Ellipsis
	case *ast.EmptyStmt:
		return EmptyStmt
	case *ast.ExprStmt:
		return ExprStmt
	case *ast.File:
		return File
	case *ast.ForStmt:
		return ForStmt
	case *ast.FuncDecl:
		return FuncDecl
	case *ast.FuncLit:
		return FuncLit
	case *ast.FuncType:
		return FuncType
	case *ast.GenDecl:
		return GenDecl
	case *ast.GoStmt:
		return GoStmt
	case *ast.Ident:
		return Ident
	case *ast.IfStmt:
		return IfStmt
	case *ast.ImportSpec:
		return ImportSpec
	case *ast.IncDecStmt:
		return IncDecStmt
	case *ast.IndexExpr:
		return IndexExpr
	case *ast.InterfaceType:
		return InterfaceType
	case *ast.KeyValueExpr:
		return KeyValueExpr
	case *ast.LabeledStmt:
		return LabeledStmt
	case *ast.MapType:
		return MapType
	case *ast.ParenExpr:
		return ParenExpr
	case *ast.RangeStmt:
		return RangeStmt
	case *ast.ReturnStmt:
		return ReturnStmt
	case *ast.SelectStmt:
		return SelectStmt
	case *ast.SelectorExpr:
		return SelectorExpr
	case *ast.SendStmt:
		return SendStmt
	case *ast.SliceExpr:
		return SliceExpr
	case *ast.StarExpr:
		return StarExpr
	case *ast.StructType:
		return StructType
	case *ast.SwitchStmt:
		return SwitchStmt
	case *ast.TypeAssertExpr:
		return TypeAssertExpr
	case *ast.TypeSpec:
		return TypeSpec
	case *ast.TypeSwitchStmt:
		return TypeSwitchStmt
	case *ast.UnaryExpr:
		return UnaryExpr
	case *ast.ValueSpec:
		return ValueSpec
	default:
		return Unknown
	}
}

func FromString(s string) Value {
	switch s {
	case "Expr":
		return Expr
	case "Stmt":
		return Stmt
	case "Node":
		return Node
	}

	switch s {
	case "ArrayType":
		return ArrayType
	case "AssignStmt":
		return AssignStmt
	case "BasicLit":
		return BasicLit
	case "BinaryExpr":
		return BinaryExpr
	case "BlockStmt":
		return BlockStmt
	case "BranchStmt":
		return BranchStmt
	case "CallExpr":
		return CallExpr
	case "CaseClause":
		return CaseClause
	case "ChanType":
		return ChanType
	case "CommClause":
		return CommClause
	case "CompositeLit":
		return CompositeLit
	case "DeclStmt":
		return DeclStmt
	case "DeferStmt":
		return DeferStmt
	case "Ellipsis":
		return Ellipsis
	case "EmptyStmt":
		return EmptyStmt
	case "ExprStmt":
		return ExprStmt
	case "File":
		return File
	case "ForStmt":
		return ForStmt
	case "FuncDecl":
		return FuncDecl
	case "FuncLit":
		return FuncLit
	case "FuncType":
		return FuncType
	case "GenDecl":
		return GenDecl
	case "GoStmt":
		return GoStmt
	case "Ident":
		return Ident
	case "IfStmt":
		return IfStmt
	case "ImportSpec":
		return ImportSpec
	case "IncDecStmt":
		return IncDecStmt
	case "IndexExpr":
		return IndexExpr
	case "InterfaceType":
		return InterfaceType
	case "KeyValueExpr":
		return KeyValueExpr
	case "LabeledStmt":
		return LabeledStmt
	case "MapType":
		return MapType
	case "ParenExpr":
		return ParenExpr
	case "RangeStmt":
		return RangeStmt
	case "ReturnStmt":
		return ReturnStmt
	case "SelectStmt":
		return SelectStmt
	case "SelectorExpr":
		return SelectorExpr
	case "SendStmt":
		return SendStmt
	case "SliceExpr":
		return SliceExpr
	case "StarExpr":
		return StarExpr
	case "StructType":
		return StructType
	case "SwitchStmt":
		return SwitchStmt
	case "TypeAssertExpr":
		return TypeAssertExpr
	case "TypeSpec":
		return TypeSpec
	case "TypeSwitchStmt":
		return TypeSwitchStmt
	case "UnaryExpr":
		return UnaryExpr
	case "ValueSpec":
		return ValueSpec
	default:
		return Unknown
	}
}
