package pattern

import (
	"fmt"
	"go/token"
	"reflect"
	"strings"
)

var (
	_ Node = Ellipsis{}
	_ Node = Binding{}
	_ Node = RangeStmt{}
	_ Node = AssignStmt{}
	_ Node = IndexExpr{}
	_ Node = Ident{}
	_ Node = Builtin{}
	_ Node = String("")
	_ Node = Any{}
	_ Node = ValueSpec{}
	_ Node = List{}
	_ Node = GenDecl{}
	_ Node = BinaryExpr{}
	_ Node = ForStmt{}
	_ Node = ArrayType{}
	_ Node = DeferStmt{}
	_ Node = MapType{}
	_ Node = ReturnStmt{}
	_ Node = SliceExpr{}
	_ Node = StarExpr{}
	_ Node = UnaryExpr{}
	_ Node = SendStmt{}
	_ Node = SelectStmt{}
	_ Node = ImportSpec{}
	_ Node = IfStmt{}
	_ Node = GoStmt{}
	_ Node = Field{}
	_ Node = SelectorExpr{}
	_ Node = StructType{}
	_ Node = KeyValueExpr{}
	_ Node = FuncType{}
	_ Node = FuncLit{}
	_ Node = FuncDecl{}
	_ Node = Token(0)
	_ Node = ChanType{}
	_ Node = CallExpr{}
	_ Node = CaseClause{}
	_ Node = CommClause{}
	_ Node = CompositeLit{}
	_ Node = EmptyStmt{}
	_ Node = SwitchStmt{}
	_ Node = TypeSwitchStmt{}
	_ Node = TypeAssertExpr{}
	_ Node = TypeSpec{}
	_ Node = InterfaceType{}
	_ Node = BranchStmt{}
	_ Node = IncDecStmt{}
	_ Node = BasicLit{}
	_ Node = Nil{}
	_ Node = Object{}
	_ Node = Function{}
	_ Node = Not{}
	_ Node = Or{}
)

type Function struct {
	Name Node
}

type Token token.Token

type Nil struct {
}

type Ellipsis struct {
	Elt Node
}

type IncDecStmt struct {
	X   Node
	Tok Node
}

type BranchStmt struct {
	Tok   Node
	Label Node
}

type InterfaceType struct {
	Methods Node
}

type TypeSpec struct {
	Name Node
	Type Node
}

type TypeAssertExpr struct {
	X    Node
	Type Node
}

type TypeSwitchStmt struct {
	Init   Node
	Assign Node
	Body   Node
}

type SwitchStmt struct {
	Init Node
	Tag  Node
	Body Node
}

type EmptyStmt struct {
}

type CompositeLit struct {
	Type Node
	Elts Node
}

type CommClause struct {
	Comm Node
	Body Node
}

type CaseClause struct {
	List Node
	Body Node
}

type CallExpr struct {
	Fun  Node
	Args Node
	// XXX handle ellipsis
}

// TODO(dh): add a ChanDir node, and a way of instantiating it.

type ChanType struct {
	Dir   Node
	Value Node
}

type FuncDecl struct {
	Recv Node
	Name Node
	Type Node
	Body Node
}

type FuncLit struct {
	Type Node
	Body Node
}

type FuncType struct {
	Params  Node
	Results Node
}

type KeyValueExpr struct {
	Key   Node
	Value Node
}

type StructType struct {
	Fields Node
}

type SelectorExpr struct {
	X   Node
	Sel Node
}

type Field struct {
	Names Node
	Type  Node
	Tag   Node
}

type GoStmt struct {
	Call Node
}

type IfStmt struct {
	Init Node
	Cond Node
	Body Node
	Else Node
}

type ImportSpec struct {
	Name Node
	Path Node
}

type SelectStmt struct {
	Body Node
}

type ArrayType struct {
	Len Node
	Elt Node
}

type DeferStmt struct {
	Call Node
}

type MapType struct {
	Key   Node
	Value Node
}

type ReturnStmt struct {
	Results Node
}

type SliceExpr struct {
	X    Node
	Low  Node
	High Node
	Max  Node
}

type StarExpr struct {
	X Node
}

type UnaryExpr struct {
	Op Node
	X  Node
}

type SendStmt struct {
	Chan  Node
	Value Node
}

type Binding struct {
	Name string
	Node Node
}

type RangeStmt struct {
	Key   Node
	Value Node
	Tok   Node
	X     Node
	Body  Node
}

type AssignStmt struct {
	Lhs Node
	Tok Node
	Rhs Node
}

type IndexExpr struct {
	X     Node
	Index Node
}

type Node interface {
	String() string
	isNode()
}

type Ident struct {
	Name Node
}

type Object struct {
	Name Node
}

type Builtin struct {
	Name Node
}

type String string

type Any struct{}

type ValueSpec struct {
	Names  Node
	Type   Node
	Values Node
}

type List struct {
	Head Node
	Tail Node
}

type GenDecl struct {
	Tok   Node
	Specs Node
}

type BasicLit struct {
	Kind  Node
	Value Node
}

type BinaryExpr struct {
	X  Node
	Op Node
	Y  Node
}

type ForStmt struct {
	Init Node
	Cond Node
	Post Node
	Body Node
}

type Or struct {
	Nodes []Node
}

type Not struct {
	Node Node
}

func stringify(n Node) string {
	v := reflect.ValueOf(n)
	var parts []string
	parts = append(parts, v.Type().Name())
	for i := 0; i < v.NumField(); i++ {
		//lint:ignore S1025 false positive in staticcheck 2019.2.3
		parts = append(parts, fmt.Sprintf("%s", v.Field(i)))
	}
	return "(" + strings.Join(parts, " ") + ")"
}

func (stmt AssignStmt) String() string     { return stringify(stmt) }
func (expr IndexExpr) String() string      { return stringify(expr) }
func (id Ident) String() string            { return stringify(id) }
func (spec ValueSpec) String() string      { return stringify(spec) }
func (decl GenDecl) String() string        { return stringify(decl) }
func (lit BasicLit) String() string        { return stringify(lit) }
func (expr BinaryExpr) String() string     { return stringify(expr) }
func (stmt ForStmt) String() string        { return stringify(stmt) }
func (stmt RangeStmt) String() string      { return stringify(stmt) }
func (typ ArrayType) String() string       { return stringify(typ) }
func (stmt DeferStmt) String() string      { return stringify(stmt) }
func (typ MapType) String() string         { return stringify(typ) }
func (stmt ReturnStmt) String() string     { return stringify(stmt) }
func (expr SliceExpr) String() string      { return stringify(expr) }
func (expr StarExpr) String() string       { return stringify(expr) }
func (expr UnaryExpr) String() string      { return stringify(expr) }
func (stmt SendStmt) String() string       { return stringify(stmt) }
func (spec ImportSpec) String() string     { return stringify(spec) }
func (stmt SelectStmt) String() string     { return stringify(stmt) }
func (stmt IfStmt) String() string         { return stringify(stmt) }
func (stmt IncDecStmt) String() string     { return stringify(stmt) }
func (stmt GoStmt) String() string         { return stringify(stmt) }
func (field Field) String() string         { return stringify(field) }
func (expr SelectorExpr) String() string   { return stringify(expr) }
func (typ StructType) String() string      { return stringify(typ) }
func (expr KeyValueExpr) String() string   { return stringify(expr) }
func (typ FuncType) String() string        { return stringify(typ) }
func (lit FuncLit) String() string         { return stringify(lit) }
func (decl FuncDecl) String() string       { return stringify(decl) }
func (stmt BranchStmt) String() string     { return stringify(stmt) }
func (expr CallExpr) String() string       { return stringify(expr) }
func (clause CaseClause) String() string   { return stringify(clause) }
func (typ ChanType) String() string        { return stringify(typ) }
func (clause CommClause) String() string   { return stringify(clause) }
func (lit CompositeLit) String() string    { return stringify(lit) }
func (stmt EmptyStmt) String() string      { return stringify(stmt) }
func (typ InterfaceType) String() string   { return stringify(typ) }
func (stmt SwitchStmt) String() string     { return stringify(stmt) }
func (expr TypeAssertExpr) String() string { return stringify(expr) }
func (spec TypeSpec) String() string       { return stringify(spec) }
func (stmt TypeSwitchStmt) String() string { return stringify(stmt) }
func (nil Nil) String() string             { return "nil" }
func (builtin Builtin) String() string     { return stringify(builtin) }
func (obj Object) String() string          { return stringify(obj) }
func (fn Function) String() string         { return stringify(fn) }
func (el Ellipsis) String() string         { return stringify(el) }
func (not Not) String() string             { return stringify(not) }

func (or Or) String() string {
	s := "(Or"
	for _, node := range or.Nodes {
		s += " "
		s += node.String()
	}
	s += ")"
	return s
}

func isProperList(l List) bool {
	if l.Head == nil && l.Tail == nil {
		return true
	}
	switch tail := l.Tail.(type) {
	case nil:
		return false
	case List:
		return isProperList(tail)
	default:
		return false
	}
}

func (l List) String() string {
	if l.Head == nil && l.Tail == nil {
		return "[]"
	}

	if isProperList(l) {
		// pretty-print the list
		var objs []string
		for l.Head != nil {
			objs = append(objs, l.Head.String())
			l = l.Tail.(List)
		}
		return fmt.Sprintf("[%s]", strings.Join(objs, " "))
	}

	return fmt.Sprintf("%s:%s", l.Head, l.Tail)
}

func (bind Binding) String() string {
	if bind.Node == nil {
		return bind.Name
	}
	return fmt.Sprintf("%s@%s", bind.Name, bind.Node)
}

func (s String) String() string { return fmt.Sprintf("%q", string(s)) }

func (tok Token) String() string {
	return fmt.Sprintf("%q", strings.ToUpper(token.Token(tok).String()))
}

func (Any) String() string { return "_" }

func (AssignStmt) isNode()     {}
func (IndexExpr) isNode()      {}
func (Ident) isNode()          {}
func (ValueSpec) isNode()      {}
func (GenDecl) isNode()        {}
func (BasicLit) isNode()       {}
func (BinaryExpr) isNode()     {}
func (ForStmt) isNode()        {}
func (RangeStmt) isNode()      {}
func (ArrayType) isNode()      {}
func (DeferStmt) isNode()      {}
func (MapType) isNode()        {}
func (ReturnStmt) isNode()     {}
func (SliceExpr) isNode()      {}
func (StarExpr) isNode()       {}
func (UnaryExpr) isNode()      {}
func (SendStmt) isNode()       {}
func (ImportSpec) isNode()     {}
func (SelectStmt) isNode()     {}
func (IfStmt) isNode()         {}
func (IncDecStmt) isNode()     {}
func (GoStmt) isNode()         {}
func (Field) isNode()          {}
func (SelectorExpr) isNode()   {}
func (StructType) isNode()     {}
func (KeyValueExpr) isNode()   {}
func (FuncType) isNode()       {}
func (FuncLit) isNode()        {}
func (FuncDecl) isNode()       {}
func (BranchStmt) isNode()     {}
func (CallExpr) isNode()       {}
func (CaseClause) isNode()     {}
func (ChanType) isNode()       {}
func (CommClause) isNode()     {}
func (CompositeLit) isNode()   {}
func (EmptyStmt) isNode()      {}
func (InterfaceType) isNode()  {}
func (SwitchStmt) isNode()     {}
func (TypeAssertExpr) isNode() {}
func (TypeSpec) isNode()       {}
func (TypeSwitchStmt) isNode() {}
func (Nil) isNode()            {}
func (Builtin) isNode()        {}
func (Object) isNode()         {}
func (Function) isNode()       {}
func (Ellipsis) isNode()       {}
func (Or) isNode()             {}
func (List) isNode()           {}
func (String) isNode()         {}
func (Token) isNode()          {}
func (Any) isNode()            {}
func (Binding) isNode()        {}
func (Not) isNode()            {}
