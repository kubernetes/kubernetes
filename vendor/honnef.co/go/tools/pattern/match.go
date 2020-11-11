package pattern

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"

	"honnef.co/go/tools/lint"
)

var tokensByString = map[string]Token{
	"INT":    Token(token.INT),
	"FLOAT":  Token(token.FLOAT),
	"IMAG":   Token(token.IMAG),
	"CHAR":   Token(token.CHAR),
	"STRING": Token(token.STRING),
	"+":      Token(token.ADD),
	"-":      Token(token.SUB),
	"*":      Token(token.MUL),
	"/":      Token(token.QUO),
	"%":      Token(token.REM),
	"&":      Token(token.AND),
	"|":      Token(token.OR),
	"^":      Token(token.XOR),
	"<<":     Token(token.SHL),
	">>":     Token(token.SHR),
	"&^":     Token(token.AND_NOT),
	"+=":     Token(token.ADD_ASSIGN),
	"-=":     Token(token.SUB_ASSIGN),
	"*=":     Token(token.MUL_ASSIGN),
	"/=":     Token(token.QUO_ASSIGN),
	"%=":     Token(token.REM_ASSIGN),
	"&=":     Token(token.AND_ASSIGN),
	"|=":     Token(token.OR_ASSIGN),
	"^=":     Token(token.XOR_ASSIGN),
	"<<=":    Token(token.SHL_ASSIGN),
	">>=":    Token(token.SHR_ASSIGN),
	"&^=":    Token(token.AND_NOT_ASSIGN),
	"&&":     Token(token.LAND),
	"||":     Token(token.LOR),
	"<-":     Token(token.ARROW),
	"++":     Token(token.INC),
	"--":     Token(token.DEC),
	"==":     Token(token.EQL),
	"<":      Token(token.LSS),
	">":      Token(token.GTR),
	"=":      Token(token.ASSIGN),
	"!":      Token(token.NOT),
	"!=":     Token(token.NEQ),
	"<=":     Token(token.LEQ),
	">=":     Token(token.GEQ),
	":=":     Token(token.DEFINE),
	"...":    Token(token.ELLIPSIS),
	"IMPORT": Token(token.IMPORT),
	"VAR":    Token(token.VAR),
	"TYPE":   Token(token.TYPE),
	"CONST":  Token(token.CONST),
}

func maybeToken(node Node) (Node, bool) {
	if node, ok := node.(String); ok {
		if tok, ok := tokensByString[string(node)]; ok {
			return tok, true
		}
		return node, false
	}
	return node, false
}

func isNil(v interface{}) bool {
	if v == nil {
		return true
	}
	if _, ok := v.(Nil); ok {
		return true
	}
	return false
}

type matcher interface {
	Match(*Matcher, interface{}) (interface{}, bool)
}

type State = map[string]interface{}

type Matcher struct {
	TypesInfo *types.Info
	State     State
}

func (m *Matcher) fork() *Matcher {
	state := make(State, len(m.State))
	for k, v := range m.State {
		state[k] = v
	}
	return &Matcher{
		TypesInfo: m.TypesInfo,
		State:     state,
	}
}

func (m *Matcher) merge(mc *Matcher) {
	m.State = mc.State
}

func (m *Matcher) Match(a Node, b ast.Node) bool {
	m.State = State{}
	_, ok := match(m, a, b)
	return ok
}

func Match(a Node, b ast.Node) (*Matcher, bool) {
	m := &Matcher{}
	ret := m.Match(a, b)
	return m, ret
}

// Match two items, which may be (Node, AST) or (AST, AST)
func match(m *Matcher, l, r interface{}) (interface{}, bool) {
	if _, ok := r.(Node); ok {
		panic("Node mustn't be on right side of match")
	}

	switch l := l.(type) {
	case *ast.ParenExpr:
		return match(m, l.X, r)
	case *ast.ExprStmt:
		return match(m, l.X, r)
	case *ast.DeclStmt:
		return match(m, l.Decl, r)
	case *ast.LabeledStmt:
		return match(m, l.Stmt, r)
	case *ast.BlockStmt:
		return match(m, l.List, r)
	case *ast.FieldList:
		return match(m, l.List, r)
	}

	switch r := r.(type) {
	case *ast.ParenExpr:
		return match(m, l, r.X)
	case *ast.ExprStmt:
		return match(m, l, r.X)
	case *ast.DeclStmt:
		return match(m, l, r.Decl)
	case *ast.LabeledStmt:
		return match(m, l, r.Stmt)
	case *ast.BlockStmt:
		if r == nil {
			return match(m, l, nil)
		}
		return match(m, l, r.List)
	case *ast.FieldList:
		if r == nil {
			return match(m, l, nil)
		}
		return match(m, l, r.List)
	case *ast.BasicLit:
		if r == nil {
			return match(m, l, nil)
		}
	}

	if l, ok := l.(matcher); ok {
		return l.Match(m, r)
	}

	if l, ok := l.(Node); ok {
		// Matching of pattern with concrete value
		return matchNodeAST(m, l, r)
	}

	if l == nil || r == nil {
		return nil, l == r
	}

	{
		ln, ok1 := l.(ast.Node)
		rn, ok2 := r.(ast.Node)
		if ok1 && ok2 {
			return matchAST(m, ln, rn)
		}
	}

	{
		obj, ok := l.(types.Object)
		if ok {
			switch r := r.(type) {
			case *ast.Ident:
				return obj, obj == m.TypesInfo.ObjectOf(r)
			case *ast.SelectorExpr:
				return obj, obj == m.TypesInfo.ObjectOf(r.Sel)
			default:
				return obj, false
			}
		}
	}

	{
		ln, ok1 := l.([]ast.Expr)
		rn, ok2 := r.([]ast.Expr)
		if ok1 || ok2 {
			if ok1 && !ok2 {
				rn = []ast.Expr{r.(ast.Expr)}
			} else if !ok1 && ok2 {
				ln = []ast.Expr{l.(ast.Expr)}
			}

			if len(ln) != len(rn) {
				return nil, false
			}
			for i, ll := range ln {
				if _, ok := match(m, ll, rn[i]); !ok {
					return nil, false
				}
			}
			return r, true
		}
	}

	{
		ln, ok1 := l.([]ast.Stmt)
		rn, ok2 := r.([]ast.Stmt)
		if ok1 || ok2 {
			if ok1 && !ok2 {
				rn = []ast.Stmt{r.(ast.Stmt)}
			} else if !ok1 && ok2 {
				ln = []ast.Stmt{l.(ast.Stmt)}
			}

			if len(ln) != len(rn) {
				return nil, false
			}
			for i, ll := range ln {
				if _, ok := match(m, ll, rn[i]); !ok {
					return nil, false
				}
			}
			return r, true
		}
	}

	panic(fmt.Sprintf("unsupported comparison: %T and %T", l, r))
}

// Match a Node with an AST node
func matchNodeAST(m *Matcher, a Node, b interface{}) (interface{}, bool) {
	switch b := b.(type) {
	case []ast.Stmt:
		// 'a' is not a List or we'd be using its Match
		// implementation.

		if len(b) != 1 {
			return nil, false
		}
		return match(m, a, b[0])
	case []ast.Expr:
		// 'a' is not a List or we'd be using its Match
		// implementation.

		if len(b) != 1 {
			return nil, false
		}
		return match(m, a, b[0])
	case ast.Node:
		ra := reflect.ValueOf(a)
		rb := reflect.ValueOf(b).Elem()

		if ra.Type().Name() != rb.Type().Name() {
			return nil, false
		}

		for i := 0; i < ra.NumField(); i++ {
			af := ra.Field(i)
			fieldName := ra.Type().Field(i).Name
			bf := rb.FieldByName(fieldName)
			if (bf == reflect.Value{}) {
				panic(fmt.Sprintf("internal error: could not find field %s in type %t when comparing with %T", fieldName, b, a))
			}
			ai := af.Interface()
			bi := bf.Interface()
			if ai == nil {
				return b, bi == nil
			}
			if _, ok := match(m, ai.(Node), bi); !ok {
				return b, false
			}
		}
		return b, true
	case nil:
		return nil, a == Nil{}
	default:
		panic(fmt.Sprintf("unhandled type %T", b))
	}
}

// Match two AST nodes
func matchAST(m *Matcher, a, b ast.Node) (interface{}, bool) {
	ra := reflect.ValueOf(a)
	rb := reflect.ValueOf(b)

	if ra.Type() != rb.Type() {
		return nil, false
	}
	if ra.IsNil() || rb.IsNil() {
		return rb, ra.IsNil() == rb.IsNil()
	}

	ra = ra.Elem()
	rb = rb.Elem()
	for i := 0; i < ra.NumField(); i++ {
		af := ra.Field(i)
		bf := rb.Field(i)
		if af.Type() == rtTokPos || af.Type() == rtObject || af.Type() == rtCommentGroup {
			continue
		}

		switch af.Kind() {
		case reflect.Slice:
			if af.Len() != bf.Len() {
				return nil, false
			}
			for j := 0; j < af.Len(); j++ {
				if _, ok := match(m, af.Index(j).Interface().(ast.Node), bf.Index(j).Interface().(ast.Node)); !ok {
					return nil, false
				}
			}
		case reflect.String:
			if af.String() != bf.String() {
				return nil, false
			}
		case reflect.Int:
			if af.Int() != bf.Int() {
				return nil, false
			}
		case reflect.Bool:
			if af.Bool() != bf.Bool() {
				return nil, false
			}
		case reflect.Ptr, reflect.Interface:
			if _, ok := match(m, af.Interface(), bf.Interface()); !ok {
				return nil, false
			}
		default:
			panic(fmt.Sprintf("internal error: unhandled kind %s (%T)", af.Kind(), af.Interface()))
		}
	}
	return b, true
}

func (b Binding) Match(m *Matcher, node interface{}) (interface{}, bool) {
	if isNil(b.Node) {
		v, ok := m.State[b.Name]
		if ok {
			// Recall value
			return match(m, v, node)
		}
		// Matching anything
		b.Node = Any{}
	}

	// Store value
	if _, ok := m.State[b.Name]; ok {
		panic(fmt.Sprintf("binding already created: %s", b.Name))
	}
	new, ret := match(m, b.Node, node)
	if ret {
		m.State[b.Name] = new
	}
	return new, ret
}

func (Any) Match(m *Matcher, node interface{}) (interface{}, bool) {
	return node, true
}

func (l List) Match(m *Matcher, node interface{}) (interface{}, bool) {
	v := reflect.ValueOf(node)
	if v.Kind() == reflect.Slice {
		if isNil(l.Head) {
			return node, v.Len() == 0
		}
		if v.Len() == 0 {
			return nil, false
		}
		// OPT(dh): don't check the entire tail if head didn't match
		_, ok1 := match(m, l.Head, v.Index(0).Interface())
		_, ok2 := match(m, l.Tail, v.Slice(1, v.Len()).Interface())
		return node, ok1 && ok2
	}
	// Our empty list does not equal an untyped Go nil. This way, we can
	// tell apart an if with no else and an if with an empty else.
	return nil, false
}

func (s String) Match(m *Matcher, node interface{}) (interface{}, bool) {
	switch o := node.(type) {
	case token.Token:
		if tok, ok := maybeToken(s); ok {
			return match(m, tok, node)
		}
		return nil, false
	case string:
		return o, string(s) == o
	default:
		return nil, false
	}
}

func (tok Token) Match(m *Matcher, node interface{}) (interface{}, bool) {
	o, ok := node.(token.Token)
	if !ok {
		return nil, false
	}
	return o, token.Token(tok) == o
}

func (Nil) Match(m *Matcher, node interface{}) (interface{}, bool) {
	return nil, isNil(node)
}

func (builtin Builtin) Match(m *Matcher, node interface{}) (interface{}, bool) {
	ident, ok := node.(*ast.Ident)
	if !ok {
		return nil, false
	}
	obj := m.TypesInfo.ObjectOf(ident)
	if obj != types.Universe.Lookup(ident.Name) {
		return nil, false
	}
	return match(m, builtin.Name, ident.Name)
}

func (obj Object) Match(m *Matcher, node interface{}) (interface{}, bool) {
	ident, ok := node.(*ast.Ident)
	if !ok {
		return nil, false
	}

	id := m.TypesInfo.ObjectOf(ident)
	_, ok = match(m, obj.Name, ident.Name)
	return id, ok
}

func (fn Function) Match(m *Matcher, node interface{}) (interface{}, bool) {
	var name string
	var obj types.Object
	switch node := node.(type) {
	case *ast.Ident:
		obj = m.TypesInfo.ObjectOf(node)
		switch obj := obj.(type) {
		case *types.Func:
			name = lint.FuncName(obj)
		case *types.Builtin:
			name = obj.Name()
		default:
			return nil, false
		}
	case *ast.SelectorExpr:
		var ok bool
		obj, ok = m.TypesInfo.ObjectOf(node.Sel).(*types.Func)
		if !ok {
			return nil, false
		}
		name = lint.FuncName(obj.(*types.Func))
	default:
		return nil, false
	}
	_, ok := match(m, fn.Name, name)
	return obj, ok
}

func (or Or) Match(m *Matcher, node interface{}) (interface{}, bool) {
	for _, opt := range or.Nodes {
		mc := m.fork()
		if ret, ok := match(mc, opt, node); ok {
			m.merge(mc)
			return ret, true
		}
	}
	return nil, false
}

func (not Not) Match(m *Matcher, node interface{}) (interface{}, bool) {
	_, ok := match(m, not.Node, node)
	if ok {
		return nil, false
	}
	return node, true
}

var (
	// Types of fields in go/ast structs that we want to skip
	rtTokPos       = reflect.TypeOf(token.Pos(0))
	rtObject       = reflect.TypeOf((*ast.Object)(nil))
	rtCommentGroup = reflect.TypeOf((*ast.CommentGroup)(nil))
)

var (
	_ matcher = Binding{}
	_ matcher = Any{}
	_ matcher = List{}
	_ matcher = String("")
	_ matcher = Token(0)
	_ matcher = Nil{}
	_ matcher = Builtin{}
	_ matcher = Object{}
	_ matcher = Function{}
	_ matcher = Or{}
	_ matcher = Not{}
)
