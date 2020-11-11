package pattern

import (
	"fmt"
	"go/ast"
	"go/token"
	"reflect"
)

type Pattern struct {
	Root Node
	// Relevant contains instances of ast.Node that could potentially
	// initiate a successful match of the pattern.
	Relevant []reflect.Type
}

func MustParse(s string) Pattern {
	p := &Parser{AllowTypeInfo: true}
	pat, err := p.Parse(s)
	if err != nil {
		panic(err)
	}
	return pat
}

func roots(node Node) []reflect.Type {
	switch node := node.(type) {
	case Or:
		var out []reflect.Type
		for _, el := range node.Nodes {
			out = append(out, roots(el)...)
		}
		return out
	case Not:
		return roots(node.Node)
	case Binding:
		return roots(node.Node)
	case Nil, nil:
		// this branch is reached via bindings
		return allTypes
	default:
		Ts, ok := nodeToASTTypes[reflect.TypeOf(node)]
		if !ok {
			panic(fmt.Sprintf("internal error: unhandled type %T", node))
		}
		return Ts
	}
}

var allTypes = []reflect.Type{
	reflect.TypeOf((*ast.RangeStmt)(nil)),
	reflect.TypeOf((*ast.AssignStmt)(nil)),
	reflect.TypeOf((*ast.IndexExpr)(nil)),
	reflect.TypeOf((*ast.Ident)(nil)),
	reflect.TypeOf((*ast.ValueSpec)(nil)),
	reflect.TypeOf((*ast.GenDecl)(nil)),
	reflect.TypeOf((*ast.BinaryExpr)(nil)),
	reflect.TypeOf((*ast.ForStmt)(nil)),
	reflect.TypeOf((*ast.ArrayType)(nil)),
	reflect.TypeOf((*ast.DeferStmt)(nil)),
	reflect.TypeOf((*ast.MapType)(nil)),
	reflect.TypeOf((*ast.ReturnStmt)(nil)),
	reflect.TypeOf((*ast.SliceExpr)(nil)),
	reflect.TypeOf((*ast.StarExpr)(nil)),
	reflect.TypeOf((*ast.UnaryExpr)(nil)),
	reflect.TypeOf((*ast.SendStmt)(nil)),
	reflect.TypeOf((*ast.SelectStmt)(nil)),
	reflect.TypeOf((*ast.ImportSpec)(nil)),
	reflect.TypeOf((*ast.IfStmt)(nil)),
	reflect.TypeOf((*ast.GoStmt)(nil)),
	reflect.TypeOf((*ast.Field)(nil)),
	reflect.TypeOf((*ast.SelectorExpr)(nil)),
	reflect.TypeOf((*ast.StructType)(nil)),
	reflect.TypeOf((*ast.KeyValueExpr)(nil)),
	reflect.TypeOf((*ast.FuncType)(nil)),
	reflect.TypeOf((*ast.FuncLit)(nil)),
	reflect.TypeOf((*ast.FuncDecl)(nil)),
	reflect.TypeOf((*ast.ChanType)(nil)),
	reflect.TypeOf((*ast.CallExpr)(nil)),
	reflect.TypeOf((*ast.CaseClause)(nil)),
	reflect.TypeOf((*ast.CommClause)(nil)),
	reflect.TypeOf((*ast.CompositeLit)(nil)),
	reflect.TypeOf((*ast.EmptyStmt)(nil)),
	reflect.TypeOf((*ast.SwitchStmt)(nil)),
	reflect.TypeOf((*ast.TypeSwitchStmt)(nil)),
	reflect.TypeOf((*ast.TypeAssertExpr)(nil)),
	reflect.TypeOf((*ast.TypeSpec)(nil)),
	reflect.TypeOf((*ast.InterfaceType)(nil)),
	reflect.TypeOf((*ast.BranchStmt)(nil)),
	reflect.TypeOf((*ast.IncDecStmt)(nil)),
	reflect.TypeOf((*ast.BasicLit)(nil)),
}

var nodeToASTTypes = map[reflect.Type][]reflect.Type{
	reflect.TypeOf(String("")):       nil,
	reflect.TypeOf(Token(0)):         nil,
	reflect.TypeOf(List{}):           {reflect.TypeOf((*ast.BlockStmt)(nil)), reflect.TypeOf((*ast.FieldList)(nil))},
	reflect.TypeOf(Builtin{}):        {reflect.TypeOf((*ast.Ident)(nil))},
	reflect.TypeOf(Object{}):         {reflect.TypeOf((*ast.Ident)(nil))},
	reflect.TypeOf(Function{}):       {reflect.TypeOf((*ast.Ident)(nil)), reflect.TypeOf((*ast.SelectorExpr)(nil))},
	reflect.TypeOf(Any{}):            allTypes,
	reflect.TypeOf(RangeStmt{}):      {reflect.TypeOf((*ast.RangeStmt)(nil))},
	reflect.TypeOf(AssignStmt{}):     {reflect.TypeOf((*ast.AssignStmt)(nil))},
	reflect.TypeOf(IndexExpr{}):      {reflect.TypeOf((*ast.IndexExpr)(nil))},
	reflect.TypeOf(Ident{}):          {reflect.TypeOf((*ast.Ident)(nil))},
	reflect.TypeOf(ValueSpec{}):      {reflect.TypeOf((*ast.ValueSpec)(nil))},
	reflect.TypeOf(GenDecl{}):        {reflect.TypeOf((*ast.GenDecl)(nil))},
	reflect.TypeOf(BinaryExpr{}):     {reflect.TypeOf((*ast.BinaryExpr)(nil))},
	reflect.TypeOf(ForStmt{}):        {reflect.TypeOf((*ast.ForStmt)(nil))},
	reflect.TypeOf(ArrayType{}):      {reflect.TypeOf((*ast.ArrayType)(nil))},
	reflect.TypeOf(DeferStmt{}):      {reflect.TypeOf((*ast.DeferStmt)(nil))},
	reflect.TypeOf(MapType{}):        {reflect.TypeOf((*ast.MapType)(nil))},
	reflect.TypeOf(ReturnStmt{}):     {reflect.TypeOf((*ast.ReturnStmt)(nil))},
	reflect.TypeOf(SliceExpr{}):      {reflect.TypeOf((*ast.SliceExpr)(nil))},
	reflect.TypeOf(StarExpr{}):       {reflect.TypeOf((*ast.StarExpr)(nil))},
	reflect.TypeOf(UnaryExpr{}):      {reflect.TypeOf((*ast.UnaryExpr)(nil))},
	reflect.TypeOf(SendStmt{}):       {reflect.TypeOf((*ast.SendStmt)(nil))},
	reflect.TypeOf(SelectStmt{}):     {reflect.TypeOf((*ast.SelectStmt)(nil))},
	reflect.TypeOf(ImportSpec{}):     {reflect.TypeOf((*ast.ImportSpec)(nil))},
	reflect.TypeOf(IfStmt{}):         {reflect.TypeOf((*ast.IfStmt)(nil))},
	reflect.TypeOf(GoStmt{}):         {reflect.TypeOf((*ast.GoStmt)(nil))},
	reflect.TypeOf(Field{}):          {reflect.TypeOf((*ast.Field)(nil))},
	reflect.TypeOf(SelectorExpr{}):   {reflect.TypeOf((*ast.SelectorExpr)(nil))},
	reflect.TypeOf(StructType{}):     {reflect.TypeOf((*ast.StructType)(nil))},
	reflect.TypeOf(KeyValueExpr{}):   {reflect.TypeOf((*ast.KeyValueExpr)(nil))},
	reflect.TypeOf(FuncType{}):       {reflect.TypeOf((*ast.FuncType)(nil))},
	reflect.TypeOf(FuncLit{}):        {reflect.TypeOf((*ast.FuncLit)(nil))},
	reflect.TypeOf(FuncDecl{}):       {reflect.TypeOf((*ast.FuncDecl)(nil))},
	reflect.TypeOf(ChanType{}):       {reflect.TypeOf((*ast.ChanType)(nil))},
	reflect.TypeOf(CallExpr{}):       {reflect.TypeOf((*ast.CallExpr)(nil))},
	reflect.TypeOf(CaseClause{}):     {reflect.TypeOf((*ast.CaseClause)(nil))},
	reflect.TypeOf(CommClause{}):     {reflect.TypeOf((*ast.CommClause)(nil))},
	reflect.TypeOf(CompositeLit{}):   {reflect.TypeOf((*ast.CompositeLit)(nil))},
	reflect.TypeOf(EmptyStmt{}):      {reflect.TypeOf((*ast.EmptyStmt)(nil))},
	reflect.TypeOf(SwitchStmt{}):     {reflect.TypeOf((*ast.SwitchStmt)(nil))},
	reflect.TypeOf(TypeSwitchStmt{}): {reflect.TypeOf((*ast.TypeSwitchStmt)(nil))},
	reflect.TypeOf(TypeAssertExpr{}): {reflect.TypeOf((*ast.TypeAssertExpr)(nil))},
	reflect.TypeOf(TypeSpec{}):       {reflect.TypeOf((*ast.TypeSpec)(nil))},
	reflect.TypeOf(InterfaceType{}):  {reflect.TypeOf((*ast.InterfaceType)(nil))},
	reflect.TypeOf(BranchStmt{}):     {reflect.TypeOf((*ast.BranchStmt)(nil))},
	reflect.TypeOf(IncDecStmt{}):     {reflect.TypeOf((*ast.IncDecStmt)(nil))},
	reflect.TypeOf(BasicLit{}):       {reflect.TypeOf((*ast.BasicLit)(nil))},
}

var requiresTypeInfo = map[string]bool{
	"Function": true,
	"Builtin":  true,
	"Object":   true,
}

type Parser struct {
	// Allow nodes that rely on type information
	AllowTypeInfo bool

	lex   *lexer
	cur   item
	last  *item
	items chan item
}

func (p *Parser) Parse(s string) (Pattern, error) {
	p.cur = item{}
	p.last = nil
	p.items = nil

	fset := token.NewFileSet()
	p.lex = &lexer{
		f:     fset.AddFile("<input>", -1, len(s)),
		input: s,
		items: make(chan item),
	}
	go p.lex.run()
	p.items = p.lex.items
	root, err := p.node()
	if err != nil {
		// drain lexer if parsing failed
		for range p.lex.items {
		}
		return Pattern{}, err
	}
	if item := <-p.lex.items; item.typ != itemEOF {
		return Pattern{}, fmt.Errorf("unexpected token %s after end of pattern", item.typ)
	}
	return Pattern{
		Root:     root,
		Relevant: roots(root),
	}, nil
}

func (p *Parser) next() item {
	if p.last != nil {
		n := *p.last
		p.last = nil
		return n
	}
	var ok bool
	p.cur, ok = <-p.items
	if !ok {
		p.cur = item{typ: eof}
	}
	return p.cur
}

func (p *Parser) rewind() {
	p.last = &p.cur
}

func (p *Parser) peek() item {
	n := p.next()
	p.rewind()
	return n
}

func (p *Parser) accept(typ itemType) (item, bool) {
	n := p.next()
	if n.typ == typ {
		return n, true
	}
	p.rewind()
	return item{}, false
}

func (p *Parser) unexpectedToken(valid string) error {
	if p.cur.typ == itemError {
		return fmt.Errorf("error lexing input: %s", p.cur.val)
	}
	var got string
	switch p.cur.typ {
	case itemTypeName, itemVariable, itemString:
		got = p.cur.val
	default:
		got = "'" + p.cur.typ.String() + "'"
	}

	pos := p.lex.f.Position(token.Pos(p.cur.pos))
	return fmt.Errorf("%s: expected %s, found %s", pos, valid, got)
}

func (p *Parser) node() (Node, error) {
	if _, ok := p.accept(itemLeftParen); !ok {
		return nil, p.unexpectedToken("'('")
	}
	typ, ok := p.accept(itemTypeName)
	if !ok {
		return nil, p.unexpectedToken("Node type")
	}

	var objs []Node
	for {
		if _, ok := p.accept(itemRightParen); ok {
			break
		} else {
			p.rewind()
			obj, err := p.object()
			if err != nil {
				return nil, err
			}
			objs = append(objs, obj)
		}
	}

	return p.populateNode(typ.val, objs)
}

func populateNode(typ string, objs []Node, allowTypeInfo bool) (Node, error) {
	T, ok := structNodes[typ]
	if !ok {
		return nil, fmt.Errorf("unknown node %s", typ)
	}

	if !allowTypeInfo && requiresTypeInfo[typ] {
		return nil, fmt.Errorf("Node %s requires type information", typ)
	}

	pv := reflect.New(T)
	v := pv.Elem()

	if v.NumField() == 1 {
		f := v.Field(0)
		if f.Type().Kind() == reflect.Slice {
			// Variadic node
			f.Set(reflect.AppendSlice(f, reflect.ValueOf(objs)))
			return v.Interface().(Node), nil
		}
	}
	if len(objs) != v.NumField() {
		return nil, fmt.Errorf("tried to initialize node %s with %d values, expected %d", typ, len(objs), v.NumField())
	}
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		if f.Kind() == reflect.String {
			if obj, ok := objs[i].(String); ok {
				f.Set(reflect.ValueOf(string(obj)))
			} else {
				return nil, fmt.Errorf("first argument of (Binding name node) must be string, but got %s", objs[i])
			}
		} else {
			f.Set(reflect.ValueOf(objs[i]))
		}
	}
	return v.Interface().(Node), nil
}

func (p *Parser) populateNode(typ string, objs []Node) (Node, error) {
	return populateNode(typ, objs, p.AllowTypeInfo)
}

var structNodes = map[string]reflect.Type{
	"Any":            reflect.TypeOf(Any{}),
	"Ellipsis":       reflect.TypeOf(Ellipsis{}),
	"List":           reflect.TypeOf(List{}),
	"Binding":        reflect.TypeOf(Binding{}),
	"RangeStmt":      reflect.TypeOf(RangeStmt{}),
	"AssignStmt":     reflect.TypeOf(AssignStmt{}),
	"IndexExpr":      reflect.TypeOf(IndexExpr{}),
	"Ident":          reflect.TypeOf(Ident{}),
	"Builtin":        reflect.TypeOf(Builtin{}),
	"ValueSpec":      reflect.TypeOf(ValueSpec{}),
	"GenDecl":        reflect.TypeOf(GenDecl{}),
	"BinaryExpr":     reflect.TypeOf(BinaryExpr{}),
	"ForStmt":        reflect.TypeOf(ForStmt{}),
	"ArrayType":      reflect.TypeOf(ArrayType{}),
	"DeferStmt":      reflect.TypeOf(DeferStmt{}),
	"MapType":        reflect.TypeOf(MapType{}),
	"ReturnStmt":     reflect.TypeOf(ReturnStmt{}),
	"SliceExpr":      reflect.TypeOf(SliceExpr{}),
	"StarExpr":       reflect.TypeOf(StarExpr{}),
	"UnaryExpr":      reflect.TypeOf(UnaryExpr{}),
	"SendStmt":       reflect.TypeOf(SendStmt{}),
	"SelectStmt":     reflect.TypeOf(SelectStmt{}),
	"ImportSpec":     reflect.TypeOf(ImportSpec{}),
	"IfStmt":         reflect.TypeOf(IfStmt{}),
	"GoStmt":         reflect.TypeOf(GoStmt{}),
	"Field":          reflect.TypeOf(Field{}),
	"SelectorExpr":   reflect.TypeOf(SelectorExpr{}),
	"StructType":     reflect.TypeOf(StructType{}),
	"KeyValueExpr":   reflect.TypeOf(KeyValueExpr{}),
	"FuncType":       reflect.TypeOf(FuncType{}),
	"FuncLit":        reflect.TypeOf(FuncLit{}),
	"FuncDecl":       reflect.TypeOf(FuncDecl{}),
	"ChanType":       reflect.TypeOf(ChanType{}),
	"CallExpr":       reflect.TypeOf(CallExpr{}),
	"CaseClause":     reflect.TypeOf(CaseClause{}),
	"CommClause":     reflect.TypeOf(CommClause{}),
	"CompositeLit":   reflect.TypeOf(CompositeLit{}),
	"EmptyStmt":      reflect.TypeOf(EmptyStmt{}),
	"SwitchStmt":     reflect.TypeOf(SwitchStmt{}),
	"TypeSwitchStmt": reflect.TypeOf(TypeSwitchStmt{}),
	"TypeAssertExpr": reflect.TypeOf(TypeAssertExpr{}),
	"TypeSpec":       reflect.TypeOf(TypeSpec{}),
	"InterfaceType":  reflect.TypeOf(InterfaceType{}),
	"BranchStmt":     reflect.TypeOf(BranchStmt{}),
	"IncDecStmt":     reflect.TypeOf(IncDecStmt{}),
	"BasicLit":       reflect.TypeOf(BasicLit{}),
	"Object":         reflect.TypeOf(Object{}),
	"Function":       reflect.TypeOf(Function{}),
	"Or":             reflect.TypeOf(Or{}),
	"Not":            reflect.TypeOf(Not{}),
}

func (p *Parser) object() (Node, error) {
	n := p.next()
	switch n.typ {
	case itemLeftParen:
		p.rewind()
		node, err := p.node()
		if err != nil {
			return node, err
		}
		if p.peek().typ == itemColon {
			p.next()
			tail, err := p.object()
			if err != nil {
				return node, err
			}
			return List{Head: node, Tail: tail}, nil
		}
		return node, nil
	case itemLeftBracket:
		p.rewind()
		return p.array()
	case itemVariable:
		v := n
		if v.val == "nil" {
			return Nil{}, nil
		}
		var b Binding
		if _, ok := p.accept(itemAt); ok {
			o, err := p.node()
			if err != nil {
				return nil, err
			}
			b = Binding{
				Name: v.val,
				Node: o,
			}
		} else {
			p.rewind()
			b = Binding{Name: v.val}
		}
		if p.peek().typ == itemColon {
			p.next()
			tail, err := p.object()
			if err != nil {
				return b, err
			}
			return List{Head: b, Tail: tail}, nil
		}
		return b, nil
	case itemBlank:
		return Any{}, nil
	case itemString:
		return String(n.val), nil
	default:
		return nil, p.unexpectedToken("object")
	}
}

func (p *Parser) array() (Node, error) {
	if _, ok := p.accept(itemLeftBracket); !ok {
		return nil, p.unexpectedToken("'['")
	}

	var objs []Node
	for {
		if _, ok := p.accept(itemRightBracket); ok {
			break
		} else {
			p.rewind()
			obj, err := p.object()
			if err != nil {
				return nil, err
			}
			objs = append(objs, obj)
		}
	}

	tail := List{}
	for i := len(objs) - 1; i >= 0; i-- {
		l := List{
			Head: objs[i],
			Tail: tail,
		}
		tail = l
	}
	return tail, nil
}

/*
Node ::= itemLeftParen itemTypeName Object* itemRightParen
Object ::= Node | Array | Binding | itemVariable | itemBlank | itemString
Array := itemLeftBracket Object* itemRightBracket
Array := Object itemColon Object
Binding ::= itemVariable itemAt Node
*/
