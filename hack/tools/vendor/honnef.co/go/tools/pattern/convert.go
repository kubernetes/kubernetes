package pattern

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
)

var astTypes = map[string]reflect.Type{
	"Ellipsis":       reflect.TypeOf(ast.Ellipsis{}),
	"RangeStmt":      reflect.TypeOf(ast.RangeStmt{}),
	"AssignStmt":     reflect.TypeOf(ast.AssignStmt{}),
	"IndexExpr":      reflect.TypeOf(ast.IndexExpr{}),
	"Ident":          reflect.TypeOf(ast.Ident{}),
	"ValueSpec":      reflect.TypeOf(ast.ValueSpec{}),
	"GenDecl":        reflect.TypeOf(ast.GenDecl{}),
	"BinaryExpr":     reflect.TypeOf(ast.BinaryExpr{}),
	"ForStmt":        reflect.TypeOf(ast.ForStmt{}),
	"ArrayType":      reflect.TypeOf(ast.ArrayType{}),
	"DeferStmt":      reflect.TypeOf(ast.DeferStmt{}),
	"MapType":        reflect.TypeOf(ast.MapType{}),
	"ReturnStmt":     reflect.TypeOf(ast.ReturnStmt{}),
	"SliceExpr":      reflect.TypeOf(ast.SliceExpr{}),
	"StarExpr":       reflect.TypeOf(ast.StarExpr{}),
	"UnaryExpr":      reflect.TypeOf(ast.UnaryExpr{}),
	"SendStmt":       reflect.TypeOf(ast.SendStmt{}),
	"SelectStmt":     reflect.TypeOf(ast.SelectStmt{}),
	"ImportSpec":     reflect.TypeOf(ast.ImportSpec{}),
	"IfStmt":         reflect.TypeOf(ast.IfStmt{}),
	"GoStmt":         reflect.TypeOf(ast.GoStmt{}),
	"Field":          reflect.TypeOf(ast.Field{}),
	"SelectorExpr":   reflect.TypeOf(ast.SelectorExpr{}),
	"StructType":     reflect.TypeOf(ast.StructType{}),
	"KeyValueExpr":   reflect.TypeOf(ast.KeyValueExpr{}),
	"FuncType":       reflect.TypeOf(ast.FuncType{}),
	"FuncLit":        reflect.TypeOf(ast.FuncLit{}),
	"FuncDecl":       reflect.TypeOf(ast.FuncDecl{}),
	"ChanType":       reflect.TypeOf(ast.ChanType{}),
	"CallExpr":       reflect.TypeOf(ast.CallExpr{}),
	"CaseClause":     reflect.TypeOf(ast.CaseClause{}),
	"CommClause":     reflect.TypeOf(ast.CommClause{}),
	"CompositeLit":   reflect.TypeOf(ast.CompositeLit{}),
	"EmptyStmt":      reflect.TypeOf(ast.EmptyStmt{}),
	"SwitchStmt":     reflect.TypeOf(ast.SwitchStmt{}),
	"TypeSwitchStmt": reflect.TypeOf(ast.TypeSwitchStmt{}),
	"TypeAssertExpr": reflect.TypeOf(ast.TypeAssertExpr{}),
	"TypeSpec":       reflect.TypeOf(ast.TypeSpec{}),
	"InterfaceType":  reflect.TypeOf(ast.InterfaceType{}),
	"BranchStmt":     reflect.TypeOf(ast.BranchStmt{}),
	"IncDecStmt":     reflect.TypeOf(ast.IncDecStmt{}),
	"BasicLit":       reflect.TypeOf(ast.BasicLit{}),
}

func ASTToNode(node interface{}) Node {
	switch node := node.(type) {
	case *ast.File:
		panic("cannot convert *ast.File to Node")
	case nil:
		return Nil{}
	case string:
		return String(node)
	case token.Token:
		return Token(node)
	case *ast.ExprStmt:
		return ASTToNode(node.X)
	case *ast.BlockStmt:
		if node == nil {
			return Nil{}
		}
		return ASTToNode(node.List)
	case *ast.FieldList:
		if node == nil {
			return Nil{}
		}
		return ASTToNode(node.List)
	case *ast.BasicLit:
		if node == nil {
			return Nil{}
		}
	case *ast.ParenExpr:
		return ASTToNode(node.X)
	}

	if node, ok := node.(ast.Node); ok {
		name := reflect.TypeOf(node).Elem().Name()
		T, ok := structNodes[name]
		if !ok {
			panic(fmt.Sprintf("internal error: unhandled type %T", node))
		}

		if reflect.ValueOf(node).IsNil() {
			return Nil{}
		}
		v := reflect.ValueOf(node).Elem()
		objs := make([]Node, T.NumField())
		for i := 0; i < T.NumField(); i++ {
			f := v.FieldByName(T.Field(i).Name)
			objs[i] = ASTToNode(f.Interface())
		}

		n, err := populateNode(name, objs, false)
		if err != nil {
			panic(fmt.Sprintf("internal error: %s", err))
		}
		return n
	}

	s := reflect.ValueOf(node)
	if s.Kind() == reflect.Slice {
		if s.Len() == 0 {
			return List{}
		}
		if s.Len() == 1 {
			return ASTToNode(s.Index(0).Interface())
		}

		tail := List{}
		for i := s.Len() - 1; i >= 0; i-- {
			head := ASTToNode(s.Index(i).Interface())
			l := List{
				Head: head,
				Tail: tail,
			}
			tail = l
		}
		return tail
	}

	panic(fmt.Sprintf("internal error: unhandled type %T", node))
}

func NodeToAST(node Node, state State) interface{} {
	switch node := node.(type) {
	case Binding:
		v, ok := state[node.Name]
		if !ok {
			// really we want to return an error here
			panic("XXX")
		}
		switch v := v.(type) {
		case types.Object:
			return &ast.Ident{Name: v.Name()}
		default:
			return v
		}
	case Builtin, Any, Object, Function, Not, Or:
		panic("XXX")
	case List:
		if (node == List{}) {
			return []ast.Node{}
		}
		x := []ast.Node{NodeToAST(node.Head, state).(ast.Node)}
		x = append(x, NodeToAST(node.Tail, state).([]ast.Node)...)
		return x
	case Token:
		return token.Token(node)
	case String:
		return string(node)
	case Nil:
		return nil
	}

	name := reflect.TypeOf(node).Name()
	T, ok := astTypes[name]
	if !ok {
		panic(fmt.Sprintf("internal error: unhandled type %T", node))
	}
	v := reflect.ValueOf(node)
	out := reflect.New(T)
	for i := 0; i < T.NumField(); i++ {
		fNode := v.FieldByName(T.Field(i).Name)
		if (fNode == reflect.Value{}) {
			continue
		}
		fAST := out.Elem().FieldByName(T.Field(i).Name)
		switch fAST.Type().Kind() {
		case reflect.Slice:
			c := reflect.ValueOf(NodeToAST(fNode.Interface().(Node), state))
			if c.Kind() != reflect.Slice {
				// it's a single node in the pattern, we have to wrap
				// it in a slice
				slice := reflect.MakeSlice(fAST.Type(), 1, 1)
				slice.Index(0).Set(c)
				c = slice
			}
			switch fAST.Interface().(type) {
			case []ast.Node:
				switch cc := c.Interface().(type) {
				case []ast.Node:
					fAST.Set(c)
				case []ast.Expr:
					var slice []ast.Node
					for _, el := range cc {
						slice = append(slice, el)
					}
					fAST.Set(reflect.ValueOf(slice))
				default:
					panic("XXX")
				}
			case []ast.Expr:
				switch cc := c.Interface().(type) {
				case []ast.Node:
					var slice []ast.Expr
					for _, el := range cc {
						slice = append(slice, el.(ast.Expr))
					}
					fAST.Set(reflect.ValueOf(slice))
				case []ast.Expr:
					fAST.Set(c)
				default:
					panic("XXX")
				}
			default:
				panic("XXX")
			}
		case reflect.Int:
			c := reflect.ValueOf(NodeToAST(fNode.Interface().(Node), state))
			switch c.Kind() {
			case reflect.String:
				tok, ok := tokensByString[c.Interface().(string)]
				if !ok {
					// really we want to return an error here
					panic("XXX")
				}
				fAST.SetInt(int64(tok))
			case reflect.Int:
				fAST.Set(c)
			default:
				panic(fmt.Sprintf("internal error: unexpected kind %s", c.Kind()))
			}
		default:
			r := NodeToAST(fNode.Interface().(Node), state)
			if r != nil {
				fAST.Set(reflect.ValueOf(r))
			}
		}
	}

	return out.Interface().(ast.Node)
}
