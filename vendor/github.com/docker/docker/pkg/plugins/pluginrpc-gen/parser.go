package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"strings"
)

var ErrBadReturn = errors.New("found return arg with no name: all args must be named")

type ErrUnexpectedType struct {
	expected string
	actual   interface{}
}

func (e ErrUnexpectedType) Error() string {
	return fmt.Sprintf("got wrong type expecting %s, got: %v", e.expected, reflect.TypeOf(e.actual))
}

type parsedPkg struct {
	Name      string
	Functions []function
}

type function struct {
	Name    string
	Args    []arg
	Returns []arg
	Doc     string
}

type arg struct {
	Name    string
	ArgType string
}

func (a *arg) String() string {
	return strings.ToLower(a.Name) + " " + strings.ToLower(a.ArgType)
}

// Parses the given file for an interface definition with the given name
func Parse(filePath string, objName string) (*parsedPkg, error) {
	fs := token.NewFileSet()
	pkg, err := parser.ParseFile(fs, filePath, nil, parser.AllErrors)
	if err != nil {
		return nil, err
	}
	p := &parsedPkg{}
	p.Name = pkg.Name.Name
	obj, exists := pkg.Scope.Objects[objName]
	if !exists {
		return nil, fmt.Errorf("could not find object %s in %s", objName, filePath)
	}
	if obj.Kind != ast.Typ {
		return nil, fmt.Errorf("exected type, got %s", obj.Kind)
	}
	spec, ok := obj.Decl.(*ast.TypeSpec)
	if !ok {
		return nil, ErrUnexpectedType{"*ast.TypeSpec", obj.Decl}
	}
	iface, ok := spec.Type.(*ast.InterfaceType)
	if !ok {
		return nil, ErrUnexpectedType{"*ast.InterfaceType", spec.Type}
	}

	p.Functions, err = parseInterface(iface)
	if err != nil {
		return nil, err
	}

	return p, nil
}

func parseInterface(iface *ast.InterfaceType) ([]function, error) {
	var functions []function
	for _, field := range iface.Methods.List {
		switch f := field.Type.(type) {
		case *ast.FuncType:
			method, err := parseFunc(field)
			if err != nil {
				return nil, err
			}
			if method == nil {
				continue
			}
			functions = append(functions, *method)
		case *ast.Ident:
			spec, ok := f.Obj.Decl.(*ast.TypeSpec)
			if !ok {
				return nil, ErrUnexpectedType{"*ast.TypeSpec", f.Obj.Decl}
			}
			iface, ok := spec.Type.(*ast.InterfaceType)
			if !ok {
				return nil, ErrUnexpectedType{"*ast.TypeSpec", spec.Type}
			}
			funcs, err := parseInterface(iface)
			if err != nil {
				fmt.Println(err)
				continue
			}
			functions = append(functions, funcs...)
		default:
			return nil, ErrUnexpectedType{"*astFuncType or *ast.Ident", f}
		}
	}
	return functions, nil
}

func parseFunc(field *ast.Field) (*function, error) {
	f := field.Type.(*ast.FuncType)
	method := &function{Name: field.Names[0].Name}
	if _, exists := skipFuncs[method.Name]; exists {
		fmt.Println("skipping:", method.Name)
		return nil, nil
	}
	if f.Params != nil {
		args, err := parseArgs(f.Params.List)
		if err != nil {
			return nil, err
		}
		method.Args = args
	}
	if f.Results != nil {
		returns, err := parseArgs(f.Results.List)
		if err != nil {
			return nil, fmt.Errorf("error parsing function returns for %q: %v", method.Name, err)
		}
		method.Returns = returns
	}
	return method, nil
}

func parseArgs(fields []*ast.Field) ([]arg, error) {
	var args []arg
	for _, f := range fields {
		if len(f.Names) == 0 {
			return nil, ErrBadReturn
		}
		for _, name := range f.Names {
			var typeName string
			switch argType := f.Type.(type) {
			case *ast.Ident:
				typeName = argType.Name
			case *ast.StarExpr:
				i, ok := argType.X.(*ast.Ident)
				if !ok {
					return nil, ErrUnexpectedType{"*ast.Ident", f.Type}
				}
				typeName = "*" + i.Name
			default:
				return nil, ErrUnexpectedType{"*ast.Ident or *ast.StarExpr", f.Type}
			}

			args = append(args, arg{name.Name, typeName})
		}
	}
	return args, nil
}
