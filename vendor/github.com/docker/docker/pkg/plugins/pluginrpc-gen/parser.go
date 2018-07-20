package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"reflect"
	"strings"
)

var errBadReturn = errors.New("found return arg with no name: all args must be named")

type errUnexpectedType struct {
	expected string
	actual   interface{}
}

func (e errUnexpectedType) Error() string {
	return fmt.Sprintf("got wrong type expecting %s, got: %v", e.expected, reflect.TypeOf(e.actual))
}

// ParsedPkg holds information about a package that has been parsed,
// its name and the list of functions.
type ParsedPkg struct {
	Name      string
	Functions []function
	Imports   []importSpec
}

type function struct {
	Name    string
	Args    []arg
	Returns []arg
	Doc     string
}

type arg struct {
	Name            string
	ArgType         string
	PackageSelector string
}

func (a *arg) String() string {
	return a.Name + " " + a.ArgType
}

type importSpec struct {
	Name string
	Path string
}

func (s *importSpec) String() string {
	var ss string
	if len(s.Name) != 0 {
		ss += s.Name
	}
	ss += s.Path
	return ss
}

// Parse parses the given file for an interface definition with the given name.
func Parse(filePath string, objName string) (*ParsedPkg, error) {
	fs := token.NewFileSet()
	pkg, err := parser.ParseFile(fs, filePath, nil, parser.AllErrors)
	if err != nil {
		return nil, err
	}
	p := &ParsedPkg{}
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
		return nil, errUnexpectedType{"*ast.TypeSpec", obj.Decl}
	}
	iface, ok := spec.Type.(*ast.InterfaceType)
	if !ok {
		return nil, errUnexpectedType{"*ast.InterfaceType", spec.Type}
	}

	p.Functions, err = parseInterface(iface)
	if err != nil {
		return nil, err
	}

	// figure out what imports will be needed
	imports := make(map[string]importSpec)
	for _, f := range p.Functions {
		args := append(f.Args, f.Returns...)
		for _, arg := range args {
			if len(arg.PackageSelector) == 0 {
				continue
			}

			for _, i := range pkg.Imports {
				if i.Name != nil {
					if i.Name.Name != arg.PackageSelector {
						continue
					}
					imports[i.Path.Value] = importSpec{Name: arg.PackageSelector, Path: i.Path.Value}
					break
				}

				_, name := path.Split(i.Path.Value)
				splitName := strings.Split(name, "-")
				if len(splitName) > 1 {
					name = splitName[len(splitName)-1]
				}
				// import paths have quotes already added in, so need to remove them for name comparison
				name = strings.TrimPrefix(name, `"`)
				name = strings.TrimSuffix(name, `"`)
				if name == arg.PackageSelector {
					imports[i.Path.Value] = importSpec{Path: i.Path.Value}
					break
				}
			}
		}
	}

	for _, spec := range imports {
		p.Imports = append(p.Imports, spec)
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
				return nil, errUnexpectedType{"*ast.TypeSpec", f.Obj.Decl}
			}
			iface, ok := spec.Type.(*ast.InterfaceType)
			if !ok {
				return nil, errUnexpectedType{"*ast.TypeSpec", spec.Type}
			}
			funcs, err := parseInterface(iface)
			if err != nil {
				fmt.Println(err)
				continue
			}
			functions = append(functions, funcs...)
		default:
			return nil, errUnexpectedType{"*astFuncType or *ast.Ident", f}
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
			return nil, errBadReturn
		}
		for _, name := range f.Names {
			p, err := parseExpr(f.Type)
			if err != nil {
				return nil, err
			}
			args = append(args, arg{name.Name, p.value, p.pkg})
		}
	}
	return args, nil
}

type parsedExpr struct {
	value string
	pkg   string
}

func parseExpr(e ast.Expr) (parsedExpr, error) {
	var parsed parsedExpr
	switch i := e.(type) {
	case *ast.Ident:
		parsed.value += i.Name
	case *ast.StarExpr:
		p, err := parseExpr(i.X)
		if err != nil {
			return parsed, err
		}
		parsed.value += "*"
		parsed.value += p.value
		parsed.pkg = p.pkg
	case *ast.SelectorExpr:
		p, err := parseExpr(i.X)
		if err != nil {
			return parsed, err
		}
		parsed.pkg = p.value
		parsed.value += p.value + "."
		parsed.value += i.Sel.Name
	case *ast.MapType:
		parsed.value += "map["
		p, err := parseExpr(i.Key)
		if err != nil {
			return parsed, err
		}
		parsed.value += p.value
		parsed.value += "]"
		p, err = parseExpr(i.Value)
		if err != nil {
			return parsed, err
		}
		parsed.value += p.value
		parsed.pkg = p.pkg
	case *ast.ArrayType:
		parsed.value += "[]"
		p, err := parseExpr(i.Elt)
		if err != nil {
			return parsed, err
		}
		parsed.value += p.value
		parsed.pkg = p.pkg
	default:
		return parsed, errUnexpectedType{"*ast.Ident or *ast.StarExpr", i}
	}
	return parsed, nil
}
