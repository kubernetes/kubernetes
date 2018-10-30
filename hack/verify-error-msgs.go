package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

func isCallFunc(f *ast.CallExpr, x string, name string) bool {
	f2, ok := f.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	x2, ok := f2.X.(*ast.Ident)
	if !ok {
		return false
	}
	return x2.Name == x && f2.Sel.Name == name
}

func printNode(fset *token.FileSet, node interface{}) {
	printer.Fprint(os.Stdout, fset, node)
	fmt.Fprint(os.Stdout, '\n')
}

func checkStr(str string) bool {
	str = strings.Replace(str, "\"", "", -1)
	buf := []rune(str)
	return !unicode.IsUpper(buf[0]) && buf[len(str)-1] != '.'
}

func checkCall(f *ast.CallExpr) bool {
	if isCallFunc(f, "fmt", "Sprintf") {
		return checkArg(f.Args[0])
	}
	return false
}

func checkAsg(a *ast.AssignStmt) bool {
	if lit, ok := a.Rhs[0].(*ast.BasicLit); ok {
		return checkStr(lit.Value)
	}
	if f, ok := a.Rhs[0].(*ast.CallExpr); ok {
		return checkCall(f)
	}
	return false
}

func checkArg(e ast.Expr) bool {
	switch ret := e.(type) {
	case *ast.BasicLit:
		return checkStr(ret.Value)
	case *ast.CallExpr:
		return checkCall(ret)
	case *ast.Ident:
		if ret.Obj == nil {
			return false
		}
		asg, ok := ret.Obj.Decl.(*ast.AssignStmt)
		if !ok {
			return false
		}
		return checkAsg(asg)
	default:
		return false
	}
}

type file struct {
	fset *token.FileSet
	node ast.Node
}

func newFile(path string) (*file, error) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, path, nil, parser.AllErrors)
	if err != nil {
		return nil, err
	}
	return &file{
		fset: fset,
		node: node,
	}, nil
}

func (f *file) collect() []ast.Node {
	out := []ast.Node{}
	ast.Inspect(f.node, func(n ast.Node) bool {
		switch ret := n.(type) {
		case *ast.CallExpr:
			if isCallFunc(ret, "errors", "New") || isCallFunc(ret, "fmt", "Errorf") {
				if !checkArg(ret.Args[0]) {
					out = append(out, n)
				}
			}
		}
		return true
	})
	return out
}

func (f *file) line(n ast.Node) int {
	return f.fset.Position(n.Pos()).Line
}

func main() {
	current, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}
	kuberoot := filepath.Join(current, "..")
	filepath.Walk(filepath.Join(kuberoot, "pkg"), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Fatal(err)
		}
		if !strings.HasSuffix(info.Name(), ".go") {
			return nil
		}

		f, err := newFile(path)
		if err != nil {
			log.Fatal(err)
		}
		nodes := f.collect()
		if len(nodes) == 0 {
			return nil
		}
		for _, node := range nodes {
			fmt.Printf("%s:%d\n", path, f.line(node))
		}
		return nil
	})
}
