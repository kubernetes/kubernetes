package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
)

var abbrs = []string{"API", "AWS", "CIDR", "CNI", "CSI", "DBus", "DNS", "EC2", "GCE", "LUN", "NLB", "PD", "PEM", "PKCS#12", "PV", "PVC", "RBAC", "SCSI", "SSH"}

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

func checkStr(str string) bool {
	str = strings.Replace(str, "\"", "", -1)
	strs := strings.Split(str, " ")
	for _, abbr := range abbrs {
		if strs[0] == abbr {
			return true
		}
	}
	buf := []rune(str)
	return !unicode.IsUpper(buf[0]) && buf[len(str)-1] != '.'
}

func checkCall(f *ast.CallExpr) bool {
	if isCallFunc(f, "fmt", "Sprintf") {
		return checkArg(f.Args[0])
	}
	return true
}

func checkValueSpec(vs *ast.ValueSpec, name string) bool {
	for i, v := range vs.Names {
		if v.Name == name {
			lit, ok := vs.Values[i].(*ast.BasicLit)
			if !ok {
				return false
			}
			return checkStr(lit.Value)
		}
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
	return true
}

func checkArg(e ast.Expr) bool {
	switch ret := e.(type) {
	case *ast.BasicLit:
		return checkStr(ret.Value)
	case *ast.BinaryExpr:
		var str string
		ast.Inspect(e, func(n ast.Node) bool {
			if lit, ok := n.(*ast.BasicLit); ok {
				str += lit.Value
			}
			return true
		})
		return checkStr(str)
	case *ast.CallExpr:
		return checkCall(ret)
	case *ast.Ident:
		if ret.Obj == nil {
			return false
		}
		if asg, ok := ret.Obj.Decl.(*ast.AssignStmt); ok {
			return checkAsg(asg)
		}
		if vs, ok := ret.Obj.Decl.(*ast.ValueSpec); ok {
			return checkValueSpec(vs, ret.Name)
		}
		return true
	default:
		return true
	}
}

type dir struct {
	fset *token.FileSet
	pkg  *ast.Package
}

func newDir(path string) (*dir, error) {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	fset := token.NewFileSet()
	files2 := make(map[string]*ast.File, len(files))
	for _, file := range files {
		filename := file.Name()
		if !strings.HasSuffix(filename, ".go") {
			continue
		}
		file2, err := parser.ParseFile(fset, filepath.Join(path, filename), nil, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		files2[filename] = file2
	}

	pkg, _ := ast.NewPackage(fset, files2, nil, nil)
	return &dir{
		fset: fset,
		pkg:  pkg,
	}, nil
}

func (d *dir) collect() []ast.Node {
	out := []ast.Node{}
	for _, node := range d.pkg.Files {
		ast.Inspect(node, func(n ast.Node) bool {
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

	}
	return out
}

func (d *dir) filename(n ast.Node) string {
	pos := d.fset.Position(n.Pos())
	return pos.Filename + ":" + strconv.Itoa(pos.Line)
}

func main() {
	current, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}
	kuberoot := filepath.Join(current, "..", "..")
	fmt.Println("Found bad error msgs in:")
	filepath.Walk(filepath.Join(kuberoot, "pkg"), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Fatal(err)
		}
		if !info.IsDir() {
			return nil
		}

		d, err := newDir(path)
		if err != nil {
			log.Fatal(err)
		}
		nodes := d.collect()
		if len(nodes) == 0 {
			return nil
		}
		for _, node := range nodes {
			fmt.Println(d.filename(node))
		}
		return nil
	})
}
