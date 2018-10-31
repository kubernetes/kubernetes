/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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

var (
	abbrs   = []string{"API", "AWS", "CIDR", "CPU", "CNI", "CSI", "DBus", "DNS", "EC2", "IPVS", "GCE", "LUN", "NLB", "PD", "PEM", "PKCS#12", "PV", "PVC", "RBAC", "SCSI", "SCTP", "SSH", "URL"}
	ignores = []string{"kubelet/container/sync_result.go", "kubelet/images/types.go", "kubelet/kuberuntime/kuberuntime_container.go"}
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

func (d *dir) checkStr(str string) bool {
	str = strings.Replace(str, "\"", "", -1)
	str = strings.Replace(str, ",", "", -1)
	strs := strings.Split(str, " ")
	for _, abbr := range abbrs {
		if strs[0] == abbr {
			return true
		}
	}
	if _, ok := d.symbols[strs[0]]; ok {
		return true
	}

	buf := []rune(str)
	return !unicode.IsUpper(buf[0]) && buf[len(str)-1] != '.'
}

func (d *dir) checkCall(f *ast.CallExpr) bool {
	if isCallFunc(f, "fmt", "Sprintf") {
		return d.checkArg(f.Args[0])
	}
	return true
}

func (d *dir) checkValueSpec(vs *ast.ValueSpec, name string) bool {
	for i, v := range vs.Names {
		if v.Name == name {
			lit, ok := vs.Values[i].(*ast.BasicLit)
			if !ok {
				return false
			}
			return d.checkStr(lit.Value)
		}
	}
	return false
}

func (d *dir) checkAsg(a *ast.AssignStmt) bool {
	if lit, ok := a.Rhs[0].(*ast.BasicLit); ok {
		return d.checkStr(lit.Value)
	}
	if f, ok := a.Rhs[0].(*ast.CallExpr); ok {
		return d.checkCall(f)
	}
	return true
}

func (d *dir) checkArg(e ast.Expr) bool {
	switch ret := e.(type) {
	case *ast.BasicLit:
		return d.checkStr(ret.Value)
	case *ast.BinaryExpr:
		var str string
		ast.Inspect(e, func(n ast.Node) bool {
			if lit, ok := n.(*ast.BasicLit); ok {
				str += lit.Value
			}
			return true
		})
		return d.checkStr(str)
	case *ast.CallExpr:
		return d.checkCall(ret)
	case *ast.Ident:
		if ret.Obj == nil {
			return false
		}
		if asg, ok := ret.Obj.Decl.(*ast.AssignStmt); ok {
			return d.checkAsg(asg)
		}
		if vs, ok := ret.Obj.Decl.(*ast.ValueSpec); ok {
			return d.checkValueSpec(vs, ret.Name)
		}
		return true
	default:
		return true
	}
}

type dir struct {
	fset    *token.FileSet
	pkg     *ast.Package
	symbols map[string]struct{}
}

func newDir(path string) (*dir, error) {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	fset := token.NewFileSet()
	files2 := make(map[string]*ast.File, len(files))
L:
	for _, file := range files {
		filename := file.Name()
		if !strings.HasSuffix(filename, ".go") {
			continue
		}
		for _, ignore := range ignores {
			if strings.HasSuffix(filepath.Join(path, filename), ignore) {
				continue L
			}
		}
		file2, err := parser.ParseFile(fset, filepath.Join(path, filename), nil, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		files2[filename] = file2
	}

	pkg, _ := ast.NewPackage(fset, files2, nil, nil)
	symbols := map[string]struct{}{}
	for _, node := range pkg.Files {
		ast.Inspect(node, func(n ast.Node) bool {
			if ret, ok := n.(*ast.Ident); ok {
				symbols[ret.Name] = struct{}{}
			}
			return true
		})
	}
	return &dir{
		fset:    fset,
		pkg:     pkg,
		symbols: symbols,
	}, nil
}

func (d *dir) collect() []ast.Node {
	out := []ast.Node{}
	for _, node := range d.pkg.Files {
		ast.Inspect(node, func(n ast.Node) bool {
			if ret, ok := n.(*ast.CallExpr); ok {
				if isCallFunc(ret, "errors", "New") || isCallFunc(ret, "fmt", "Errorf") {
					if !d.checkArg(ret.Args[0]) {
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
	name, _ := filepath.Abs(pos.Filename)
	return name + ":" + strconv.Itoa(pos.Line)
}

func main() {
	if len(os.Args) == 1 {
		log.Fatal("Input kubernetes root path")
	}
	kuberoot := os.Args[1]
	files := []string{}
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
			files = append(files, d.filename(node))
		}
		return nil
	})
	if len(files) != 0 {
		fmt.Println("Found bad error msgs in:")
		fmt.Println(strings.Join(files, "\n"))
		os.Exit(1)
	}
}
