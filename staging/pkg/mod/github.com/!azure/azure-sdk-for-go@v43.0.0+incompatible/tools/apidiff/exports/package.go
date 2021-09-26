// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package exports

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
)

// Package represents a Go package.
type Package struct {
	f     *token.FileSet
	p     *ast.Package
	files map[string][]byte
}

// LoadPackageErrorInfo provides extended information about certain LoadPackage() failures.
type LoadPackageErrorInfo interface {
	PkgCount() int
}

type errorInfo struct {
	m string
	c int
}

func (ei errorInfo) Error() string {
	return ei.m
}

func (ei errorInfo) PkgCount() int {
	return ei.c
}

// LoadPackage loads the package in the specified directory.
// It's required there is only one package in the directory.
func LoadPackage(dir string) (pkg Package, err error) {
	pkg.files = map[string][]byte{}
	pkg.f = token.NewFileSet()
	packages, err := parser.ParseDir(pkg.f, dir, nil, 0)
	if err != nil {
		return
	}
	if len(packages) < 1 {
		err = errorInfo{
			m: fmt.Sprintf("didn't find any packages in '%s'", dir),
			c: len(packages),
		}
		return
	}
	if len(packages) > 1 {
		err = errorInfo{
			m: fmt.Sprintf("found more than one package in '%s'", dir),
			c: len(packages),
		}
		return
	}
	for pn := range packages {
		p := packages[pn]
		// trim any non-exported nodes
		if exp := ast.PackageExports(p); !exp {
			err = fmt.Errorf("package '%s' doesn't contain any exports", pn)
			return
		}
		pkg.p = p
		return
	}
	// shouldn't ever get here...
	panic("failed to return package")
}

// GetExports returns the exported content of the package.
func (pkg Package) GetExports() (c Content) {
	c = NewContent()
	ast.Inspect(pkg.p, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.TypeSpec:
			if t, ok := x.Type.(*ast.StructType); ok {
				c.addStruct(pkg, x.Name.Name, t)
			} else if t, ok := x.Type.(*ast.InterfaceType); ok {
				c.addInterface(pkg, x.Name.Name, t)
			}
		case *ast.FuncDecl:
			c.addFunc(pkg, x)
			// return false as we don't care about the function body.
			// this is super important as it filters out the majority of
			// the package's AST making it WAY easier to find the bits
			// of interest (not doing this will break a lot of code).
			return false
		case *ast.GenDecl:
			if x.Tok == token.CONST {
				c.addConst(pkg, x)
			}
		}
		return true
	})
	return
}

// Name returns the package name.
func (pkg Package) Name() string {
	return pkg.p.Name
}

// Get loads the package in the specified directory and returns the exported
// content.  It's a convenience wrapper around LoadPackage() and GetExports().
func Get(pkgDir string) (Content, error) {
	pkg, err := LoadPackage(pkgDir)
	if err != nil {
		return Content{}, err
	}
	return pkg.GetExports(), nil
}

// returns the text between [start, end]
func (pkg Package) getText(start token.Pos, end token.Pos) string {
	// convert to absolute position within the containing file
	p := pkg.f.Position(start)
	// check if the file has been loaded, if not then load it
	if _, ok := pkg.files[p.Filename]; !ok {
		b, err := ioutil.ReadFile(p.Filename)
		if err != nil {
			panic(err)
		}
		pkg.files[p.Filename] = b
	}
	return string(pkg.files[p.Filename][p.Offset : p.Offset+int(end-start)])
}

// iterates over the specified field list, for each field the specified
// callback is invoked with the name of the field and the type name.  the field
// name can be nil, e.g. anonymous fields in structs, unnamed return types etc.
func (pkg Package) translateFieldList(fl []*ast.Field, cb func(*string, string)) {
	for _, f := range fl {
		var name *string
		if f.Names != nil {
			n := pkg.getText(f.Names[0].Pos(), f.Names[0].End())
			name = &n
		}
		t := pkg.getText(f.Type.Pos(), f.Type.End())
		cb(name, t)
	}
}

// creates a Func object from the specified ast.FuncType
func (pkg Package) buildFunc(ft *ast.FuncType) (f Func) {
	// appends a to s, comma-delimited style, and returns s
	appendString := func(s, a string) string {
		if s != "" {
			s += ","
		}
		s += a
		return s
	}

	// build the params type list
	if ft.Params.List != nil {
		p := ""
		pkg.translateFieldList(ft.Params.List, func(n *string, t string) {
			p = appendString(p, t)
		})
		f.Params = &p
	}

	// build the return types list
	if ft.Results != nil {
		r := ""
		pkg.translateFieldList(ft.Results.List, func(n *string, t string) {
			r = appendString(r, t)
		})
		f.Returns = &r
	}
	return
}
