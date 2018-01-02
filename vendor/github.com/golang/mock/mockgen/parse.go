// Copyright 2012 Google Inc.
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

package main

// This file contains the model construction by parsing source files.

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"path"
	"strconv"
	"strings"

	"github.com/golang/mock/mockgen/model"
)

var (
	imports  = flag.String("imports", "", "(source mode) Comma-separated name=path pairs of explicit imports to use.")
	auxFiles = flag.String("aux_files", "", "(source mode) Comma-separated pkg=path pairs of auxiliary Go source files.")
)

// TODO: simplify error reporting

func ParseFile(source string) (*model.Package, error) {
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, source, nil, 0)
	if err != nil {
		return nil, fmt.Errorf("failed parsing source file %v: %v", source, err)
	}

	p := &fileParser{
		fileSet:       fs,
		imports:       make(map[string]string),
		auxInterfaces: make(map[string]map[string]*ast.InterfaceType),
	}

	// Handle -imports.
	dotImports := make(map[string]bool)
	if *imports != "" {
		for _, kv := range strings.Split(*imports, ",") {
			eq := strings.Index(kv, "=")
			k, v := kv[:eq], kv[eq+1:]
			if k == "." {
				// TODO: Catch dupes?
				dotImports[v] = true
			} else {
				// TODO: Catch dupes?
				p.imports[k] = v
			}
		}
	}

	// Handle -aux_files.
	if err := p.parseAuxFiles(*auxFiles); err != nil {
		return nil, err
	}
	p.addAuxInterfacesFromFile("", file) // this file

	pkg, err := p.parseFile(file)
	if err != nil {
		return nil, err
	}
	pkg.DotImports = make([]string, 0, len(dotImports))
	for path := range dotImports {
		pkg.DotImports = append(pkg.DotImports, path)
	}
	return pkg, nil
}

type fileParser struct {
	fileSet *token.FileSet
	imports map[string]string // package name => import path

	auxFiles      []*ast.File
	auxInterfaces map[string]map[string]*ast.InterfaceType // package (or "") => name => interface
}

func (p *fileParser) errorf(pos token.Pos, format string, args ...interface{}) error {
	ps := p.fileSet.Position(pos)
	format = "%s:%d:%d: " + format
	args = append([]interface{}{ps.Filename, ps.Line, ps.Column}, args...)
	return fmt.Errorf(format, args...)
}

func (p *fileParser) parseAuxFiles(auxFiles string) error {
	auxFiles = strings.TrimSpace(auxFiles)
	if auxFiles == "" {
		return nil
	}
	for _, kv := range strings.Split(auxFiles, ",") {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("bad aux file spec: %v", kv)
		}
		file, err := parser.ParseFile(p.fileSet, parts[1], nil, 0)
		if err != nil {
			return err
		}
		p.auxFiles = append(p.auxFiles, file)
		p.addAuxInterfacesFromFile(parts[0], file)
	}
	return nil
}

func (p *fileParser) addAuxInterfacesFromFile(pkg string, file *ast.File) {
	if _, ok := p.auxInterfaces[pkg]; !ok {
		p.auxInterfaces[pkg] = make(map[string]*ast.InterfaceType)
	}
	for ni := range iterInterfaces(file) {
		p.auxInterfaces[pkg][ni.name.Name] = ni.it
	}
}

func (p *fileParser) parseFile(file *ast.File) (*model.Package, error) {
	allImports := importsOfFile(file)
	// Don't stomp imports provided by -imports. Those should take precedence.
	for pkg, path := range allImports {
		if _, ok := p.imports[pkg]; !ok {
			p.imports[pkg] = path
		}
	}
	// Add imports from auxiliary files, which might be needed for embedded interfaces.
	// Don't stomp any other imports.
	for _, f := range p.auxFiles {
		for pkg, path := range importsOfFile(f) {
			if _, ok := p.imports[pkg]; !ok {
				p.imports[pkg] = path
			}
		}
	}

	var is []*model.Interface
	for ni := range iterInterfaces(file) {
		i, err := p.parseInterface(ni.name.String(), "", ni.it)
		if err != nil {
			return nil, err
		}
		is = append(is, i)
	}
	return &model.Package{
		Name:       file.Name.String(),
		Interfaces: is,
	}, nil
}

func (p *fileParser) parseInterface(name, pkg string, it *ast.InterfaceType) (*model.Interface, error) {
	intf := &model.Interface{Name: name}
	for _, field := range it.Methods.List {
		switch v := field.Type.(type) {
		case *ast.FuncType:
			if nn := len(field.Names); nn != 1 {
				return nil, fmt.Errorf("expected one name for interface %v, got %d", intf.Name, nn)
			}
			m := &model.Method{
				Name: field.Names[0].String(),
			}
			var err error
			m.In, m.Variadic, m.Out, err = p.parseFunc(pkg, v)
			if err != nil {
				return nil, err
			}
			intf.Methods = append(intf.Methods, m)
		case *ast.Ident:
			// Embedded interface in this package.
			ei := p.auxInterfaces[""][v.String()]
			if ei == nil {
				return nil, p.errorf(v.Pos(), "unknown embedded interface %s", v.String())
			}
			eintf, err := p.parseInterface(v.String(), pkg, ei)
			if err != nil {
				return nil, err
			}
			// Copy the methods.
			// TODO: apply shadowing rules.
			for _, m := range eintf.Methods {
				intf.Methods = append(intf.Methods, m)
			}
		case *ast.SelectorExpr:
			// Embedded interface in another package.
			fpkg, sel := v.X.(*ast.Ident).String(), v.Sel.String()
			ei := p.auxInterfaces[fpkg][sel]
			if ei == nil {
				return nil, p.errorf(v.Pos(), "unknown embedded interface %s.%s", fpkg, sel)
			}
			epkg, ok := p.imports[fpkg]
			if !ok {
				return nil, p.errorf(v.X.Pos(), "unknown package %s", fpkg)
			}
			eintf, err := p.parseInterface(sel, epkg, ei)
			if err != nil {
				return nil, err
			}
			// Copy the methods.
			// TODO: apply shadowing rules.
			for _, m := range eintf.Methods {
				intf.Methods = append(intf.Methods, m)
			}
		default:
			return nil, fmt.Errorf("don't know how to mock method of type %T", field.Type)
		}
	}
	return intf, nil
}

func (p *fileParser) parseFunc(pkg string, f *ast.FuncType) (in []*model.Parameter, variadic *model.Parameter, out []*model.Parameter, err error) {
	if f.Params != nil {
		regParams := f.Params.List
		if isVariadic(f) {
			n := len(regParams)
			varParams := regParams[n-1:]
			regParams = regParams[:n-1]
			vp, err := p.parseFieldList(pkg, varParams)
			if err != nil {
				return nil, nil, nil, p.errorf(varParams[0].Pos(), "failed parsing variadic argument: %v", err)
			}
			variadic = vp[0]
		}
		in, err = p.parseFieldList(pkg, regParams)
		if err != nil {
			return nil, nil, nil, p.errorf(f.Pos(), "failed parsing arguments: %v", err)
		}
	}
	if f.Results != nil {
		out, err = p.parseFieldList(pkg, f.Results.List)
		if err != nil {
			return nil, nil, nil, p.errorf(f.Pos(), "failed parsing returns: %v", err)
		}
	}
	return
}

func (p *fileParser) parseFieldList(pkg string, fields []*ast.Field) ([]*model.Parameter, error) {
	nf := 0
	for _, f := range fields {
		nn := len(f.Names)
		if nn == 0 {
			nn = 1 // anonymous parameter
		}
		nf += nn
	}
	if nf == 0 {
		return nil, nil
	}
	ps := make([]*model.Parameter, nf)
	i := 0 // destination index
	for _, f := range fields {
		t, err := p.parseType(pkg, f.Type)
		if err != nil {
			return nil, err
		}

		if len(f.Names) == 0 {
			// anonymous arg
			ps[i] = &model.Parameter{Type: t}
			i++
			continue
		}
		for _, name := range f.Names {
			ps[i] = &model.Parameter{Name: name.Name, Type: t}
			i++
		}
	}
	return ps, nil
}

func (p *fileParser) parseType(pkg string, typ ast.Expr) (model.Type, error) {
	switch v := typ.(type) {
	case *ast.ArrayType:
		ln := -1
		if v.Len != nil {
			x, err := strconv.Atoi(v.Len.(*ast.BasicLit).Value)
			if err != nil {
				return nil, p.errorf(v.Len.Pos(), "bad array size: %v", err)
			}
			ln = x
		}
		t, err := p.parseType(pkg, v.Elt)
		if err != nil {
			return nil, err
		}
		return &model.ArrayType{Len: ln, Type: t}, nil
	case *ast.ChanType:
		t, err := p.parseType(pkg, v.Value)
		if err != nil {
			return nil, err
		}
		var dir model.ChanDir
		if v.Dir == ast.SEND {
			dir = model.SendDir
		}
		if v.Dir == ast.RECV {
			dir = model.RecvDir
		}
		return &model.ChanType{Dir: dir, Type: t}, nil
	case *ast.Ellipsis:
		// assume we're parsing a variadic argument
		return p.parseType(pkg, v.Elt)
	case *ast.FuncType:
		in, variadic, out, err := p.parseFunc(pkg, v)
		if err != nil {
			return nil, err
		}
		return &model.FuncType{In: in, Out: out, Variadic: variadic}, nil
	case *ast.Ident:
		if v.IsExported() {
			// assume type in this package
			return &model.NamedType{Package: pkg, Type: v.Name}, nil
		} else {
			// assume predeclared type
			return model.PredeclaredType(v.Name), nil
		}
	case *ast.InterfaceType:
		if v.Methods != nil && len(v.Methods.List) > 0 {
			return nil, p.errorf(v.Pos(), "can't handle non-empty unnamed interface types")
		}
		return model.PredeclaredType("interface{}"), nil
	case *ast.MapType:
		key, err := p.parseType(pkg, v.Key)
		if err != nil {
			return nil, err
		}
		value, err := p.parseType(pkg, v.Value)
		if err != nil {
			return nil, err
		}
		return &model.MapType{Key: key, Value: value}, nil
	case *ast.SelectorExpr:
		pkgName := v.X.(*ast.Ident).String()
		pkg, ok := p.imports[pkgName]
		if !ok {
			return nil, p.errorf(v.Pos(), "unknown package %q", pkgName)
		}
		return &model.NamedType{Package: pkg, Type: v.Sel.String()}, nil
	case *ast.StarExpr:
		t, err := p.parseType(pkg, v.X)
		if err != nil {
			return nil, err
		}
		return &model.PointerType{Type: t}, nil
	case *ast.StructType:
		if v.Fields != nil && len(v.Fields.List) > 0 {
			return nil, p.errorf(v.Pos(), "can't handle non-empty unnamed struct types")
		}
		return model.PredeclaredType("struct{}"), nil
	}

	return nil, fmt.Errorf("don't know how to parse type %T", typ)
}

// importsOfFile returns a map of package name to import path
// of the imports in file.
func importsOfFile(file *ast.File) map[string]string {
	/* We have to make guesses about some imports, because imports are not required
	 * to have names. Named imports are always certain. Unnamed imports are guessed
	 * to have a name of the last path component; if the last path component has dots,
	 * the first dot-delimited field is used as the name.
	 */

	m := make(map[string]string)
	for _, decl := range file.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.IMPORT {
			continue
		}
		for _, spec := range gd.Specs {
			is, ok := spec.(*ast.ImportSpec)
			if !ok {
				continue
			}
			pkg, importPath := "", string(is.Path.Value)
			importPath = importPath[1 : len(importPath)-1] // remove quotes

			if is.Name != nil {
				if is.Name.Name == "_" {
					continue
				}
				pkg = removeDot(is.Name.Name)
			} else {
				_, last := path.Split(importPath)
				pkg = strings.SplitN(last, ".", 2)[0]
			}
			if _, ok := m[pkg]; ok {
				log.Fatalf("imported package collision: %q imported twice", pkg)
			}
			m[pkg] = importPath
		}
	}
	return m
}

type namedInterface struct {
	name *ast.Ident
	it   *ast.InterfaceType
}

// Create an iterator over all interfaces in file.
func iterInterfaces(file *ast.File) <-chan namedInterface {
	ch := make(chan namedInterface)
	go func() {
		for _, decl := range file.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				it, ok := ts.Type.(*ast.InterfaceType)
				if !ok {
					continue
				}

				ch <- namedInterface{ts.Name, it}
			}
		}
		close(ch)
	}()
	return ch
}

// isVariadic returns whether the function is variadic.
func isVariadic(f *ast.FuncType) bool {
	nargs := len(f.Params.List)
	if nargs == 0 {
		return false
	}
	_, ok := f.Params.List[nargs-1].Type.(*ast.Ellipsis)
	return ok
}
