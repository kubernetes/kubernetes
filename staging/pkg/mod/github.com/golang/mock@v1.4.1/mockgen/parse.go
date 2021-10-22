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
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/golang/mock/mockgen/model"
	"golang.org/x/tools/go/packages"
)

var (
	imports  = flag.String("imports", "", "(source mode) Comma-separated name=path pairs of explicit imports to use.")
	auxFiles = flag.String("aux_files", "", "(source mode) Comma-separated pkg=path pairs of auxiliary Go source files.")
)

// TODO: simplify error reporting

// sourceMode generates mocks via source file.
func sourceMode(source string) (*model.Package, error) {
	srcDir, err := filepath.Abs(filepath.Dir(source))
	if err != nil {
		return nil, fmt.Errorf("failed getting source directory: %v", err)
	}

	packageImport, err := parsePackageImport(source, srcDir)
	if err != nil {
		return nil, err
	}

	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, source, nil, 0)
	if err != nil {
		return nil, fmt.Errorf("failed parsing source file %v: %v", source, err)
	}

	p := &fileParser{
		fileSet:            fs,
		imports:            make(map[string]string),
		importedInterfaces: make(map[string]map[string]*ast.InterfaceType),
		auxInterfaces:      make(map[string]map[string]*ast.InterfaceType),
		srcDir:             srcDir,
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
	p.addAuxInterfacesFromFile(packageImport, file) // this file

	pkg, err := p.parseFile(packageImport, file)
	if err != nil {
		return nil, err
	}
	for pkgPath := range dotImports {
		pkg.DotImports = append(pkg.DotImports, pkgPath)
	}
	return pkg, nil
}

type fileParser struct {
	fileSet            *token.FileSet
	imports            map[string]string                        // package name => import path
	importedInterfaces map[string]map[string]*ast.InterfaceType // package (or "") => name => interface

	auxFiles      []*ast.File
	auxInterfaces map[string]map[string]*ast.InterfaceType // package (or "") => name => interface

	srcDir string
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
		pkg, fpath := parts[0], parts[1]

		file, err := parser.ParseFile(p.fileSet, fpath, nil, 0)
		if err != nil {
			return err
		}
		p.auxFiles = append(p.auxFiles, file)
		p.addAuxInterfacesFromFile(pkg, file)
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

// parseFile loads all file imports and auxiliary files import into the
// fileParser, parses all file interfaces and returns package model.
func (p *fileParser) parseFile(importPath string, file *ast.File) (*model.Package, error) {
	allImports, dotImports := importsOfFile(file)
	// Don't stomp imports provided by -imports. Those should take precedence.
	for pkg, pkgPath := range allImports {
		if _, ok := p.imports[pkg]; !ok {
			p.imports[pkg] = pkgPath
		}
	}
	// Add imports from auxiliary files, which might be needed for embedded interfaces.
	// Don't stomp any other imports.
	for _, f := range p.auxFiles {
		auxImports, _ := importsOfFile(f)
		for pkg, pkgPath := range auxImports {
			if _, ok := p.imports[pkg]; !ok {
				p.imports[pkg] = pkgPath
			}
		}
	}

	var is []*model.Interface
	for ni := range iterInterfaces(file) {
		i, err := p.parseInterface(ni.name.String(), importPath, ni.it)
		if err != nil {
			return nil, err
		}
		is = append(is, i)
	}
	return &model.Package{
		Name:       file.Name.String(),
		PkgPath:    importPath,
		Interfaces: is,
		DotImports: dotImports,
	}, nil
}

// parsePackage loads package specified by path, parses it and populates
// corresponding imports and importedInterfaces into the fileParser.
func (p *fileParser) parsePackage(path string) error {
	var pkgs map[string]*ast.Package
	if imp, err := build.Import(path, p.srcDir, build.FindOnly); err != nil {
		return err
	} else if pkgs, err = parser.ParseDir(p.fileSet, imp.Dir, nil, 0); err != nil {
		return err
	}
	for _, pkg := range pkgs {
		file := ast.MergePackageFiles(pkg, ast.FilterFuncDuplicates|ast.FilterUnassociatedComments|ast.FilterImportDuplicates)
		if _, ok := p.importedInterfaces[path]; !ok {
			p.importedInterfaces[path] = make(map[string]*ast.InterfaceType)
		}
		for ni := range iterInterfaces(file) {
			p.importedInterfaces[path][ni.name.Name] = ni.it
		}
		imports, _ := importsOfFile(file)
		for pkgName, pkgPath := range imports {
			if _, ok := p.imports[pkgName]; !ok {
				p.imports[pkgName] = pkgPath
			}
		}
	}
	return nil
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
			ei := p.auxInterfaces[pkg][v.String()]
			if ei == nil {
				if ei = p.importedInterfaces[pkg][v.String()]; ei == nil {
					return nil, p.errorf(v.Pos(), "unknown embedded interface %s", v.String())
				}
			}
			eintf, err := p.parseInterface(v.String(), pkg, ei)
			if err != nil {
				return nil, err
			}
			// Copy the methods.
			// TODO: apply shadowing rules.
			intf.Methods = append(intf.Methods, eintf.Methods...)
		case *ast.SelectorExpr:
			// Embedded interface in another package.
			fpkg, sel := v.X.(*ast.Ident).String(), v.Sel.String()
			epkg, ok := p.imports[fpkg]
			if !ok {
				return nil, p.errorf(v.X.Pos(), "unknown package %s", fpkg)
			}
			ei := p.auxInterfaces[fpkg][sel]
			if ei == nil {
				fpkg = epkg
				if _, ok = p.importedInterfaces[epkg]; !ok {
					if err := p.parsePackage(epkg); err != nil {
						return nil, p.errorf(v.Pos(), "could not parse package %s: %v", fpkg, err)
					}
				}
				if ei = p.importedInterfaces[epkg][sel]; ei == nil {
					return nil, p.errorf(v.Pos(), "unknown embedded interface %s.%s", fpkg, sel)
				}
			}
			eintf, err := p.parseInterface(sel, fpkg, ei)
			if err != nil {
				return nil, err
			}
			// Copy the methods.
			// TODO: apply shadowing rules.
			intf.Methods = append(intf.Methods, eintf.Methods...)
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
			// `pkg` may be an aliased imported pkg
			// if so, patch the import w/ the fully qualified import
			maybeImportedPkg, ok := p.imports[pkg]
			if ok {
				pkg = maybeImportedPkg
			}
			// assume type in this package
			return &model.NamedType{Package: pkg, Type: v.Name}, nil
		}

		// assume predeclared type
		return model.PredeclaredType(v.Name), nil
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
func importsOfFile(file *ast.File) (normalImports map[string]string, dotImports []string) {
	var importPaths []string
	for _, is := range file.Imports {
		if is.Name != nil {
			continue
		}
		importPath := is.Path.Value[1 : len(is.Path.Value)-1] // remove quotes
		importPaths = append(importPaths, importPath)
	}
	packagesName := createPackageMap(importPaths)
	normalImports = make(map[string]string)
	dotImports = make([]string, 0)
	for _, is := range file.Imports {
		var pkgName string
		importPath := is.Path.Value[1 : len(is.Path.Value)-1] // remove quotes

		if is.Name != nil {
			// Named imports are always certain.
			if is.Name.Name == "_" {
				continue
			}
			pkgName = is.Name.Name
		} else {
			pkg, ok := packagesName[importPath]
			if !ok {
				// Fallback to import path suffix. Note that this is uncertain.
				_, last := path.Split(importPath)
				// If the last path component has dots, the first dot-delimited
				// field is used as the name.
				pkgName = strings.SplitN(last, ".", 2)[0]
			} else {
				pkgName = pkg
			}
		}

		if pkgName == "." {
			dotImports = append(dotImports, importPath)
		} else {

			if _, ok := normalImports[pkgName]; ok {
				log.Fatalf("imported package collision: %q imported twice", pkgName)
			}
			normalImports[pkgName] = importPath
		}
	}
	return
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

// packageNameOfDir get package import path via dir
func packageNameOfDir(srcDir string) (string, error) {
	files, err := ioutil.ReadDir(srcDir)
	if err != nil {
		log.Fatal(err)
	}

	var goFilePath string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".go") {
			goFilePath = file.Name()
			break
		}
	}
	if goFilePath == "" {
		return "", fmt.Errorf("go source file not found %s", srcDir)
	}

	packageImport, err := parsePackageImport(goFilePath, srcDir)
	if err != nil {
		return "", err
	}
	return packageImport, nil
}

// parseImportPackage get package import path via source file
func parsePackageImport(source, srcDir string) (string, error) {
	cfg := &packages.Config{Mode: packages.LoadFiles, Tests: true, Dir: srcDir}
	pkgs, err := packages.Load(cfg, "file="+source)
	if err != nil {
		return "", err
	}
	if packages.PrintErrors(pkgs) > 0 || len(pkgs) == 0 {
		return "", errors.New("loading package failed")
	}

	packageImport := pkgs[0].PkgPath

	// It is illegal to import a _test package.
	packageImport = strings.TrimSuffix(packageImport, "_test")
	return packageImport, nil
}
