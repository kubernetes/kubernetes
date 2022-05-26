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
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io/ioutil"
	"log"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/golang/mock/mockgen/model"
)

var (
	imports  = flag.String("imports", "", "(source mode) Comma-separated name=path pairs of explicit imports to use.")
	auxFiles = flag.String("aux_files", "", "(source mode) Comma-separated pkg=path pairs of auxiliary Go source files.")
)

// sourceMode generates mocks via source file.
func sourceMode(source string) (*model.Package, error) {
	srcDir, err := filepath.Abs(filepath.Dir(source))
	if err != nil {
		return nil, fmt.Errorf("failed getting source directory: %v", err)
	}

	packageImport, err := parsePackageImport(srcDir)
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
		imports:            make(map[string]importedPackage),
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
				dotImports[v] = true
			} else {
				p.imports[k] = importedPkg{path: v}
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

type importedPackage interface {
	Path() string
	Parser() *fileParser
}

type importedPkg struct {
	path   string
	parser *fileParser
}

func (i importedPkg) Path() string        { return i.path }
func (i importedPkg) Parser() *fileParser { return i.parser }

// duplicateImport is a bit of a misnomer. Currently the parser can't
// handle cases of multi-file packages importing different packages
// under the same name. Often these imports would not be problematic,
// so this type lets us defer raising an error unless the package name
// is actually used.
type duplicateImport struct {
	name       string
	duplicates []string
}

func (d duplicateImport) Error() string {
	return fmt.Sprintf("%q is ambiguous because of duplicate imports: %v", d.name, d.duplicates)
}

func (d duplicateImport) Path() string        { log.Fatal(d.Error()); return "" }
func (d duplicateImport) Parser() *fileParser { log.Fatal(d.Error()); return nil }

type fileParser struct {
	fileSet            *token.FileSet
	imports            map[string]importedPackage               // package name => imported package
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
	for pkg, pkgI := range allImports {
		if _, ok := p.imports[pkg]; !ok {
			p.imports[pkg] = pkgI
		}
	}
	// Add imports from auxiliary files, which might be needed for embedded interfaces.
	// Don't stomp any other imports.
	for _, f := range p.auxFiles {
		auxImports, _ := importsOfFile(f)
		for pkg, pkgI := range auxImports {
			if _, ok := p.imports[pkg]; !ok {
				p.imports[pkg] = pkgI
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

// parsePackage loads package specified by path, parses it and returns
// a new fileParser with the parsed imports and interfaces.
func (p *fileParser) parsePackage(path string) (*fileParser, error) {
	newP := &fileParser{
		fileSet:            token.NewFileSet(),
		imports:            make(map[string]importedPackage),
		importedInterfaces: make(map[string]map[string]*ast.InterfaceType),
		auxInterfaces:      make(map[string]map[string]*ast.InterfaceType),
		srcDir:             p.srcDir,
	}

	var pkgs map[string]*ast.Package
	if imp, err := build.Import(path, newP.srcDir, build.FindOnly); err != nil {
		return nil, err
	} else if pkgs, err = parser.ParseDir(newP.fileSet, imp.Dir, nil, 0); err != nil {
		return nil, err
	}

	for _, pkg := range pkgs {
		file := ast.MergePackageFiles(pkg, ast.FilterFuncDuplicates|ast.FilterUnassociatedComments|ast.FilterImportDuplicates)
		if _, ok := newP.importedInterfaces[path]; !ok {
			newP.importedInterfaces[path] = make(map[string]*ast.InterfaceType)
		}
		for ni := range iterInterfaces(file) {
			newP.importedInterfaces[path][ni.name.Name] = ni.it
		}
		imports, _ := importsOfFile(file)
		for pkgName, pkgI := range imports {
			newP.imports[pkgName] = pkgI
		}
	}
	return newP, nil
}

func (p *fileParser) parseInterface(name, pkg string, it *ast.InterfaceType) (*model.Interface, error) {
	iface := &model.Interface{Name: name}
	for _, field := range it.Methods.List {
		switch v := field.Type.(type) {
		case *ast.FuncType:
			if nn := len(field.Names); nn != 1 {
				return nil, fmt.Errorf("expected one name for interface %v, got %d", iface.Name, nn)
			}
			m := &model.Method{
				Name: field.Names[0].String(),
			}
			var err error
			m.In, m.Variadic, m.Out, err = p.parseFunc(pkg, v)
			if err != nil {
				return nil, err
			}
			iface.AddMethod(m)
		case *ast.Ident:
			// Embedded interface in this package.
			embeddedIfaceType := p.auxInterfaces[pkg][v.String()]
			if embeddedIfaceType == nil {
				embeddedIfaceType = p.importedInterfaces[pkg][v.String()]
			}

			var embeddedIface *model.Interface
			if embeddedIfaceType != nil {
				var err error
				embeddedIface, err = p.parseInterface(v.String(), pkg, embeddedIfaceType)
				if err != nil {
					return nil, err
				}
			} else {
				// This is built-in error interface.
				if v.String() == model.ErrorInterface.Name {
					embeddedIface = &model.ErrorInterface
				} else {
					return nil, p.errorf(v.Pos(), "unknown embedded interface %s", v.String())
				}
			}
			// Copy the methods.
			for _, m := range embeddedIface.Methods {
				iface.AddMethod(m)
			}
		case *ast.SelectorExpr:
			// Embedded interface in another package.
			filePkg, sel := v.X.(*ast.Ident).String(), v.Sel.String()
			embeddedPkg, ok := p.imports[filePkg]
			if !ok {
				return nil, p.errorf(v.X.Pos(), "unknown package %s", filePkg)
			}

			var embeddedIface *model.Interface
			var err error
			embeddedIfaceType := p.auxInterfaces[filePkg][sel]
			if embeddedIfaceType != nil {
				embeddedIface, err = p.parseInterface(sel, filePkg, embeddedIfaceType)
				if err != nil {
					return nil, err
				}
			} else {
				path := embeddedPkg.Path()
				parser := embeddedPkg.Parser()
				if parser == nil {
					ip, err := p.parsePackage(path)
					if err != nil {
						return nil, p.errorf(v.Pos(), "could not parse package %s: %v", path, err)
					}
					parser = ip
					p.imports[filePkg] = importedPkg{
						path:   embeddedPkg.Path(),
						parser: parser,
					}
				}
				if embeddedIfaceType = parser.importedInterfaces[path][sel]; embeddedIfaceType == nil {
					return nil, p.errorf(v.Pos(), "unknown embedded interface %s.%s", path, sel)
				}
				embeddedIface, err = parser.parseInterface(sel, path, embeddedIfaceType)
				if err != nil {
					return nil, err
				}
			}
			// Copy the methods.
			// TODO: apply shadowing rules.
			for _, m := range embeddedIface.Methods {
				iface.AddMethod(m)
			}
		default:
			return nil, fmt.Errorf("don't know how to mock method of type %T", field.Type)
		}
	}
	return iface, nil
}

func (p *fileParser) parseFunc(pkg string, f *ast.FuncType) (inParam []*model.Parameter, variadic *model.Parameter, outParam []*model.Parameter, err error) {
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
		inParam, err = p.parseFieldList(pkg, regParams)
		if err != nil {
			return nil, nil, nil, p.errorf(f.Pos(), "failed parsing arguments: %v", err)
		}
	}
	if f.Results != nil {
		outParam, err = p.parseFieldList(pkg, f.Results.List)
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
			var value string
			switch val := v.Len.(type) {
			case (*ast.BasicLit):
				value = val.Value
			case (*ast.Ident):
				// when the length is a const defined locally
				value = val.Obj.Decl.(*ast.ValueSpec).Values[0].(*ast.BasicLit).Value
			case (*ast.SelectorExpr):
				// when the length is a const defined in an external package
				usedPkg, err := importer.Default().Import(fmt.Sprintf("%s", val.X))
				if err != nil {
					return nil, p.errorf(v.Len.Pos(), "unknown package in array length: %v", err)
				}
				ev, err := types.Eval(token.NewFileSet(), usedPkg, token.NoPos, val.Sel.Name)
				if err != nil {
					return nil, p.errorf(v.Len.Pos(), "unknown constant in array length: %v", err)
				}
				value = ev.Value.String()
			}

			x, err := strconv.Atoi(value)
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
				pkg = maybeImportedPkg.Path()
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
		return &model.NamedType{Package: pkg.Path(), Type: v.Sel.String()}, nil
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
	case *ast.ParenExpr:
		return p.parseType(pkg, v.X)
	}

	return nil, fmt.Errorf("don't know how to parse type %T", typ)
}

// importsOfFile returns a map of package name to import path
// of the imports in file.
func importsOfFile(file *ast.File) (normalImports map[string]importedPackage, dotImports []string) {
	var importPaths []string
	for _, is := range file.Imports {
		if is.Name != nil {
			continue
		}
		importPath := is.Path.Value[1 : len(is.Path.Value)-1] // remove quotes
		importPaths = append(importPaths, importPath)
	}
	packagesName := createPackageMap(importPaths)
	normalImports = make(map[string]importedPackage)
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
			if pkg, ok := normalImports[pkgName]; ok {
				switch p := pkg.(type) {
				case duplicateImport:
					normalImports[pkgName] = duplicateImport{
						name:       p.name,
						duplicates: append([]string{importPath}, p.duplicates...),
					}
				case importedPkg:
					normalImports[pkgName] = duplicateImport{
						name:       pkgName,
						duplicates: []string{p.path, importPath},
					}
				}
			} else {
				normalImports[pkgName] = importedPkg{path: importPath}
			}
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

	packageImport, err := parsePackageImport(srcDir)
	if err != nil {
		return "", err
	}
	return packageImport, nil
}

var errOutsideGoPath = errors.New("source directory is outside GOPATH")
