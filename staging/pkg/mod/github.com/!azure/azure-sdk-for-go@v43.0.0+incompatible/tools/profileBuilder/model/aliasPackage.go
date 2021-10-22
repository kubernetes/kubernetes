// +build go1.9

// Copyright 2018 Microsoft Corporation and contributors
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

// Package model holds the business logic for the operations made available by
// profileBuilder.
//
// This package is not governed by the SemVer associated with the rest of the
// Azure-SDK-for-Go.
package model

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"sort"
)

// AliasPackage is an abstraction around ast.Package to provide convenience methods for manipulating it.
type AliasPackage ast.Package

// ErrorUnexpectedToken is returned when AST parsing encounters an unexpected token, it includes the expected token.
type ErrorUnexpectedToken struct {
	Expected token.Token
	Received token.Token
}

var errUnexpectedNil = errors.New("unexpected nil")

func (utoken ErrorUnexpectedToken) Error() string {
	return fmt.Sprintf("Unexpected token %d expecting type: %d", utoken.Received, utoken.Expected)
}

const modelFile = "models.go"
const origImportAlias = "original"

// ModelFile is a getter for the file accumulating aliased content.
func (alias AliasPackage) ModelFile() *ast.File {
	if alias.Files != nil {
		return alias.Files[modelFile]
	}
	return nil
}

// NewAliasPackage creates an alias package from the specified input package.
// Parameter importPath is the import path specified to consume the package.
func NewAliasPackage(original *ast.Package, importPath string) (*AliasPackage, error) {
	models := &ast.File{
		Name: &ast.Ident{
			Name:    original.Name,
			NamePos: token.Pos(len("package") + 2),
		},
		Package: 1,
	}

	alias := &AliasPackage{
		Name: original.Name,
		Files: map[string]*ast.File{
			modelFile: models,
		},
	}

	models.Decls = append(models.Decls, &ast.GenDecl{
		Tok: token.IMPORT,
		Specs: []ast.Spec{
			&ast.ImportSpec{
				Name: &ast.Ident{
					Name: origImportAlias,
				},
				Path: &ast.BasicLit{
					Kind:  token.STRING,
					Value: fmt.Sprintf("\"%s\"", importPath),
				},
			},
		},
	})

	genDecls := []*ast.GenDecl{}
	funcDecls := []*ast.FuncDecl{}

	// node traversal is non-deterministic so we maintain a collection
	// that allows us to emit the nodes in a sort order of our choice
	ast.Inspect(original, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			// exclude methods as they're exposed on the aliased types
			if node.Recv == nil {
				funcDecls = append(funcDecls, node)
			}
			// return false as we don't care about the function body
			return false
		case *ast.GenDecl:
			genDecls = append(genDecls, node)
		}
		return true
	})

	// genDecls contains constants and type definitions.  group them so that
	// type defs for consts are next to their respective list of constants.

	type constType struct {
		name     string
		typeSpec *ast.GenDecl
		values   *ast.GenDecl
	}

	untypedConsts := []*ast.GenDecl{}
	constTypeMap := map[string]*constType{}

	// first build a map from all the constants
	for _, gd := range genDecls {
		if gd.Tok == token.CONST {
			// get the const type from the first item
			vs := gd.Specs[0].(*ast.ValueSpec)
			if vs.Type == nil {
				// untyped consts go first
				untypedConsts = append(untypedConsts, gd)
				continue
			}
			typeName := vs.Type.(*ast.Ident).Name
			constTypeMap[typeName] = &constType{
				name:   typeName,
				values: gd,
			}
		}
	}

	typeSpecs := []*ast.GenDecl{}

	// now update the map with the type specs
	for _, gd := range genDecls {
		if gd.Tok == token.TYPE {
			spec := gd.Specs[0].(*ast.TypeSpec)
			// check if the typespec is in the map, if it is it's for a constant
			if typeMap, ok := constTypeMap[spec.Name.Name]; ok {
				typeMap.typeSpec = gd
			} else {
				typeSpecs = append(typeSpecs, gd)
			}
		}
	}

	// add consts, types, and funcs, in that order, in sorted order

	sort.SliceStable(untypedConsts, func(i, j int) bool {
		tsI := untypedConsts[i].Specs[0].(*ast.TypeSpec)
		tsJ := untypedConsts[j].Specs[0].(*ast.TypeSpec)
		return tsI.Name.Name < tsJ.Name.Name
	})
	for _, uc := range untypedConsts {
		err := alias.AddConst(uc)
		if err != nil {
			return nil, err
		}
	}

	// convert to slice for sorting
	constDecls := []*constType{}
	for _, v := range constTypeMap {
		constDecls = append(constDecls, v)
	}
	sort.SliceStable(constDecls, func(i, j int) bool {
		return constDecls[i].name < constDecls[j].name
	})
	for _, cd := range constDecls {
		err := alias.AddType(cd.typeSpec)
		if err != nil {
			return nil, err
		}
		err = alias.AddConst(cd.values)
		if err != nil {
			return nil, err
		}
	}

	// now do the typespecs
	sort.SliceStable(typeSpecs, func(i, j int) bool {
		tsI := typeSpecs[i].Specs[0].(*ast.TypeSpec)
		tsJ := typeSpecs[j].Specs[0].(*ast.TypeSpec)
		return tsI.Name.Name < tsJ.Name.Name
	})
	for _, td := range typeSpecs {
		err := alias.AddType(td)
		if err != nil {
			return nil, err
		}
	}

	// funcs
	sort.SliceStable(funcDecls, func(i, j int) bool {
		return funcDecls[i].Name.Name < funcDecls[j].Name.Name
	})
	for _, fd := range funcDecls {
		err := alias.AddFunc(fd)
		if err != nil {
			return nil, err
		}
	}

	return alias, nil
}

// AddGeneral handles dispatching a GenDecl to either AddConst or AddType.
func (alias *AliasPackage) AddGeneral(decl *ast.GenDecl) error {
	var adder func(*ast.GenDecl) error

	switch decl.Tok {
	case token.CONST:
		adder = alias.AddConst
	case token.TYPE:
		adder = alias.AddType
	default:
		adder = func(item *ast.GenDecl) error {
			return fmt.Errorf("Unusable token: %v", item.Tok)
		}
	}

	return adder(decl)
}

// AddConst adds a Const block with indiviual aliases for each Spec in `decl`.
func (alias *AliasPackage) AddConst(decl *ast.GenDecl) error {
	if decl == nil {
		return errors.New("unexpected nil")
	} else if decl.Tok != token.CONST {
		return ErrorUnexpectedToken{Expected: token.CONST, Received: decl.Tok}
	}

	targetFile := alias.ModelFile()

	for _, spec := range decl.Specs {
		cast := spec.(*ast.ValueSpec)
		for j, name := range cast.Names {
			cast.Values[j] = &ast.SelectorExpr{
				X: &ast.Ident{
					Name: origImportAlias,
				},
				Sel: &ast.Ident{
					Name: name.Name,
				},
			}
		}
	}

	targetFile.Decls = append(targetFile.Decls, decl)
	return nil
}

// AddType adds a Type delcaration block with individual alias for each Spec handed in `decl`
func (alias *AliasPackage) AddType(decl *ast.GenDecl) error {
	if decl == nil {
		return errUnexpectedNil
	} else if decl.Tok != token.TYPE {
		return ErrorUnexpectedToken{Expected: token.TYPE, Received: decl.Tok}
	}

	targetFile := alias.ModelFile()

	for _, spec := range decl.Specs {
		cast := spec.(*ast.TypeSpec)
		cast.Assign = 1
		cast.Type = &ast.SelectorExpr{
			X: &ast.Ident{
				Name: origImportAlias,
			},
			Sel: &ast.Ident{
				Name: cast.Name.Name,
			},
		}
	}

	targetFile.Decls = append(targetFile.Decls, decl)
	return nil
}

// AddFunc creates a stub method to redirect the call to the original package, then adds it to the model file.
func (alias *AliasPackage) AddFunc(decl *ast.FuncDecl) error {
	if decl == nil {
		return errUnexpectedNil
	}

	arguments := []ast.Expr{}
	for _, p := range decl.Type.Params.List {
		arguments = append(arguments, p.Names[0])
	}

	decl.Body = &ast.BlockStmt{
		List: []ast.Stmt{
			&ast.ReturnStmt{
				Results: []ast.Expr{
					&ast.CallExpr{
						Fun: &ast.SelectorExpr{
							X: &ast.Ident{
								Name: origImportAlias,
							},
							Sel: &ast.Ident{
								Name: decl.Name.Name,
							},
						},
						Args: arguments,
					},
				},
			},
		},
	}

	targetFile := alias.ModelFile()
	targetFile.Decls = append(targetFile.Decls, decl)
	return nil
}
