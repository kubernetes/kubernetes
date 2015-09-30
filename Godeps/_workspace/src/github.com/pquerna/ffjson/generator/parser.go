/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package generator

import (
	"flag"
	"fmt"
	"github.com/pquerna/ffjson/shared"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"regexp"
	"strings"
)

var noEncoder = flag.Bool("noencoder", false, "Do not generate encoder functions")
var noDecoder = flag.Bool("nodecoder", false, "Do not generate decoder functions")

type StructField struct {
	Name string
}

type StructInfo struct {
	Name    string
	Options shared.StructOptions
}

func NewStructInfo(name string) *StructInfo {
	return &StructInfo{
		Name: name,
		Options: shared.StructOptions{
			SkipDecoder: *noDecoder,
			SkipEncoder: *noEncoder,
		},
	}
}

var skipre = regexp.MustCompile("(.*)ffjson:(\\s*)((skip)|(ignore))(.*)")
var skipdec = regexp.MustCompile("(.*)ffjson:(\\s*)((skipdecoder)|(nodecoder))(.*)")
var skipenc = regexp.MustCompile("(.*)ffjson:(\\s*)((skipencoder)|(noencoder))(.*)")

func shouldInclude(d *ast.Object) (bool, error) {
	ts, ok := d.Decl.(*ast.TypeSpec)
	if !ok {
		return false, fmt.Errorf("Unknown type without TypeSec: %v", d)
	}

	_, ok = ts.Type.(*ast.StructType)
	if !ok {
		ident, ok := ts.Type.(*ast.Ident)
		if !ok || ident.Name == "" {
			return false, nil
		}

		// It must be in this package, and not a pointer alias
		if strings.Contains(ident.Name, ".") || strings.Contains(ident.Name, "*") {
			return false, nil
		}

		// if Obj is nil, we have an external type or built-in.
		if ident.Obj == nil || ident.Obj.Decl == nil {
			return false, nil
		}
		return shouldInclude(ident.Obj)
	}
	return true, nil
}

func ExtractStructs(inputPath string) (string, []*StructInfo, error) {
	fset := token.NewFileSet()

	f, err := parser.ParseFile(fset, inputPath, nil, parser.ParseComments)

	if err != nil {
		return "", nil, err
	}

	packageName := f.Name.String()
	structs := make(map[string]*StructInfo)

	for k, d := range f.Scope.Objects {
		if d.Kind == ast.Typ {
			incl, err := shouldInclude(d)
			if err != nil {
				return "", nil, err
			}
			if incl {
				stobj := NewStructInfo(k)

				structs[k] = stobj
			}
		}
	}

	files := map[string]*ast.File{
		inputPath: f,
	}

	pkg, _ := ast.NewPackage(fset, files, nil, nil)

	d := doc.New(pkg, f.Name.String(), doc.AllDecls)
	for _, t := range d.Types {
		if skipre.MatchString(t.Doc) {
			delete(structs, t.Name)
		} else {
			if skipdec.MatchString(t.Doc) {
				s, ok := structs[t.Name]
				if ok {
					s.Options.SkipDecoder = true
				}
			}
			if skipenc.MatchString(t.Doc) {
				s, ok := structs[t.Name]
				if ok {
					s.Options.SkipEncoder = true
				}
			}
		}
	}

	rv := make([]*StructInfo, 0)
	for _, v := range structs {
		rv = append(rv, v)
	}
	return packageName, rv, nil
}
