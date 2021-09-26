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
	"go/ast"
	"strings"
)

// Content defines the set of exported constants, funcs, and structs.
type Content struct {
	// the list of exported constants.
	// key is the exported name, value is its type and value.
	Consts map[string]Const `json:"consts,omitempty"`

	// the list of exported functions and methods.
	// key is the exported name, for methods it's prefixed with the receiver type (e.g. "Type.Method").
	// value contains the list of params and return types.
	Funcs map[string]Func `json:"funcs,omitempty"`

	// the list of exported interfaces.
	// key is the exported name, value contains the interface definition.
	Interfaces map[string]Interface `json:"interfaces,omitempty"`

	// the list of exported struct types.
	// key is the exported name, value contains field information.
	Structs map[string]Struct `json:"structs,omitempty"`
}

// Const is a const definition.
type Const struct {
	// the type of the constant
	Type string `json:"type"`

	// the value of the constant
	Value string `json:"value"`
}

// Func contains parameter and return types of a function/method.
type Func struct {
	// a comma-delimited list of the param types
	Params *string `json:"params,omitempty"`

	// a comma-delimited list of the return types
	Returns *string `json:"returns,omitempty"`
}

// Interface contains the list of methods for an interface.
type Interface struct {
	Methods map[string]Func
}

// Struct contains field info about a struct.
type Struct struct {
	// a list of anonymous fields
	AnonymousFields []string `json:"anon,omitempty"`

	// key/value pairs of the field names and types respectively.
	Fields map[string]string `json:"fields,omitempty"`
}

// NewContent returns an initialized Content object.
func NewContent() Content {
	return Content{
		Consts:     make(map[string]Const),
		Funcs:      make(map[string]Func),
		Interfaces: make(map[string]Interface),
		Structs:    make(map[string]Struct),
	}
}

// IsEmpty returns true if there is no content in any of the fields.
func (c Content) IsEmpty() bool {
	return len(c.Consts) == 0 && len(c.Funcs) == 0 && len(c.Interfaces) == 0 && len(c.Structs) == 0
}

// adds the specified const declaration to the exports list
func (c *Content) addConst(pkg Package, g *ast.GenDecl) {
	for _, s := range g.Specs {
		co := Const{}
		vs := s.(*ast.ValueSpec)
		v := ""
		// Type is nil for untyped consts
		if vs.Type != nil {
			co.Type = vs.Type.(*ast.Ident).Name
			v = vs.Values[0].(*ast.BasicLit).Value
		} else {
			// get the type from the token type
			if bl, ok := vs.Values[0].(*ast.BasicLit); ok {
				co.Type = strings.ToLower(bl.Kind.String())
				v = bl.Value
			} else if ce, ok := vs.Values[0].(*ast.CallExpr); ok {
				// const FooConst = FooType("value")
				co.Type = pkg.getText(ce.Fun.Pos(), ce.Fun.End())
				v = pkg.getText(ce.Args[0].Pos(), ce.Args[0].End())
			} else {
				panic("unhandled case for adding constant")
			}
		}
		// remove any surrounding quotes
		if v[0] == '"' {
			v = v[1 : len(v)-1]
		}
		co.Value = v
		c.Consts[vs.Names[0].Name] = co
	}
}

// adds the specified function declaration to the exports list
func (c *Content) addFunc(pkg Package, f *ast.FuncDecl) {
	// create a method sig, for methods it's a combination of the receiver type
	// with the function name e.g. "FooReceiver.Method", else just the function name.
	sig := ""
	if f.Recv != nil {
		sig = pkg.getText(f.Recv.List[0].Type.Pos(), f.Recv.List[0].Type.End())
		sig += "."
	}
	sig += f.Name.Name
	c.Funcs[sig] = pkg.buildFunc(f.Type)
}

// adds the specified interface type to the exports list.
func (c *Content) addInterface(pkg Package, name string, i *ast.InterfaceType) {
	in := Interface{Methods: map[string]Func{}}
	if i.Methods != nil {
		for _, m := range i.Methods.List {
			n := m.Names[0].Name
			f := pkg.buildFunc(m.Type.(*ast.FuncType))
			in.Methods[n] = f
		}
	}
	c.Interfaces[name] = in
}

// adds the specified struct type to the exports list.
func (c *Content) addStruct(pkg Package, name string, s *ast.StructType) {
	sd := Struct{}
	// assumes all struct types have fields
	pkg.translateFieldList(s.Fields.List, func(n *string, t string) {
		if n == nil {
			sd.AnonymousFields = append(sd.AnonymousFields, t)
		} else {
			if sd.Fields == nil {
				sd.Fields = map[string]string{}
			}
			sd.Fields[*n] = t
		}
	})
	c.Structs[name] = sd
}
