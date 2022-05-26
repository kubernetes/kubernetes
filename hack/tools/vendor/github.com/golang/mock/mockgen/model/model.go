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

// Package model contains the data model necessary for generating mock implementations.
package model

import (
	"encoding/gob"
	"fmt"
	"io"
	"reflect"
	"strings"
)

// pkgPath is the importable path for package model
const pkgPath = "github.com/golang/mock/mockgen/model"

// Package is a Go package. It may be a subset.
type Package struct {
	Name       string
	PkgPath    string
	Interfaces []*Interface
	DotImports []string
}

// Print writes the package name and its exported interfaces.
func (pkg *Package) Print(w io.Writer) {
	_, _ = fmt.Fprintf(w, "package %s\n", pkg.Name)
	for _, intf := range pkg.Interfaces {
		intf.Print(w)
	}
}

// Imports returns the imports needed by the Package as a set of import paths.
func (pkg *Package) Imports() map[string]bool {
	im := make(map[string]bool)
	for _, intf := range pkg.Interfaces {
		intf.addImports(im)
	}
	return im
}

// Interface is a Go interface.
type Interface struct {
	Name    string
	Methods []*Method
}

// Print writes the interface name and its methods.
func (intf *Interface) Print(w io.Writer) {
	_, _ = fmt.Fprintf(w, "interface %s\n", intf.Name)
	for _, m := range intf.Methods {
		m.Print(w)
	}
}

func (intf *Interface) addImports(im map[string]bool) {
	for _, m := range intf.Methods {
		m.addImports(im)
	}
}

// AddMethod adds a new method, de-duplicating by method name.
func (intf *Interface) AddMethod(m *Method) {
	for _, me := range intf.Methods {
		if me.Name == m.Name {
			return
		}
	}
	intf.Methods = append(intf.Methods, m)
}

// Method is a single method of an interface.
type Method struct {
	Name     string
	In, Out  []*Parameter
	Variadic *Parameter // may be nil
}

// Print writes the method name and its signature.
func (m *Method) Print(w io.Writer) {
	_, _ = fmt.Fprintf(w, "  - method %s\n", m.Name)
	if len(m.In) > 0 {
		_, _ = fmt.Fprintf(w, "    in:\n")
		for _, p := range m.In {
			p.Print(w)
		}
	}
	if m.Variadic != nil {
		_, _ = fmt.Fprintf(w, "    ...:\n")
		m.Variadic.Print(w)
	}
	if len(m.Out) > 0 {
		_, _ = fmt.Fprintf(w, "    out:\n")
		for _, p := range m.Out {
			p.Print(w)
		}
	}
}

func (m *Method) addImports(im map[string]bool) {
	for _, p := range m.In {
		p.Type.addImports(im)
	}
	if m.Variadic != nil {
		m.Variadic.Type.addImports(im)
	}
	for _, p := range m.Out {
		p.Type.addImports(im)
	}
}

// Parameter is an argument or return parameter of a method.
type Parameter struct {
	Name string // may be empty
	Type Type
}

// Print writes a method parameter.
func (p *Parameter) Print(w io.Writer) {
	n := p.Name
	if n == "" {
		n = `""`
	}
	_, _ = fmt.Fprintf(w, "    - %v: %v\n", n, p.Type.String(nil, ""))
}

// Type is a Go type.
type Type interface {
	String(pm map[string]string, pkgOverride string) string
	addImports(im map[string]bool)
}

func init() {
	gob.Register(&ArrayType{})
	gob.Register(&ChanType{})
	gob.Register(&FuncType{})
	gob.Register(&MapType{})
	gob.Register(&NamedType{})
	gob.Register(&PointerType{})

	// Call gob.RegisterName to make sure it has the consistent name registered
	// for both gob decoder and encoder.
	//
	// For a non-pointer type, gob.Register will try to get package full path by
	// calling rt.PkgPath() for a name to register. If your project has vendor
	// directory, it is possible that PkgPath will get a path like this:
	//     ../../../vendor/github.com/golang/mock/mockgen/model
	gob.RegisterName(pkgPath+".PredeclaredType", PredeclaredType(""))
}

// ArrayType is an array or slice type.
type ArrayType struct {
	Len  int // -1 for slices, >= 0 for arrays
	Type Type
}

func (at *ArrayType) String(pm map[string]string, pkgOverride string) string {
	s := "[]"
	if at.Len > -1 {
		s = fmt.Sprintf("[%d]", at.Len)
	}
	return s + at.Type.String(pm, pkgOverride)
}

func (at *ArrayType) addImports(im map[string]bool) { at.Type.addImports(im) }

// ChanType is a channel type.
type ChanType struct {
	Dir  ChanDir // 0, 1 or 2
	Type Type
}

func (ct *ChanType) String(pm map[string]string, pkgOverride string) string {
	s := ct.Type.String(pm, pkgOverride)
	if ct.Dir == RecvDir {
		return "<-chan " + s
	}
	if ct.Dir == SendDir {
		return "chan<- " + s
	}
	return "chan " + s
}

func (ct *ChanType) addImports(im map[string]bool) { ct.Type.addImports(im) }

// ChanDir is a channel direction.
type ChanDir int

// Constants for channel directions.
const (
	RecvDir ChanDir = 1
	SendDir ChanDir = 2
)

// FuncType is a function type.
type FuncType struct {
	In, Out  []*Parameter
	Variadic *Parameter // may be nil
}

func (ft *FuncType) String(pm map[string]string, pkgOverride string) string {
	args := make([]string, len(ft.In))
	for i, p := range ft.In {
		args[i] = p.Type.String(pm, pkgOverride)
	}
	if ft.Variadic != nil {
		args = append(args, "..."+ft.Variadic.Type.String(pm, pkgOverride))
	}
	rets := make([]string, len(ft.Out))
	for i, p := range ft.Out {
		rets[i] = p.Type.String(pm, pkgOverride)
	}
	retString := strings.Join(rets, ", ")
	if nOut := len(ft.Out); nOut == 1 {
		retString = " " + retString
	} else if nOut > 1 {
		retString = " (" + retString + ")"
	}
	return "func(" + strings.Join(args, ", ") + ")" + retString
}

func (ft *FuncType) addImports(im map[string]bool) {
	for _, p := range ft.In {
		p.Type.addImports(im)
	}
	if ft.Variadic != nil {
		ft.Variadic.Type.addImports(im)
	}
	for _, p := range ft.Out {
		p.Type.addImports(im)
	}
}

// MapType is a map type.
type MapType struct {
	Key, Value Type
}

func (mt *MapType) String(pm map[string]string, pkgOverride string) string {
	return "map[" + mt.Key.String(pm, pkgOverride) + "]" + mt.Value.String(pm, pkgOverride)
}

func (mt *MapType) addImports(im map[string]bool) {
	mt.Key.addImports(im)
	mt.Value.addImports(im)
}

// NamedType is an exported type in a package.
type NamedType struct {
	Package string // may be empty
	Type    string
}

func (nt *NamedType) String(pm map[string]string, pkgOverride string) string {
	if pkgOverride == nt.Package {
		return nt.Type
	}
	prefix := pm[nt.Package]
	if prefix != "" {
		return prefix + "." + nt.Type
	}

	return nt.Type
}

func (nt *NamedType) addImports(im map[string]bool) {
	if nt.Package != "" {
		im[nt.Package] = true
	}
}

// PointerType is a pointer to another type.
type PointerType struct {
	Type Type
}

func (pt *PointerType) String(pm map[string]string, pkgOverride string) string {
	return "*" + pt.Type.String(pm, pkgOverride)
}
func (pt *PointerType) addImports(im map[string]bool) { pt.Type.addImports(im) }

// PredeclaredType is a predeclared type such as "int".
type PredeclaredType string

func (pt PredeclaredType) String(map[string]string, string) string { return string(pt) }
func (pt PredeclaredType) addImports(map[string]bool)              {}

// The following code is intended to be called by the program generated by ../reflect.go.

// InterfaceFromInterfaceType returns a pointer to an interface for the
// given reflection interface type.
func InterfaceFromInterfaceType(it reflect.Type) (*Interface, error) {
	if it.Kind() != reflect.Interface {
		return nil, fmt.Errorf("%v is not an interface", it)
	}
	intf := &Interface{}

	for i := 0; i < it.NumMethod(); i++ {
		mt := it.Method(i)
		// TODO: need to skip unexported methods? or just raise an error?
		m := &Method{
			Name: mt.Name,
		}

		var err error
		m.In, m.Variadic, m.Out, err = funcArgsFromType(mt.Type)
		if err != nil {
			return nil, err
		}

		intf.AddMethod(m)
	}

	return intf, nil
}

// t's Kind must be a reflect.Func.
func funcArgsFromType(t reflect.Type) (in []*Parameter, variadic *Parameter, out []*Parameter, err error) {
	nin := t.NumIn()
	if t.IsVariadic() {
		nin--
	}
	var p *Parameter
	for i := 0; i < nin; i++ {
		p, err = parameterFromType(t.In(i))
		if err != nil {
			return
		}
		in = append(in, p)
	}
	if t.IsVariadic() {
		p, err = parameterFromType(t.In(nin).Elem())
		if err != nil {
			return
		}
		variadic = p
	}
	for i := 0; i < t.NumOut(); i++ {
		p, err = parameterFromType(t.Out(i))
		if err != nil {
			return
		}
		out = append(out, p)
	}
	return
}

func parameterFromType(t reflect.Type) (*Parameter, error) {
	tt, err := typeFromType(t)
	if err != nil {
		return nil, err
	}
	return &Parameter{Type: tt}, nil
}

var errorType = reflect.TypeOf((*error)(nil)).Elem()

var byteType = reflect.TypeOf(byte(0))

func typeFromType(t reflect.Type) (Type, error) {
	// Hack workaround for https://golang.org/issue/3853.
	// This explicit check should not be necessary.
	if t == byteType {
		return PredeclaredType("byte"), nil
	}

	if imp := t.PkgPath(); imp != "" {
		return &NamedType{
			Package: impPath(imp),
			Type:    t.Name(),
		}, nil
	}

	// only unnamed or predeclared types after here

	// Lots of types have element types. Let's do the parsing and error checking for all of them.
	var elemType Type
	switch t.Kind() {
	case reflect.Array, reflect.Chan, reflect.Map, reflect.Ptr, reflect.Slice:
		var err error
		elemType, err = typeFromType(t.Elem())
		if err != nil {
			return nil, err
		}
	}

	switch t.Kind() {
	case reflect.Array:
		return &ArrayType{
			Len:  t.Len(),
			Type: elemType,
		}, nil
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128, reflect.String:
		return PredeclaredType(t.Kind().String()), nil
	case reflect.Chan:
		var dir ChanDir
		switch t.ChanDir() {
		case reflect.RecvDir:
			dir = RecvDir
		case reflect.SendDir:
			dir = SendDir
		}
		return &ChanType{
			Dir:  dir,
			Type: elemType,
		}, nil
	case reflect.Func:
		in, variadic, out, err := funcArgsFromType(t)
		if err != nil {
			return nil, err
		}
		return &FuncType{
			In:       in,
			Out:      out,
			Variadic: variadic,
		}, nil
	case reflect.Interface:
		// Two special interfaces.
		if t.NumMethod() == 0 {
			return PredeclaredType("interface{}"), nil
		}
		if t == errorType {
			return PredeclaredType("error"), nil
		}
	case reflect.Map:
		kt, err := typeFromType(t.Key())
		if err != nil {
			return nil, err
		}
		return &MapType{
			Key:   kt,
			Value: elemType,
		}, nil
	case reflect.Ptr:
		return &PointerType{
			Type: elemType,
		}, nil
	case reflect.Slice:
		return &ArrayType{
			Len:  -1,
			Type: elemType,
		}, nil
	case reflect.Struct:
		if t.NumField() == 0 {
			return PredeclaredType("struct{}"), nil
		}
	}

	// TODO: Struct, UnsafePointer
	return nil, fmt.Errorf("can't yet turn %v (%v) into a model.Type", t, t.Kind())
}

// impPath sanitizes the package path returned by `PkgPath` method of a reflect Type so that
// it is importable. PkgPath might return a path that includes "vendor". These paths do not
// compile, so we need to remove everything up to and including "/vendor/".
// See https://github.com/golang/go/issues/12019.
func impPath(imp string) string {
	if strings.HasPrefix(imp, "vendor/") {
		imp = "/" + imp
	}
	if i := strings.LastIndex(imp, "/vendor/"); i != -1 {
		imp = imp[i+len("/vendor/"):]
	}
	return imp
}

// ErrorInterface represent built-in error interface.
var ErrorInterface = Interface{
	Name: "error",
	Methods: []*Method{
		{
			Name: "Error",
			Out: []*Parameter{
				{
					Name: "",
					Type: PredeclaredType("string"),
				},
			},
		},
	},
}
