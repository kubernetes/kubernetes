/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"unicode"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util"
)

/////////// This stuff would go in pkg/codegen/deep_visit.go

// The map argument tracks types that have already been seen.
func deepVisitType(t reflect.Type, parentType reflect.Type, fieldName string, visitor Visitor, visited map[reflect.Type]bool) {
	//defer makeUsefulPanic(v1)

	isNamedType := (t.PkgPath() != "")

	if isNamedType {
		if visited[t] {
			return
		}
		visited[t] = true
		visitor.OpenType(t)
	}

	switch t.Kind() {
	case reflect.Map:
		deepVisitType(t.Key(), nil, "", visitor, visited)
		fallthrough
	case reflect.Array, reflect.Slice, reflect.Ptr:
		deepVisitType(t.Elem(), nil, "", visitor, visited)
	case reflect.Struct:
		pfx := ""
		par := t
		if !isNamedType {
			// For anonymous structs we keep the parentType and expand the name
			// expression.  An alternative to this would be to make up an
			// anonymous type name, ensure it is unique, and pretend it was a
			// named type.
			pfx = fieldName + "."
			par = parentType
		}
		for i, n := 0, t.NumField(); i < n; i++ {
			if t.Field(i).PkgPath == "" {
				// Deal with exported fields only.
				fld := pfx + t.Field(i).Name
				deepVisitType(t.Field(i).Type, par, fld, visitor, visited)
				visitor.VisitField(t.Field(i), par, fld)
			}
		}
	}

	if isNamedType {
		visitor.CloseType(t)
	}
}

// Visitor is the interface that callers must implement to deep-visit a type.
type Visitor interface {
	OpenType(t reflect.Type)
	// FIXME: document when fieldName differs from f.Name
	VisitField(f reflect.StructField, parentType reflect.Type, fieldNameInParent string)
	CloseType(t reflect.Type)
}

// DeepVisitType visits all fields of a type, recursively.
func DeepVisitType(t reflect.Type, visitor Visitor) {
	if t == nil {
		return
	}
	DeepVisitTypeAccumulate(t, visitor, make(map[reflect.Type]bool))
}

// DeepVisitTypeAccumulate visits all fields of a type, recursively,
// accumulating a set of previously visited types.  This is usefule to visit a
// group of types without repeating work.
func DeepVisitTypeAccumulate(t reflect.Type, visitor Visitor, accumulator map[reflect.Type]bool) {
	if t == nil {
		return
	}
	deepVisitType(t, nil, "", visitor, accumulator)
}

/////////// This stuff would go in pkg/codegen/line_buffer.go

type LineBuffer []string

func (lb *LineBuffer) Append(str string) {
	*lb = append(*lb, str)
}

func (lb *LineBuffer) Appendf(format string, args ...interface{}) {
	*lb = append(*lb, fmt.Sprintf(format, args...))
}

func (lb *LineBuffer) Prepend(str string) {
	*lb = append([]string{str}, *lb...)
}

func (lb *LineBuffer) Prependf(format string, args ...interface{}) {
	*lb = append([]string{fmt.Sprintf(format, args...)}, *lb...)
}

func (lb *LineBuffer) Join(other LineBuffer) {
	*lb = append(*lb, other...)
}

func (lb LineBuffer) String() string {
	return strings.Join(lb, "\n") + "\n"
}

/////////// This stuff would go in pkg/codegen/expr.go

// A helper type for tracking whether an expression is a value or pointer.
type Expr struct {
	base   string
	derefs int // > 0 means to dereference, < 0 means to take the address.
}

func (e Expr) Deref() Expr {
	e.derefs++
	return e
}

func (e Expr) Addr() Expr {
	e.derefs--
	return e
}

func (e Expr) String() string {
	for ; e.derefs > 0; e.derefs-- {
		e.base = fmt.Sprintf("(*%s)", e.base)
	}
	for ; e.derefs < 0; e.derefs++ {
		e.base = fmt.Sprintf("(&%s)", e.base)
	}
	return e.base
}

func ptrExpr(base string) Expr {
	return Expr{base: fmt.Sprintf("(%s)", base), derefs: 1}
}

func valExpr(base string) Expr {
	return Expr{base: fmt.Sprintf("(%s)", base), derefs: 0}
}

/////////// This stuff would go in cmd/gendefaults/gendefaults.go

type typeRecord struct {
	buffer   LineBuffer
	pkgAlias string
	mustEmit bool
}

type myVisitor struct {
	index  []reflect.Type
	types  map[reflect.Type]*typeRecord
	outPkg string
}

func NewVisitor(outPkg string) *myVisitor {
	return &myVisitor{
		index:  []reflect.Type{},
		types:  map[reflect.Type]*typeRecord{},
		outPkg: outPkg,
	}
}

func implementsDefaulter(t reflect.Type) bool {
	x := reflect.New(t).Interface()
	_, ok := x.(defaulter)
	return ok
}

func (v *myVisitor) OpenType(t reflect.Type) {
	//fmt.Printf("// DBG: OpenType %q  %s\n", t.PkgPath(), t)
	v.types[t] = &typeRecord{
		buffer:   nil,
		pkgAlias: v.pkgSymbol(t),
	}
	if implementsDefaulter(t) {
		//fmt.Printf("// DBG: %v implements ApplyDefaults()\n", t)
		v.types[t].mustEmit = true
	}
	v.index = append(v.index, t)
}

func (v *myVisitor) CloseType(t reflect.Type) {
	//fmt.Printf("// DBG: CloseType %q  %s\n", t.PkgPath(), t)
	if implementsDefaulter(t) {
		v.types[t].buffer.Appendf("obj.ApplyDefaults()")
	}
}

// FIXME: would be nice to expose this somewhere, but circular deps hurt.
// Maybe pkg/api
type defaulter interface {
	ApplyDefaults()
}

func (v *myVisitor) VisitField(f reflect.StructField, parentType reflect.Type, fieldName string) {
	//fmt.Printf("// DBG: VisitField %v.%s type %v\n", parentType, fieldName, f.Type)
	buf := v.visitExpr(f.Type, parentType, valExpr(fmt.Sprintf("obj.%s", fieldName)))
	if buf != nil {
		buf.Prependf("// %s %v", fieldName, f.Type)
		v.types[parentType].buffer.Join(buf)
		v.types[parentType].mustEmit = true
	}
}

// Returns true if the value might have changed.
func (v *myVisitor) visitExpr(t reflect.Type, parentType reflect.Type, expr Expr) LineBuffer {
	if t.PkgPath() == "" {
		// Unpack unnamed aggregate types.
		switch t.Kind() {
		case reflect.Array, reflect.Slice:
			return v.visitArray(t, parentType, expr)
		case reflect.Ptr:
			return v.visitPtr(t, parentType, expr)
		case reflect.Map:
			return v.visitMap(t, parentType, expr)
		default:
			// Struct is handed by the visitor machinery.
			return nil
		}
	}

	if v.types[t].mustEmit {
		ret := LineBuffer{}
		ret.Appendf("applyDefaults_%s(%s)", v.pkgTypeSymbol(t), expr.Addr())
		return ret
	}
	return nil
}

func (v *myVisitor) visitArray(t reflect.Type, parentType reflect.Type, expr Expr) LineBuffer {
	buf := LineBuffer{}

	buf.Appendf("for i := range %s {", expr)
	buf.Appendf("  p := &(%s[i])", expr)
	childBuf := v.visitExpr(t.Elem(), parentType, ptrExpr("p"))
	if childBuf == nil {
		return nil
	}
	buf.Join(childBuf)
	buf.Appendf("}")

	return buf
}

func (v *myVisitor) visitPtr(t reflect.Type, parentType reflect.Type, expr Expr) LineBuffer {
	buf := LineBuffer{}

	buf.Appendf("if (%s) != nil {", expr)
	childBuf := v.visitExpr(t.Elem(), parentType, expr.Deref())
	if childBuf == nil {
		return nil
	}
	buf.Join(childBuf)
	buf.Appendf("}")

	return buf
}

func (v *myVisitor) visitMap(t reflect.Type, parentType reflect.Type, expr Expr) LineBuffer {
	buf := LineBuffer{}

	buf.Appendf("for k, v := range %s {", expr)
	childBuf := v.visitExpr(t.Elem(), parentType, valExpr("v"))
	if childBuf == nil {
		return nil
	}
	buf.Join(childBuf)
	buf.Appendf("    %s[k] = v", expr)
	buf.Appendf("}")

	return buf
}

func pathToSymbol(path string) string {
	// At the cost of some ugly, the leading '_' ensures that any string we get
	// (even if it starts with a digit) becomes a valid symbol.
	out := bytes.NewBuffer([]byte("_"))
	for _, r := range path {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			out.WriteRune(r)
		} else {
			out.WriteRune('_')
		}
	}
	return out.String()
}

func (v *myVisitor) pkgSymbol(t reflect.Type) string {
	return pathToSymbol(t.PkgPath())
}

func (v *myVisitor) pkgTypeSymbol(t reflect.Type) string {
	return fmt.Sprintf("%s_%s", v.pkgSymbol(t), t.Name())
}

func (v *myVisitor) typeString(t reflect.Type) string {
	pkg := v.types[t].pkgAlias
	tn := t.Name()
	if pkg != "" {
		return pkg + "." + tn
	}
	return t.Name()
}

func (v *myVisitor) Dump() {
	buf := LineBuffer{}

	// Emit package line.
	buf.Appendf("package %s", v.outPkg)

	// Emit import lines.
	imports := util.NewStringSet()
	for _, t := range v.index {
		if v.types[t].buffer == nil {
			continue
		}
		imports.Insert(fmt.Sprintf("%s %q\n", v.types[t].pkgAlias, t.PkgPath()))
	}
	for i := range imports {
		buf.Appendf("import %s", i)
	}

	// Emit per-type functions.
	for _, t := range v.index {
		if v.types[t].buffer == nil {
			continue
		}
		ts := v.typeString(t)
		ss := v.pkgTypeSymbol(t)
		buf.Appendf("func applyDefaults_%s(obj *%s) {", ss, ts)
		buf.Join(v.types[t].buffer)
		buf.Append("}")
	}

	// Emit top-level entrypoints.
	//FIXME: only register for registered API Objects or for all?
	for _, t := range v.index {
		if v.types[t].buffer == nil {
			continue
		}
		ts := v.typeString(t)
		ss := v.pkgTypeSymbol(t)
		buf.Appendf("func applyDefaults_entry_%s(obj interface{}) {", ss)
		buf.Appendf("    concrete, ok := obj.(*%s)", ts)
		//FIXME: maybe don't bother to panic, just let the assertion do it
		buf.Appendf("    if !ok { panic(fmt.Sprintf(\"applyDefaults_entry_%s called for type %%T\", obj)) }", ss)
		buf.Appendf("    applyDefaults_%s(concrete)", ss)
		buf.Append("}\n")
	}

	// Register entrypoints.
	buf.Append("func InitDefaultableTypes(register func(reflect.Type, func(obj interface{}))) {")
	for _, t := range v.index {
		if v.types[t].buffer == nil {
			continue
		}
		ts := v.typeString(t)
		buf.Append("register(")
		buf.Appendf("    reflect.TypeOf(new(%s)),", ts)
		buf.Appendf("    applyDefaults_entry_%s)", v.pkgTypeSymbol(t))
	}
	buf.Append("}")

	fmt.Print(buf)
}

func main() {
	// The assumption is that you generate into a new pkg.
	visitor := NewVisitor("generated")
	acc := map[reflect.Type]bool{}
	//FIXME: need to sort input types for stable output
	for _, knownType := range api.Scheme.KnownTypes("v1") {
		DeepVisitTypeAccumulate(knownType, visitor, acc)
	}
	visitor.Dump()
}
