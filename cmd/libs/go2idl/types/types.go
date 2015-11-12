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

package types

// A type name may have a package qualifier.
type Name struct {
	// Empty if embedded or builtin. This is the package path.
	Package string
	// The type name.
	Name string
}

// String returns the name formatted as a string.
func (n Name) String() string {
	if n.Package == "" {
		return n.Name
	}
	return n.Package + "." + n.Name
}

// The possible classes of types.
type Kind string

const (
	// Builtin is a primitive, like bool, string, int.
	Builtin Kind = "Builtin"
	Struct  Kind = "Struct"
	Map     Kind = "Map"
	Slice   Kind = "Slice"
	Pointer Kind = "Pointer"

	// Alias is an alias of another type, e.g. in:
	//   type Foo string
	//   type Bar Foo
	// Bar is an alias of Foo.
	//
	// In the real go type system, Foo is a "Named" string; but to simplify
	// generation, this type system will just say that Foo *is* a builtin.
	// We then need "Alias" as a way for us to say that Bar *is* a Foo.
	Alias Kind = "Alias"

	// Interface is any type that could have differing types at run time.
	Interface Kind = "Interface"

	// The remaining types are included for completeness, but are not well
	// supported.
	Array       Kind = "Array" // Array is just like slice, but has a fixed length.
	Chan        Kind = "Chan"
	Func        Kind = "Func"
	Unknown     Kind = ""
	Unsupported Kind = "Unsupported"
)

// Package holds package-level information.
// Fields are public, as everything in this package, to enable consumption by
// templates (for example). But it is strongly encouraged for code to build by
// using the provided functions.
type Package struct {
	// Canonical name of this package-- its path.
	Path string

	// Short name of this package; the name that appears in the
	// 'package x' line.
	Name string

	// Types within this package, indexed by their name (*not* including
	// package name).
	Types map[string]*Type

	// Packages imported by this package, indexed by (canonicalized)
	// package path.
	Imports map[string]*Package
}

// Has returns true if the given name references a type known to this package.
func (p *Package) Has(name string) bool {
	_, has := p.Types[name]
	return has
}

// Get (or add) the given type
func (p *Package) Get(typeName string) *Type {
	if t, ok := p.Types[typeName]; ok {
		return t
	}
	if p.Path == "" {
		// Import the standard builtin types!
		if t, ok := builtins.Types[typeName]; ok {
			p.Types[typeName] = t
			return t
		}
	}
	t := &Type{Name: Name{p.Path, typeName}}
	p.Types[typeName] = t
	return t
}

// HasImport returns true if p imports packageName. Package names include the
// package directory.
func (p *Package) HasImport(packageName string) bool {
	_, has := p.Imports[packageName]
	return has
}

// Universe is a map of all packages. The key is the package name, but you
// should use Get() or Package() instead of direct access.
type Universe map[string]*Package

// Get returns the canonical type for the given fully-qualified name. Builtin
// types will always be found, even if they haven't been explicitly added to
// the map. If a non-existing type is requested, u will create (a marker for)
// it.
func (u Universe) Get(n Name) *Type {
	return u.Package(n.Package).Get(n.Name)
}

// AddImports registers import lines for packageName. May be called multiple times.
// You are responsible for canonicalizing all package paths.
func (u Universe) AddImports(packagePath string, importPaths ...string) {
	p := u.Package(packagePath)
	for _, i := range importPaths {
		p.Imports[i] = u.Package(i)
	}
}

// Get (create if needed) the package.
func (u Universe) Package(packagePath string) *Package {
	if p, ok := u[packagePath]; ok {
		return p
	}
	p := &Package{
		Path:    packagePath,
		Types:   map[string]*Type{},
		Imports: map[string]*Package{},
	}
	u[packagePath] = p
	return p
}

// Type represents a subset of possible go types.
type Type struct {
	// There are two general categories of types, those explicitly named
	// and those anonymous. Named ones will have a non-empty package in the
	// name field.
	Name Name

	// The general kind of this type.
	Kind Kind

	// If there are comment lines immediately before the type definition,
	// they will be recorded here.
	CommentLines string

	// If Kind == Struct
	Members []Member

	// If Kind == Map, Slice, Pointer, or Chan
	Elem *Type

	// If Kind == Map, this is the map's key type.
	Key *Type

	// If Kind == Alias, this is the underlying type.
	Underlying *Type

	// If Kind == Interface, this is the list of all required functions.
	// Otherwise, if this is a named type, this is the list of methods that
	// type has. (All elements will have Kind=="Func")
	Methods []*Type

	// If Kind == func, this is the signature of the function.
	Signature *Signature

	// TODO: Add:
	// * channel direction
	// * array length
}

// String returns the name of the type.
func (t *Type) String() string {
	return t.Name.String()
}

// IsAssignable returns whether the type is assignable.
func (t *Type) IsAssignable() bool {
	return t.Kind == Builtin || (t.Kind == Alias && t.Underlying.Kind == Builtin)
}

// A single struct member
type Member struct {
	// The name of the member.
	Name string

	// If the member is embedded (anonymous) this will be true, and the
	// Name will be the type name.
	Embedded bool

	// If there are comment lines immediately before the member in the type
	// definition, they will be recorded here.
	CommentLines string

	// If there are tags along with this member, they will be saved here.
	Tags string

	// The type of this member.
	Type *Type
}

// String returns the name and type of the member.
func (m Member) String() string {
	return m.Name + " " + m.Type.String()
}

// Signature is a function's signature.
type Signature struct {
	// TODO: store the parameter names, not just types.

	// If a method of some type, this is the type it's a member of.
	Receiver   *Type
	Parameters []*Type
	Results    []*Type

	// True if the last in parameter is of the form ...T.
	Variadic bool

	// If there are comment lines immediately before this
	// signature/method/function declaration, they will be recorded here.
	CommentLines string
}

// Built in types.
var (
	String = &Type{
		Name: Name{Name: "string"},
		Kind: Builtin,
	}
	Int64 = &Type{
		Name: Name{Name: "int64"},
		Kind: Builtin,
	}
	Int32 = &Type{
		Name: Name{Name: "int32"},
		Kind: Builtin,
	}
	Int16 = &Type{
		Name: Name{Name: "int16"},
		Kind: Builtin,
	}
	Int = &Type{
		Name: Name{Name: "int"},
		Kind: Builtin,
	}
	Uint64 = &Type{
		Name: Name{Name: "uint64"},
		Kind: Builtin,
	}
	Uint32 = &Type{
		Name: Name{Name: "uint32"},
		Kind: Builtin,
	}
	Uint16 = &Type{
		Name: Name{Name: "uint16"},
		Kind: Builtin,
	}
	Uint = &Type{
		Name: Name{Name: "uint"},
		Kind: Builtin,
	}
	Bool = &Type{
		Name: Name{Name: "bool"},
		Kind: Builtin,
	}
	Byte = &Type{
		Name: Name{Name: "byte"},
		Kind: Builtin,
	}

	builtins = &Package{
		Types: map[string]*Type{
			"bool":   Bool,
			"string": String,
			"int":    Int,
			"int64":  Int64,
			"int32":  Int32,
			"int16":  Int16,
			"int8":   Byte,
			"uint":   Uint,
			"uint64": Uint64,
			"uint32": Uint32,
			"uint16": Uint16,
			"uint8":  Byte,
			"byte":   Byte,
		},
		Imports: map[string]*Package{},
		Path:    "",
		Name:    "",
	}
)
