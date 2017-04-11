/*
Copyright 2015 The Kubernetes Authors.

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

package generator

import (
	"bytes"
	"io"

	"k8s.io/gengo/namer"
	"k8s.io/gengo/parser"
	"k8s.io/gengo/types"
)

// Package contains the contract for generating a package.
type Package interface {
	// Name returns the package short name.
	Name() string
	// Path returns the package import path.
	Path() string

	// Filter should return true if this package cares about this type.
	// Otherwise, this type will be omitted from the type ordering for
	// this package.
	Filter(*Context, *types.Type) bool

	// Header should return a header for the file, including comment markers.
	// Useful for copyright notices and doc strings. Include an
	// autogeneration notice! Do not include the "package x" line.
	Header(filename string) []byte

	// Generators returns the list of generators for this package. It is
	// allowed for more than one generator to write to the same file.
	// A Context is passed in case the list of generators depends on the
	// input types.
	Generators(*Context) []Generator
}

type File struct {
	Name        string
	FileType    string
	PackageName string
	Header      []byte
	Imports     map[string]struct{}
	Vars        bytes.Buffer
	Consts      bytes.Buffer
	Body        bytes.Buffer
}

type FileType interface {
	AssembleFile(f *File, path string) error
	VerifyFile(f *File, path string) error
}

// Packages is a list of packages to generate.
type Packages []Package

// Generator is the contract for anything that wants to do auto-generation.
// It's expected that the io.Writers passed to the below functions will be
// ErrorTrackers; this allows implementations to not check for io errors,
// making more readable code.
//
// The call order for the functions that take a Context is:
// 1. Filter()        // Subsequent calls see only types that pass this.
// 2. Namers()        // Subsequent calls see the namers provided by this.
// 3. PackageVars()
// 4. PackageConsts()
// 5. Init()
// 6. GenerateType()  // Called N times, once per type in the context's Order.
// 7. Imports()
//
// You may have multiple generators for the same file.
type Generator interface {
	// The name of this generator. Will be included in generated comments.
	Name() string

	// Filter should return true if this generator cares about this type.
	// (otherwise, GenerateType will not be called.)
	//
	// Filter is called before any of the generator's other functions;
	// subsequent calls will get a context with only the types that passed
	// this filter.
	Filter(*Context, *types.Type) bool

	// If this generator needs special namers, return them here. These will
	// override the original namers in the context if there is a collision.
	// You may return nil if you don't need special names. These names will
	// be available in the context passed to the rest of the generator's
	// functions.
	//
	// A use case for this is to return a namer that tracks imports.
	Namers(*Context) namer.NameSystems

	// Init should write an init function, and any other content that's not
	// generated per-type. (It's not intended for generator specific
	// initialization! Do that when your Package constructs the
	// Generators.)
	Init(*Context, io.Writer) error

	// Finalize should write finish up functions, and any other content that's not
	// generated per-type.
	Finalize(*Context, io.Writer) error

	// PackageVars should emit an array of variable lines. They will be
	// placed in a var ( ... ) block. There's no need to include a leading
	// \t or trailing \n.
	PackageVars(*Context) []string

	// PackageConsts should emit an array of constant lines. They will be
	// placed in a const ( ... ) block. There's no need to include a leading
	// \t or trailing \n.
	PackageConsts(*Context) []string

	// GenerateType should emit the code for a particular type.
	GenerateType(*Context, *types.Type, io.Writer) error

	// Imports should return a list of necessary imports. They will be
	// formatted correctly. You do not need to include quotation marks,
	// return only the package name; alternatively, you can also return
	// imports in the format `name "path/to/pkg"`. Imports will be called
	// after Init, PackageVars, PackageConsts, and GenerateType, to allow
	// you to keep track of what imports you actually need.
	Imports(*Context) []string

	// Preferred file name of this generator, not including a path. It is
	// allowed for multiple generators to use the same filename, but it's
	// up to you to make sure they don't have colliding import names.
	// TODO: provide per-file import tracking, removing the requirement
	// that generators coordinate..
	Filename() string

	// A registered file type in the context to generate this file with. If
	// the FileType is not found in the context, execution will stop.
	FileType() string
}

// Context is global context for individual generators to consume.
type Context struct {
	// A map from the naming system to the names for that system. E.g., you
	// might have public names and several private naming systems.
	Namers namer.NameSystems

	// All the types, in case you want to look up something.
	Universe types.Universe

	// All the user-specified packages.  This is after recursive expansion.
	Inputs []string

	// The canonical ordering of the types (will be filtered by both the
	// Package's and Generator's Filter methods).
	Order []*types.Type

	// A set of types this context can process. If this is empty or nil,
	// the default "golang" filetype will be provided.
	FileTypes map[string]FileType

	// If true, Execute* calls will just verify that the existing output is
	// correct. (You may set this after calling NewContext.)
	Verify bool

	// Allows generators to add packages at runtime.
	builder *parser.Builder
}

// NewContext generates a context from the given builder, naming systems, and
// the naming system you wish to construct the canonical ordering from.
func NewContext(b *parser.Builder, nameSystems namer.NameSystems, canonicalOrderName string) (*Context, error) {
	universe, err := b.FindTypes()
	if err != nil {
		return nil, err
	}

	c := &Context{
		Namers:   namer.NameSystems{},
		Universe: universe,
		Inputs:   b.FindPackages(),
		FileTypes: map[string]FileType{
			GolangFileType: NewGolangFile(),
		},
		builder: b,
	}

	for name, systemNamer := range nameSystems {
		c.Namers[name] = systemNamer
		if name == canonicalOrderName {
			orderer := namer.Orderer{Namer: systemNamer}
			c.Order = orderer.OrderUniverse(universe)
		}
	}
	return c, nil
}

// AddDir adds a Go package to the context. The specified path must be a single
// go package import path.  GOPATH, GOROOT, and the location of your go binary
// (`which go`) will all be searched, in the normal Go fashion.
func (ctxt *Context) AddDir(path string) error {
	return ctxt.builder.AddDirTo(path, &ctxt.Universe)
}
