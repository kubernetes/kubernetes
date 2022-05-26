// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package buildir defines an Analyzer that constructs the IR
// of an error-free package and returns the set of all
// functions within it. It does not report any diagnostics itself but
// may be used as an input to other analyzers.
//
// THIS INTERFACE IS EXPERIMENTAL AND MAY BE SUBJECT TO INCOMPATIBLE CHANGE.
package buildir

import (
	"go/ast"
	"go/types"
	"reflect"

	"honnef.co/go/tools/go/ir"

	"golang.org/x/tools/go/analysis"
)

type noReturn struct {
	Kind ir.NoReturn
}

func (*noReturn) AFact() {}

var Analyzer = &analysis.Analyzer{
	Name:       "buildir",
	Doc:        "build IR for later passes",
	Run:        run,
	ResultType: reflect.TypeOf(new(IR)),
	FactTypes:  []analysis.Fact{new(noReturn)},
}

// IR provides intermediate representation for all the
// non-blank source functions in the current package.
type IR struct {
	Pkg      *ir.Package
	SrcFuncs []*ir.Function
}

func run(pass *analysis.Pass) (interface{}, error) {
	// Plundered from ssautil.BuildPackage.

	// We must create a new Program for each Package because the
	// analysis API provides no place to hang a Program shared by
	// all Packages. Consequently, IR Packages and Functions do not
	// have a canonical representation across an analysis session of
	// multiple packages. This is unlikely to be a problem in
	// practice because the analysis API essentially forces all
	// packages to be analysed independently, so any given call to
	// Analysis.Run on a package will see only IR objects belonging
	// to a single Program.

	mode := ir.GlobalDebug

	prog := ir.NewProgram(pass.Fset, mode)

	// Create IR packages for all imports.
	// Order is not significant.
	created := make(map[*types.Package]bool)
	var createAll func(pkgs []*types.Package)
	createAll = func(pkgs []*types.Package) {
		for _, p := range pkgs {
			if !created[p] {
				created[p] = true
				irpkg := prog.CreatePackage(p, nil, nil, true)
				for _, fn := range irpkg.Functions {
					if ast.IsExported(fn.Name()) {
						var noRet noReturn
						if pass.ImportObjectFact(fn.Object(), &noRet) {
							fn.NoReturn = noRet.Kind
						}
					}
				}
				createAll(p.Imports())
			}
		}
	}
	createAll(pass.Pkg.Imports())

	// Create and build the primary package.
	irpkg := prog.CreatePackage(pass.Pkg, pass.Files, pass.TypesInfo, false)
	irpkg.Build()

	// Compute list of source functions, including literals,
	// in source order.
	var addAnons func(f *ir.Function)
	funcs := make([]*ir.Function, len(irpkg.Functions))
	copy(funcs, irpkg.Functions)
	addAnons = func(f *ir.Function) {
		for _, anon := range f.AnonFuncs {
			funcs = append(funcs, anon)
			addAnons(anon)
		}
	}
	for _, fn := range irpkg.Functions {
		addAnons(fn)
		if fn.NoReturn > 0 {
			pass.ExportObjectFact(fn.Object(), &noReturn{fn.NoReturn})
		}
	}

	return &IR{Pkg: irpkg, SrcFuncs: funcs}, nil
}
