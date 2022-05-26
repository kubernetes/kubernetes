// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package irutil

import "honnef.co/go/tools/go/ir"

// This file defines utilities for visiting the IR of
// a Program.
//
// TODO(adonovan): test coverage.

// AllFunctions finds and returns the set of functions potentially
// needed by program prog, as determined by a simple linker-style
// reachability algorithm starting from the members and method-sets of
// each package.  The result may include anonymous functions and
// synthetic wrappers.
//
// Precondition: all packages are built.
//
func AllFunctions(prog *ir.Program) map[*ir.Function]bool {
	visit := visitor{
		prog: prog,
		seen: make(map[*ir.Function]bool),
	}
	visit.program()
	return visit.seen
}

type visitor struct {
	prog *ir.Program
	seen map[*ir.Function]bool
}

func (visit *visitor) program() {
	for _, pkg := range visit.prog.AllPackages() {
		for _, mem := range pkg.Members {
			if fn, ok := mem.(*ir.Function); ok {
				visit.function(fn)
			}
		}
	}
	for _, T := range visit.prog.RuntimeTypes() {
		mset := visit.prog.MethodSets.MethodSet(T)
		for i, n := 0, mset.Len(); i < n; i++ {
			visit.function(visit.prog.MethodValue(mset.At(i)))
		}
	}
}

func (visit *visitor) function(fn *ir.Function) {
	if !visit.seen[fn] {
		visit.seen[fn] = true
		var buf [10]*ir.Value // avoid alloc in common case
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				for _, op := range instr.Operands(buf[:0]) {
					if fn, ok := (*op).(*ir.Function); ok {
						visit.function(fn)
					}
				}
			}
		}
	}
}

// MainPackages returns the subset of the specified packages
// named "main" that define a main function.
// The result may include synthetic "testmain" packages.
func MainPackages(pkgs []*ir.Package) []*ir.Package {
	var mains []*ir.Package
	for _, pkg := range pkgs {
		if pkg.Pkg.Name() == "main" && pkg.Func("main") != nil {
			mains = append(mains, pkg)
		}
	}
	return mains
}
