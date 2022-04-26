// Copyright (c) 2015, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

package check

import (
	"go/ast"
	"go/types"
)

type pkgTypes struct {
	ifaces    map[string]string
	funcSigns map[string]bool
}

func (p *pkgTypes) getTypes(pkg *types.Package) {
	p.ifaces = make(map[string]string)
	p.funcSigns = make(map[string]bool)
	done := make(map[*types.Package]bool)
	addTypes := func(pkg *types.Package, top bool) {
		if done[pkg] {
			return
		}
		done[pkg] = true
		ifs, funs := fromScope(pkg.Scope())
		fullName := func(name string) string {
			if !top {
				return pkg.Path() + "." + name
			}
			return name
		}
		for iftype, name := range ifs {
			// only suggest exported interfaces
			if ast.IsExported(name) {
				p.ifaces[iftype] = fullName(name)
			}
		}
		for ftype := range funs {
			// ignore non-exported func signatures too
			p.funcSigns[ftype] = true
		}
	}
	for _, imp := range pkg.Imports() {
		addTypes(imp, false)
		for _, imp2 := range imp.Imports() {
			addTypes(imp2, false)
		}
	}
	addTypes(pkg, true)
}
