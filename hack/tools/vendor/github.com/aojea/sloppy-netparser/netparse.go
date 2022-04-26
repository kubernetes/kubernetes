package main

import (
	"go/ast"
)

func sloppyParsers(f *ast.File) bool {
	if ok, _ := getImport(f, "net"); !ok {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		ce, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}
		se, ok := ce.Fun.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if !isTopName(se.X, "net") || se.Sel == nil {
			return
		}

		switch ss := se.Sel.String(); ss {
		case "ParseIP":
			id, _ := se.X.(*ast.Ident)
			id.Name = "netutils"
			se.Sel.Name = "ParseIPSloppy"
			fixed = true
		case "ParseCIDR":
			se.X.(*ast.Ident).Name = "netutils"
			se.Sel.Name = "ParseCIDRSloppy"
			fixed = true
		}
	})
	if fixed {
		addImport(f, "netutils", "k8s.io/utils/net")
		rewriteImportName(f, "k8s.io/utils/net", "netutils", "k8s.io/utils/net")
	}
	return fixed
}
