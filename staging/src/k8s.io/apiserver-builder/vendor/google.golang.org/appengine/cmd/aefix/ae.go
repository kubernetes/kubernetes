// Copyright 2016 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"path"
	"strconv"
	"strings"
)

const (
	ctxPackage = "golang.org/x/net/context"

	newPackageBase = "google.golang.org/"
	stutterPackage = false
)

func init() {
	register(fix{
		"ae",
		"2016-04-15",
		aeFn,
		`Update old App Engine APIs to new App Engine APIs`,
	})
}

// logMethod is the set of methods on appengine.Context used for logging.
var logMethod = map[string]bool{
	"Debugf":    true,
	"Infof":     true,
	"Warningf":  true,
	"Errorf":    true,
	"Criticalf": true,
}

// mapPackage turns "appengine" into "google.golang.org/appengine", etc.
func mapPackage(s string) string {
	if stutterPackage {
		s += "/" + path.Base(s)
	}
	return newPackageBase + s
}

func aeFn(f *ast.File) bool {
	// During the walk, we track the last thing seen that looks like
	// an appengine.Context, and reset it once the walk leaves a func.
	var lastContext *ast.Ident

	fixed := false

	// Update imports.
	mainImp := "appengine"
	for _, imp := range f.Imports {
		pth, _ := strconv.Unquote(imp.Path.Value)
		if pth == "appengine" || strings.HasPrefix(pth, "appengine/") {
			newPth := mapPackage(pth)
			imp.Path.Value = strconv.Quote(newPth)
			fixed = true

			if pth == "appengine" {
				mainImp = newPth
			}
		}
	}

	// Update any API changes.
	walk(f, func(n interface{}) {
		if ft, ok := n.(*ast.FuncType); ok && ft.Params != nil {
			// See if this func has an `appengine.Context arg`.
			// If so, remember its identifier.
			for _, param := range ft.Params.List {
				if !isPkgDot(param.Type, "appengine", "Context") {
					continue
				}
				if len(param.Names) == 1 {
					lastContext = param.Names[0]
					break
				}
			}
			return
		}

		if as, ok := n.(*ast.AssignStmt); ok {
			if len(as.Lhs) == 1 && len(as.Rhs) == 1 {
				// If this node is an assignment from an appengine.NewContext invocation,
				// remember the identifier on the LHS.
				if isCall(as.Rhs[0], "appengine", "NewContext") {
					if ident, ok := as.Lhs[0].(*ast.Ident); ok {
						lastContext = ident
						return
					}
				}
				// x (=|:=) appengine.Timeout(y, z)
				//   should become
				// x, _ (=|:=) context.WithTimeout(y, z)
				if isCall(as.Rhs[0], "appengine", "Timeout") {
					addImport(f, ctxPackage)
					as.Lhs = append(as.Lhs, ast.NewIdent("_"))
					// isCall already did the type checking.
					sel := as.Rhs[0].(*ast.CallExpr).Fun.(*ast.SelectorExpr)
					sel.X = ast.NewIdent("context")
					sel.Sel = ast.NewIdent("WithTimeout")
					fixed = true
					return
				}
			}
			return
		}

		// If this node is a FuncDecl, we've finished the function, so reset lastContext.
		if _, ok := n.(*ast.FuncDecl); ok {
			lastContext = nil
			return
		}

		if call, ok := n.(*ast.CallExpr); ok {
			if isPkgDot(call.Fun, "appengine", "Datacenter") && len(call.Args) == 0 {
				insertContext(f, call, lastContext)
				fixed = true
				return
			}
			if isPkgDot(call.Fun, "taskqueue", "QueueStats") && len(call.Args) == 3 {
				call.Args = call.Args[:2] // drop last arg
				fixed = true
				return
			}

			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return
			}
			if lastContext != nil && refersTo(sel.X, lastContext) && logMethod[sel.Sel.Name] {
				// c.Errorf(...)
				//   should become
				// log.Errorf(c, ...)
				addImport(f, mapPackage("appengine/log"))
				sel.X = &ast.Ident{ // ast.NewIdent doesn't preserve the position.
					NamePos: sel.X.Pos(),
					Name:    "log",
				}
				insertContext(f, call, lastContext)
				fixed = true
				return
			}
		}
	})

	// Change any `appengine.Context` to `context.Context`.
	// Do this in a separate walk because the previous walk
	// wants to identify "appengine.Context".
	walk(f, func(n interface{}) {
		expr, ok := n.(ast.Expr)
		if ok && isPkgDot(expr, "appengine", "Context") {
			addImport(f, ctxPackage)
			// isPkgDot did the type checking.
			n.(*ast.SelectorExpr).X.(*ast.Ident).Name = "context"
			fixed = true
			return
		}
	})

	// The changes above might remove the need to import "appengine".
	// Check if it's used, and drop it if it isn't.
	if fixed && !usesImport(f, mainImp) {
		deleteImport(f, mainImp)
	}

	return fixed
}

// ctx may be nil.
func insertContext(f *ast.File, call *ast.CallExpr, ctx *ast.Ident) {
	if ctx == nil {
		// context is unknown, so use a plain "ctx".
		ctx = ast.NewIdent("ctx")
	} else {
		// Create a fresh *ast.Ident so we drop the position information.
		ctx = ast.NewIdent(ctx.Name)
	}

	call.Args = append([]ast.Expr{ctx}, call.Args...)
}
