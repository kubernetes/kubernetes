package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "builtinShadowDecl"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects top-level declarations that shadow the predeclared identifiers"
	info.Before = `type int struct {}`
	info.After = `type myInt struct {}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return &builtinShadowDeclChecker{ctx: ctx}, nil
	})
}

type builtinShadowDeclChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *builtinShadowDeclChecker) WalkFile(f *ast.File) {
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			// Don't check methods. They can shadow anything safely.
			if decl.Recv == nil {
				c.checkName(decl.Name)
			}
		case *ast.GenDecl:
			c.visitGenDecl(decl)
		}
	}
}

func (c *builtinShadowDeclChecker) visitGenDecl(decl *ast.GenDecl) {
	for _, spec := range decl.Specs {
		switch spec := spec.(type) {
		case *ast.ValueSpec:
			for _, name := range spec.Names {
				c.checkName(name)
			}
		case *ast.TypeSpec:
			c.checkName(spec.Name)
		}
	}
}

func (c *builtinShadowDeclChecker) checkName(name *ast.Ident) {
	if isBuiltin(name.Name) {
		c.warn(name)
	}
}

func (c *builtinShadowDeclChecker) warn(ident *ast.Ident) {
	c.ctx.Warn(ident, "shadowing of predeclared identifier: %s", ident)
}
