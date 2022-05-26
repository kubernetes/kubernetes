package checkers

import (
	"go/ast"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "importShadow"
	info.Tags = []string{"style", "opinionated"}
	info.Summary = "Detects when imported package names shadowed in the assignments"
	info.Before = `
// "path/filepath" is imported.
filepath := "foo.txt"`
	info.After = `
filename := "foo.txt"`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		ctx.Require.PkgObjects = true
		return astwalk.WalkerForLocalDef(&importShadowChecker{ctx: ctx}, ctx.TypesInfo), nil
	})
}

type importShadowChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *importShadowChecker) VisitLocalDef(def astwalk.Name, _ ast.Expr) {
	for pkgObj, name := range c.ctx.PkgObjects {
		if name == def.ID.Name && name != "_" {
			c.warn(def.ID, name, pkgObj.Imported())
		}
	}
}

func (c *importShadowChecker) warn(id ast.Node, importedName string, pkg *types.Package) {
	if isStdlibPkg(pkg) {
		c.ctx.Warn(id, "shadow of imported package '%s'", importedName)
	} else {
		c.ctx.Warn(id, "shadow of imported from '%s' package '%s'", pkg.Path(), importedName)
	}
}
