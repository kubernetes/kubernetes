package checkers

import (
	"fmt"
	"go/ast"

	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "dupImport"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects multiple imports of the same package under different aliases"
	info.Before = `
import (
	"fmt"
	priting "fmt" // Imported the second time
)`
	info.After = `
import(
	"fmt"
)`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return &dupImportChecker{ctx: ctx}, nil
	})
}

type dupImportChecker struct {
	ctx *linter.CheckerContext
}

func (c *dupImportChecker) WalkFile(f *ast.File) {
	imports := make(map[string][]*ast.ImportSpec)
	for _, importDcl := range f.Imports {
		pkg := importDcl.Path.Value
		imports[pkg] = append(imports[pkg], importDcl)
	}

	for _, importList := range imports {
		if len(importList) == 1 {
			continue
		}
		c.warn(importList)
	}
}

func (c *dupImportChecker) warn(importList []*ast.ImportSpec) {
	msg := fmt.Sprintf("package is imported %d times under different aliases on lines", len(importList))
	for idx, importDcl := range importList {
		switch {
		case idx == len(importList)-1:
			msg += " and"
		case idx > 0:
			msg += ","
		}
		msg += fmt.Sprintf(" %d", c.ctx.FileSet.Position(importDcl.Pos()).Line)
	}
	for _, importDcl := range importList {
		c.ctx.Warn(importDcl, msg)
	}
}
