package checkers

import (
	"go/ast"
	"go/token"
	"regexp"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "commentedOutImport"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects commented-out imports"
	info.Before = `
import (
	"fmt"
	//"os"
)`
	info.After = `
import (
	"fmt"
)`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		const pattern = `(?m)^(?://|/\*)?\s*"([a-zA-Z0-9_/]+)"\s*(?:\*/)?$`
		return &commentedOutImportChecker{
			ctx:            ctx,
			importStringRE: regexp.MustCompile(pattern),
		}, nil
	})
}

type commentedOutImportChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	importStringRE *regexp.Regexp
}

func (c *commentedOutImportChecker) WalkFile(f *ast.File) {
	// TODO(quasilyte): handle commented-out import spec,
	// for example: // import "errors".

	for _, decl := range f.Decls {
		decl, ok := decl.(*ast.GenDecl)
		if !ok || decl.Tok != token.IMPORT {
			// Import decls can only be in the beginning of the file.
			// If we've met some other decl, there will be no more
			// import decls.
			break
		}

		// Find comments inside this import decl span.
		for _, cg := range f.Comments {
			if cg.Pos() > decl.Rparen {
				break // Below the decl, stop.
			}
			if cg.Pos() < decl.Lparen {
				continue // Before the decl, skip.
			}

			for _, comment := range cg.List {
				for _, m := range c.importStringRE.FindAllStringSubmatch(comment.Text, -1) {
					c.warn(comment, m[1])
				}
			}
		}
	}
}

func (c *commentedOutImportChecker) warn(cause ast.Node, path string) {
	c.ctx.Warn(cause, "remove commented-out %q import", path)
}
