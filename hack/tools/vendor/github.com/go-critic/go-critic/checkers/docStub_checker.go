package checkers

import (
	"go/ast"
	"go/token"
	"regexp"
	"strings"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "docStub"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects comments that silence go lint complaints about doc-comment"
	info.Before = `
// Foo ...
func Foo() {
}`
	info.After = `
// (A) - remove the doc-comment stub
func Foo() {}
// (B) - replace it with meaningful comment
// Foo is a demonstration-only function.
func Foo() {}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		re := `(?i)^\.\.\.$|^\.$|^xxx\.?$|^whatever\.?$`
		c := &docStubChecker{
			ctx:           ctx,
			stubCommentRE: regexp.MustCompile(re),
		}
		return c, nil
	})
}

type docStubChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	stubCommentRE *regexp.Regexp
}

func (c *docStubChecker) WalkFile(f *ast.File) {
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			c.visitDoc(decl, decl.Name, decl.Doc, false)
		case *ast.GenDecl:
			if decl.Tok != token.TYPE {
				continue
			}
			if len(decl.Specs) == 1 {
				spec := decl.Specs[0].(*ast.TypeSpec)
				// Only 1 spec, use doc from the decl itself.
				c.visitDoc(spec, spec.Name, decl.Doc, true)
			}
			// N specs, use per-spec doc.
			for _, spec := range decl.Specs {
				spec := spec.(*ast.TypeSpec)
				c.visitDoc(spec, spec.Name, spec.Doc, true)
			}
		}
	}
}

func (c *docStubChecker) visitDoc(decl ast.Node, sym *ast.Ident, doc *ast.CommentGroup, article bool) {
	if !sym.IsExported() || doc == nil {
		return
	}
	line := strings.TrimSpace(doc.List[0].Text[len("//"):])
	if article {
		// Skip optional article.
		for _, a := range []string{"The ", "An ", "A "} {
			if strings.HasPrefix(line, a) {
				line = line[len(a):]
				break
			}
		}
	}
	if !strings.HasPrefix(line, sym.Name) {
		return
	}
	line = strings.TrimSpace(line[len(sym.Name):])
	// Now try to detect the "stub" part.
	if c.stubCommentRE.MatchString(line) {
		c.warn(decl)
	}
}

func (c *docStubChecker) warn(cause ast.Node) {
	c.ctx.Warn(cause, "silencing go lint doc-comment warnings is unadvised")
}
