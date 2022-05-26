package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "typeDefFirst"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects method declarations preceding the type definition itself"
	info.Before = `
func (r rec) Method() {}
type rec struct{}
`
	info.After = `
type rec struct{}
func (r rec) Method() {}
`
	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return &typeDefFirstChecker{
			ctx: ctx,
		}, nil
	})
}

type typeDefFirstChecker struct {
	astwalk.WalkHandler
	ctx          *linter.CheckerContext
	trackedTypes map[string]bool
}

func (c *typeDefFirstChecker) WalkFile(f *ast.File) {
	if len(f.Decls) == 0 {
		return
	}

	c.trackedTypes = make(map[string]bool)
	for _, decl := range f.Decls {
		c.walkDecl(decl)
	}
}

func (c *typeDefFirstChecker) walkDecl(decl ast.Decl) {
	switch decl := decl.(type) {
	case *ast.FuncDecl:
		if decl.Recv == nil {
			return
		}
		receiver := decl.Recv.List[0]
		typeName := c.receiverType(receiver.Type)
		c.trackedTypes[typeName] = true

	case *ast.GenDecl:
		if decl.Tok != token.TYPE {
			return
		}
		for _, spec := range decl.Specs {
			spec, ok := spec.(*ast.TypeSpec)
			if !ok {
				return
			}
			typeName := spec.Name.Name
			if val, ok := c.trackedTypes[typeName]; ok && val {
				c.warn(decl, typeName)
			}
		}
	}
}

func (c *typeDefFirstChecker) receiverType(e ast.Expr) string {
	switch e := e.(type) {
	case *ast.StarExpr:
		return c.receiverType(e.X)
	case *ast.Ident:
		return e.Name
	default:
		panic("unreachable")
	}
}

func (c *typeDefFirstChecker) warn(cause ast.Node, typeName string) {
	c.ctx.Warn(cause, "definition of type '%s' should appear before its methods", typeName)
}
