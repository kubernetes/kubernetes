package vars

import (
	"go/ast"
	"go/token"

	"github.com/leonklingele/grouper/pkg/analyzer/globals"

	"golang.org/x/tools/go/analysis"
)

// https://go.dev/ref/spec#Variable_declarations

func Filepass(c *Config, p *analysis.Pass, f *ast.File) error {
	return globals.Filepass(
		p, f,
		token.VAR, c.RequireSingleVar, c.RequireGrouping,
	)
}
