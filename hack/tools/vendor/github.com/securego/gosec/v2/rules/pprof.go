package rules

import (
	"go/ast"

	"github.com/securego/gosec/v2"
)

type pprofCheck struct {
	gosec.MetaData
	importPath string
	importName string
}

// ID returns the ID of the check
func (p *pprofCheck) ID() string {
	return p.MetaData.ID
}

// Match checks for pprof imports
func (p *pprofCheck) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	if node, ok := n.(*ast.ImportSpec); ok {
		if p.importPath == unquote(node.Path.Value) && node.Name != nil && p.importName == node.Name.Name {
			return gosec.NewIssue(c, node, p.ID(), p.What, p.Severity, p.Confidence), nil
		}
	}
	return nil, nil
}

// NewPprofCheck detects when the profiling endpoint is automatically exposed
func NewPprofCheck(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return &pprofCheck{
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.High,
			Confidence: gosec.High,
			What:       "Profiling endpoint is automatically exposed on /debug/pprof",
		},
		importPath: "net/http/pprof",
		importName: "_",
	}, []ast.Node{(*ast.ImportSpec)(nil)}
}
