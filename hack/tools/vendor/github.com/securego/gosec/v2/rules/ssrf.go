package rules

import (
	"go/ast"
	"go/types"

	"github.com/securego/gosec/v2"
)

type ssrf struct {
	gosec.MetaData
	gosec.CallList
}

// ID returns the identifier for this rule
func (r *ssrf) ID() string {
	return r.MetaData.ID
}

// ResolveVar tries to resolve the first argument of a call expression
// The first argument is the url
func (r *ssrf) ResolveVar(n *ast.CallExpr, c *gosec.Context) bool {
	if len(n.Args) > 0 {
		arg := n.Args[0]
		if ident, ok := arg.(*ast.Ident); ok {
			obj := c.Info.ObjectOf(ident)
			if _, ok := obj.(*types.Var); ok {
				scope := c.Pkg.Scope()
				if scope != nil && scope.Lookup(ident.Name) != nil {
					// a URL defined in a variable at package scope can be changed at any time
					return true
				}
				if !gosec.TryResolve(ident, c) {
					return true
				}
			}
		}
	}
	return false
}

// Match inspects AST nodes to determine if certain net/http methods are called with variable input
func (r *ssrf) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	// Call expression is using http package directly
	if node := r.ContainsPkgCallExpr(n, c, false); node != nil {
		if r.ResolveVar(node, c) {
			return gosec.NewIssue(c, n, r.ID(), r.What, r.Severity, r.Confidence), nil
		}
	}
	return nil, nil
}

// NewSSRFCheck detects cases where HTTP requests are sent
func NewSSRFCheck(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	rule := &ssrf{
		CallList: gosec.NewCallList(),
		MetaData: gosec.MetaData{
			ID:         id,
			What:       "Potential HTTP request made with variable url",
			Severity:   gosec.Medium,
			Confidence: gosec.Medium,
		},
	}
	rule.AddAll("net/http", "Do", "Get", "Head", "Post", "PostForm", "RoundTrip")
	return rule, []ast.Node{(*ast.CallExpr)(nil)}
}
