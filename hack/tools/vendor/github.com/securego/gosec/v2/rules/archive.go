package rules

import (
	"go/ast"
	"go/types"

	"github.com/securego/gosec/v2"
)

type archive struct {
	gosec.MetaData
	calls    gosec.CallList
	argTypes []string
}

func (a *archive) ID() string {
	return a.MetaData.ID
}

// Match inspects AST nodes to determine if the filepath.Joins uses any argument derived from type zip.File or tar.Header
func (a *archive) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	if node := a.calls.ContainsPkgCallExpr(n, c, false); node != nil {
		for _, arg := range node.Args {
			var argType types.Type
			if selector, ok := arg.(*ast.SelectorExpr); ok {
				argType = c.Info.TypeOf(selector.X)
			} else if ident, ok := arg.(*ast.Ident); ok {
				if ident.Obj != nil && ident.Obj.Kind == ast.Var {
					decl := ident.Obj.Decl
					if assign, ok := decl.(*ast.AssignStmt); ok {
						if selector, ok := assign.Rhs[0].(*ast.SelectorExpr); ok {
							argType = c.Info.TypeOf(selector.X)
						}
					}
				}
			}

			if argType != nil {
				for _, t := range a.argTypes {
					if argType.String() == t {
						return gosec.NewIssue(c, n, a.ID(), a.What, a.Severity, a.Confidence), nil
					}
				}
			}
		}
	}
	return nil, nil
}

// NewArchive creates a new rule which detects the file traversal when extracting zip/tar archives
func NewArchive(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	calls := gosec.NewCallList()
	calls.Add("path/filepath", "Join")
	calls.Add("path", "Join")
	return &archive{
		calls:    calls,
		argTypes: []string{"*archive/zip.File", "*archive/tar.Header"},
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.Medium,
			Confidence: gosec.High,
			What:       "File traversal when extracting zip/tar archive",
		},
	}, []ast.Node{(*ast.CallExpr)(nil)}
}
