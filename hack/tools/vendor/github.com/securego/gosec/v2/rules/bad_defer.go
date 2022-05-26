package rules

import (
	"fmt"
	"go/ast"
	"strings"

	"github.com/securego/gosec/v2"
)

type deferType struct {
	typ     string
	methods []string
}

type badDefer struct {
	gosec.MetaData
	types []deferType
}

func (r *badDefer) ID() string {
	return r.MetaData.ID
}

func normalize(typ string) string {
	return strings.TrimPrefix(typ, "*")
}

func contains(methods []string, method string) bool {
	for _, m := range methods {
		if m == method {
			return true
		}
	}
	return false
}

func (r *badDefer) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	if deferStmt, ok := n.(*ast.DeferStmt); ok {
		for _, deferTyp := range r.types {
			if typ, method, err := gosec.GetCallInfo(deferStmt.Call, c); err == nil {
				if normalize(typ) == deferTyp.typ && contains(deferTyp.methods, method) {
					return gosec.NewIssue(c, n, r.ID(), fmt.Sprintf(r.What, method, typ), r.Severity, r.Confidence), nil
				}
			}
		}
	}

	return nil, nil
}

// NewDeferredClosing detects unsafe defer of error returning methods
func NewDeferredClosing(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return &badDefer{
		types: []deferType{
			{
				typ:     "os.File",
				methods: []string{"Close"},
			},
		},
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.Medium,
			Confidence: gosec.High,
			What:       "Deferring unsafe method %q on type %q",
		},
	}, []ast.Node{(*ast.DeferStmt)(nil)}
}
