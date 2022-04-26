package rules

import (
	"go/ast"

	"github.com/securego/gosec/v2"
)

type sshHostKey struct {
	gosec.MetaData
	pkg   string
	calls []string
}

func (r *sshHostKey) ID() string {
	return r.MetaData.ID
}

func (r *sshHostKey) Match(n ast.Node, c *gosec.Context) (gi *gosec.Issue, err error) {
	if _, matches := gosec.MatchCallByPackage(n, c, r.pkg, r.calls...); matches {
		return gosec.NewIssue(c, n, r.ID(), r.What, r.Severity, r.Confidence), nil
	}
	return nil, nil
}

// NewSSHHostKey rule detects the use of insecure ssh HostKeyCallback.
func NewSSHHostKey(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return &sshHostKey{
		pkg:   "golang.org/x/crypto/ssh",
		calls: []string{"InsecureIgnoreHostKey"},
		MetaData: gosec.MetaData{
			ID:         id,
			What:       "Use of ssh InsecureIgnoreHostKey should be audited",
			Severity:   gosec.Medium,
			Confidence: gosec.High,
		},
	}, []ast.Node{(*ast.CallExpr)(nil)}
}
