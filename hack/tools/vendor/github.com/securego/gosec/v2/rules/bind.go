// (c) Copyright 2016 Hewlett Packard Enterprise Development LP
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rules

import (
	"go/ast"
	"regexp"

	"github.com/securego/gosec/v2"
)

// Looks for net.Listen("0.0.0.0") or net.Listen(":8080")
type bindsToAllNetworkInterfaces struct {
	gosec.MetaData
	calls   gosec.CallList
	pattern *regexp.Regexp
}

func (r *bindsToAllNetworkInterfaces) ID() string {
	return r.MetaData.ID
}

func (r *bindsToAllNetworkInterfaces) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	callExpr := r.calls.ContainsPkgCallExpr(n, c, false)
	if callExpr == nil {
		return nil, nil
	}
	if len(callExpr.Args) > 1 {
		arg := callExpr.Args[1]
		if bl, ok := arg.(*ast.BasicLit); ok {
			if arg, err := gosec.GetString(bl); err == nil {
				if r.pattern.MatchString(arg) {
					return gosec.NewIssue(c, n, r.ID(), r.What, r.Severity, r.Confidence), nil
				}
			}
		} else if ident, ok := arg.(*ast.Ident); ok {
			values := gosec.GetIdentStringValues(ident)
			for _, value := range values {
				if r.pattern.MatchString(value) {
					return gosec.NewIssue(c, n, r.ID(), r.What, r.Severity, r.Confidence), nil
				}
			}
		}
	} else if len(callExpr.Args) > 0 {
		values := gosec.GetCallStringArgsValues(callExpr.Args[0], c)
		for _, value := range values {
			if r.pattern.MatchString(value) {
				return gosec.NewIssue(c, n, r.ID(), r.What, r.Severity, r.Confidence), nil
			}
		}
	}
	return nil, nil
}

// NewBindsToAllNetworkInterfaces detects socket connections that are setup to
// listen on all network interfaces.
func NewBindsToAllNetworkInterfaces(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	calls := gosec.NewCallList()
	calls.Add("net", "Listen")
	calls.Add("crypto/tls", "Listen")
	return &bindsToAllNetworkInterfaces{
		calls:   calls,
		pattern: regexp.MustCompile(`^(0.0.0.0|:).*$`),
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.Medium,
			Confidence: gosec.High,
			What:       "Binds to all network interfaces",
		},
	}, []ast.Node{(*ast.CallExpr)(nil)}
}
