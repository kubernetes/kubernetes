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
	"fmt"
	"go/ast"

	"github.com/securego/gosec/v2"
)

type weakKeyStrength struct {
	gosec.MetaData
	calls gosec.CallList
	bits  int
}

func (w *weakKeyStrength) ID() string {
	return w.MetaData.ID
}

func (w *weakKeyStrength) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	if callExpr := w.calls.ContainsPkgCallExpr(n, c, false); callExpr != nil {
		if bits, err := gosec.GetInt(callExpr.Args[1]); err == nil && bits < (int64)(w.bits) {
			return gosec.NewIssue(c, n, w.ID(), w.What, w.Severity, w.Confidence), nil
		}
	}
	return nil, nil
}

// NewWeakKeyStrength builds a rule that detects RSA keys < 2048 bits
func NewWeakKeyStrength(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	calls := gosec.NewCallList()
	calls.Add("crypto/rsa", "GenerateKey")
	bits := 2048
	return &weakKeyStrength{
		calls: calls,
		bits:  bits,
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.Medium,
			Confidence: gosec.High,
			What:       fmt.Sprintf("RSA keys should be at least %d bits", bits),
		},
	}, []ast.Node{(*ast.CallExpr)(nil)}
}
