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

	"github.com/securego/gosec/v2"
)

type weakRand struct {
	gosec.MetaData
	funcNames   []string
	packagePath string
}

func (w *weakRand) ID() string {
	return w.MetaData.ID
}

func (w *weakRand) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	for _, funcName := range w.funcNames {
		if _, matched := gosec.MatchCallByPackage(n, c, w.packagePath, funcName); matched {
			return gosec.NewIssue(c, n, w.ID(), w.What, w.Severity, w.Confidence), nil
		}
	}

	return nil, nil
}

// NewWeakRandCheck detects the use of random number generator that isn't cryptographically secure
func NewWeakRandCheck(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return &weakRand{
		funcNames: []string{
			"New", "Read", "Float32", "Float64", "Int", "Int31",
			"Int31n", "Int63", "Int63n", "Intn", "NormalFloat64", "Uint32", "Uint64",
		},
		packagePath: "math/rand",
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.High,
			Confidence: gosec.Medium,
			What:       "Use of weak random number generator (math/rand instead of crypto/rand)",
		},
	}, []ast.Node{(*ast.CallExpr)(nil)}
}
