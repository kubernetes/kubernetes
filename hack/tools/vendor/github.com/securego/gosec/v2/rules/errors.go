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
	"go/types"

	"github.com/securego/gosec/v2"
)

type noErrorCheck struct {
	gosec.MetaData
	whitelist gosec.CallList
}

func (r *noErrorCheck) ID() string {
	return r.MetaData.ID
}

func returnsError(callExpr *ast.CallExpr, ctx *gosec.Context) int {
	if tv := ctx.Info.TypeOf(callExpr); tv != nil {
		switch t := tv.(type) {
		case *types.Tuple:
			for pos := 0; pos < t.Len(); pos++ {
				variable := t.At(pos)
				if variable != nil && variable.Type().String() == "error" {
					return pos
				}
			}
		case *types.Named:
			if t.String() == "error" {
				return 0
			}
		}
	}
	return -1
}

func (r *noErrorCheck) Match(n ast.Node, ctx *gosec.Context) (*gosec.Issue, error) {
	switch stmt := n.(type) {
	case *ast.AssignStmt:
		cfg := ctx.Config
		if enabled, err := cfg.IsGlobalEnabled(gosec.Audit); err == nil && enabled {
			for _, expr := range stmt.Rhs {
				if callExpr, ok := expr.(*ast.CallExpr); ok && r.whitelist.ContainsCallExpr(expr, ctx) == nil {
					pos := returnsError(callExpr, ctx)
					if pos < 0 || pos >= len(stmt.Lhs) {
						return nil, nil
					}
					if id, ok := stmt.Lhs[pos].(*ast.Ident); ok && id.Name == "_" {
						return gosec.NewIssue(ctx, n, r.ID(), r.What, r.Severity, r.Confidence), nil
					}
				}
			}
		}
	case *ast.ExprStmt:
		if callExpr, ok := stmt.X.(*ast.CallExpr); ok && r.whitelist.ContainsCallExpr(stmt.X, ctx) == nil {
			pos := returnsError(callExpr, ctx)
			if pos >= 0 {
				return gosec.NewIssue(ctx, n, r.ID(), r.What, r.Severity, r.Confidence), nil
			}
		}
	}
	return nil, nil
}

// NewNoErrorCheck detects if the returned error is unchecked
func NewNoErrorCheck(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	// TODO(gm) Come up with sensible defaults here. Or flip it to use a
	// black list instead.
	whitelist := gosec.NewCallList()
	whitelist.AddAll("bytes.Buffer", "Write", "WriteByte", "WriteRune", "WriteString")
	whitelist.AddAll("fmt", "Print", "Printf", "Println", "Fprint", "Fprintf", "Fprintln")
	whitelist.AddAll("strings.Builder", "Write", "WriteByte", "WriteRune", "WriteString")
	whitelist.Add("io.PipeWriter", "CloseWithError")
	whitelist.Add("hash.Hash", "Write")
	whitelist.Add("os", "Unsetenv")

	if configured, ok := conf["G104"]; ok {
		if whitelisted, ok := configured.(map[string]interface{}); ok {
			for pkg, funcs := range whitelisted {
				if funcs, ok := funcs.([]interface{}); ok {
					whitelist.AddAll(pkg, toStringSlice(funcs)...)
				}
			}
		}
	}

	return &noErrorCheck{
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.Low,
			Confidence: gosec.High,
			What:       "Errors unhandled.",
		},
		whitelist: whitelist,
	}, []ast.Node{(*ast.AssignStmt)(nil), (*ast.ExprStmt)(nil)}
}

func toStringSlice(values []interface{}) []string {
	result := []string{}
	for _, value := range values {
		if value, ok := value.(string); ok {
			result = append(result, value)
		}
	}
	return result
}
