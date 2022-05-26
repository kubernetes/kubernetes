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
	"go/token"
	"regexp"
	"strconv"

	zxcvbn "github.com/nbutton23/zxcvbn-go"
	"github.com/securego/gosec/v2"
)

type credentials struct {
	gosec.MetaData
	pattern          *regexp.Regexp
	entropyThreshold float64
	perCharThreshold float64
	truncate         int
	ignoreEntropy    bool
}

func (r *credentials) ID() string {
	return r.MetaData.ID
}

func truncate(s string, n int) string {
	if n > len(s) {
		return s
	}
	return s[:n]
}

func (r *credentials) isHighEntropyString(str string) bool {
	s := truncate(str, r.truncate)
	info := zxcvbn.PasswordStrength(s, []string{})
	entropyPerChar := info.Entropy / float64(len(s))
	return (info.Entropy >= r.entropyThreshold ||
		(info.Entropy >= (r.entropyThreshold/2) &&
			entropyPerChar >= r.perCharThreshold))
}

func (r *credentials) Match(n ast.Node, ctx *gosec.Context) (*gosec.Issue, error) {
	switch node := n.(type) {
	case *ast.AssignStmt:
		return r.matchAssign(node, ctx)
	case *ast.ValueSpec:
		return r.matchValueSpec(node, ctx)
	case *ast.BinaryExpr:
		return r.matchEqualityCheck(node, ctx)
	}
	return nil, nil
}

func (r *credentials) matchAssign(assign *ast.AssignStmt, ctx *gosec.Context) (*gosec.Issue, error) {
	for _, i := range assign.Lhs {
		if ident, ok := i.(*ast.Ident); ok {
			if r.pattern.MatchString(ident.Name) {
				for _, e := range assign.Rhs {
					if val, err := gosec.GetString(e); err == nil {
						if r.ignoreEntropy || (!r.ignoreEntropy && r.isHighEntropyString(val)) {
							return gosec.NewIssue(ctx, assign, r.ID(), r.What, r.Severity, r.Confidence), nil
						}
					}
				}
			}
		}
	}
	return nil, nil
}

func (r *credentials) matchValueSpec(valueSpec *ast.ValueSpec, ctx *gosec.Context) (*gosec.Issue, error) {
	for index, ident := range valueSpec.Names {
		if r.pattern.MatchString(ident.Name) && valueSpec.Values != nil {
			// const foo, bar = "same value"
			if len(valueSpec.Values) <= index {
				index = len(valueSpec.Values) - 1
			}
			if val, err := gosec.GetString(valueSpec.Values[index]); err == nil {
				if r.ignoreEntropy || (!r.ignoreEntropy && r.isHighEntropyString(val)) {
					return gosec.NewIssue(ctx, valueSpec, r.ID(), r.What, r.Severity, r.Confidence), nil
				}
			}
		}
	}
	return nil, nil
}

func (r *credentials) matchEqualityCheck(binaryExpr *ast.BinaryExpr, ctx *gosec.Context) (*gosec.Issue, error) {
	if binaryExpr.Op == token.EQL || binaryExpr.Op == token.NEQ {
		if ident, ok := binaryExpr.X.(*ast.Ident); ok {
			if r.pattern.MatchString(ident.Name) {
				if val, err := gosec.GetString(binaryExpr.Y); err == nil {
					if r.ignoreEntropy || (!r.ignoreEntropy && r.isHighEntropyString(val)) {
						return gosec.NewIssue(ctx, binaryExpr, r.ID(), r.What, r.Severity, r.Confidence), nil
					}
				}
			}
		}
	}
	return nil, nil
}

// NewHardcodedCredentials attempts to find high entropy string constants being
// assigned to variables that appear to be related to credentials.
func NewHardcodedCredentials(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	pattern := `(?i)passwd|pass|password|pwd|secret|token|pw|apiKey|bearer|cred`
	entropyThreshold := 80.0
	perCharThreshold := 3.0
	ignoreEntropy := false
	truncateString := 16
	if val, ok := conf["G101"]; ok {
		conf := val.(map[string]interface{})
		if configPattern, ok := conf["pattern"]; ok {
			if cfgPattern, ok := configPattern.(string); ok {
				pattern = cfgPattern
			}
		}
		if configIgnoreEntropy, ok := conf["ignore_entropy"]; ok {
			if cfgIgnoreEntropy, ok := configIgnoreEntropy.(bool); ok {
				ignoreEntropy = cfgIgnoreEntropy
			}
		}
		if configEntropyThreshold, ok := conf["entropy_threshold"]; ok {
			if cfgEntropyThreshold, ok := configEntropyThreshold.(string); ok {
				if parsedNum, err := strconv.ParseFloat(cfgEntropyThreshold, 64); err == nil {
					entropyThreshold = parsedNum
				}
			}
		}
		if configCharThreshold, ok := conf["per_char_threshold"]; ok {
			if cfgCharThreshold, ok := configCharThreshold.(string); ok {
				if parsedNum, err := strconv.ParseFloat(cfgCharThreshold, 64); err == nil {
					perCharThreshold = parsedNum
				}
			}
		}
		if configTruncate, ok := conf["truncate"]; ok {
			if cfgTruncate, ok := configTruncate.(string); ok {
				if parsedInt, err := strconv.Atoi(cfgTruncate); err == nil {
					truncateString = parsedInt
				}
			}
		}
	}

	return &credentials{
		pattern:          regexp.MustCompile(pattern),
		entropyThreshold: entropyThreshold,
		perCharThreshold: perCharThreshold,
		ignoreEntropy:    ignoreEntropy,
		truncate:         truncateString,
		MetaData: gosec.MetaData{
			ID:         id,
			What:       "Potential hardcoded credentials",
			Confidence: gosec.Low,
			Severity:   gosec.High,
		},
	}, []ast.Node{(*ast.AssignStmt)(nil), (*ast.ValueSpec)(nil), (*ast.BinaryExpr)(nil)}
}
