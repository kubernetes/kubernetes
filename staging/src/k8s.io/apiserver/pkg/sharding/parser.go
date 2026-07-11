/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package sharding

import (
	"fmt"
	"strings"

	celparser "github.com/google/cel-go/parser"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"

	apisharding "k8s.io/apimachinery/pkg/sharding"
)

// Parse parses a CEL-based shard selector expression into a Selector.
//
// The expression format is:
//
//	shardRange(object.metadata.uid, '0x0', '0x8000000000000000')
//	shardRange(object.metadata.uid, '0x0', '0x8000000000000000') || shardRange(...)
//
// Only the shardRange() function and || operator are permitted. The CEL
// expression is parsed but never evaluated — the AST is walked to extract
// shard range requirements.
func Parse(expr string) (apisharding.Selector, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return nil, fmt.Errorf("empty shard selector is not allowed; omit the parameter for unfiltered lists")
	}

	p, err := celparser.NewParser(celparser.Macros( /* no macros */ ))
	if err != nil {
		return nil, fmt.Errorf("failed to create CEL parser: %w", err)
	}

	parsed, errs := p.Parse(common.NewTextSource(expr))
	if errs != nil && len(errs.GetErrors()) > 0 {
		return nil, fmt.Errorf("CEL parse error: %s", errs.GetErrors()[0].Message)
	}

	reqs, err := walkExpr(parsed.Expr())
	if err != nil {
		return nil, err
	}

	// Validate that all requirements use the same field key.
	for i := 1; i < len(reqs); i++ {
		if reqs[i].Key != reqs[0].Key {
			return nil, fmt.Errorf("all shard ranges must use the same field, got %q and %q", reqs[0].Key, reqs[i].Key)
		}
	}

	return apisharding.NewSelector(reqs...), nil
}

// walkExpr recursively walks the CEL AST and extracts ShardRangeRequirement values.
// The root expression must be either a shardRange() call or a chain of || operators
// combining shardRange() calls.
func walkExpr(e ast.Expr) ([]apisharding.ShardRangeRequirement, error) {
	if e.Kind() == ast.CallKind {
		call := e.AsCall()
		fn := call.FunctionName()

		if fn == operators.LogicalOr {
			// _||_ operator: recurse into both sides
			args := call.Args()
			if len(args) != 2 {
				return nil, fmt.Errorf("|| operator requires exactly 2 arguments")
			}
			left, err := walkExpr(args[0])
			if err != nil {
				return nil, err
			}
			right, err := walkExpr(args[1])
			if err != nil {
				return nil, err
			}
			return append(left, right...), nil
		}

		if fn == "shardRange" {
			req, err := parseShardRangeCall(call)
			if err != nil {
				return nil, err
			}
			return []apisharding.ShardRangeRequirement{req}, nil
		}

		return nil, fmt.Errorf("unsupported function %q; only shardRange() and || are allowed", fn)
	}

	return nil, fmt.Errorf("unexpected expression kind %v; expected shardRange() call or || operator", e.Kind())
}

// parseShardRangeCall extracts a ShardRangeRequirement from a shardRange(field, start, end) call.
func parseShardRangeCall(call ast.CallExpr) (apisharding.ShardRangeRequirement, error) {
	args := call.Args()
	if len(args) != 3 {
		return apisharding.ShardRangeRequirement{}, fmt.Errorf("shardRange() requires exactly 3 arguments, got %d", len(args))
	}

	// Arg 0: field path (select chain like object.metadata.uid)
	fieldPath, err := extractFieldPath(args[0])
	if err != nil {
		return apisharding.ShardRangeRequirement{}, fmt.Errorf("shardRange() first argument: %w", err)
	}

	// Validate field path
	switch fieldPath {
	case "object.metadata.uid", "object.metadata.namespace":
		// ok
	default:
		return apisharding.ShardRangeRequirement{}, fmt.Errorf("unsupported field path %q; supported: object.metadata.uid, object.metadata.namespace", fieldPath)
	}

	// Arg 1: hex start (string literal)
	hexStart, err := extractHexLiteral(args[1], "hexStart")
	if err != nil {
		return apisharding.ShardRangeRequirement{}, err
	}

	// Arg 2: hex end (string literal)
	hexEnd, err := extractHexLiteral(args[2], "hexEnd")
	if err != nil {
		return apisharding.ShardRangeRequirement{}, err
	}

	// Validate start < end using the canonical hex comparison.
	if !apisharding.HexLess(hexStart, hexEnd) {
		return apisharding.ShardRangeRequirement{}, fmt.Errorf("shard range start %s must be less than end %s", hexStart, hexEnd)
	}

	return apisharding.ShardRangeRequirement{
		Key:   fieldPath,
		Start: hexStart,
		End:   hexEnd,
	}, nil
}

// extractFieldPath walks a select chain (object.metadata.uid) and returns the dot-joined path.
func extractFieldPath(e ast.Expr) (string, error) {
	switch e.Kind() {
	case ast.IdentKind:
		return e.AsIdent(), nil
	case ast.SelectKind:
		sel := e.AsSelect()
		operand, err := extractFieldPath(sel.Operand())
		if err != nil {
			return "", err
		}
		return operand + "." + sel.FieldName(), nil
	default:
		return "", fmt.Errorf("expected field path (e.g. object.metadata.uid), got expression kind %v", e.Kind())
	}
}

// extractHexLiteral extracts a hex string from a CEL string literal.
// The literal must be a single-quoted string like '0xff' with a 0x prefix.
func extractHexLiteral(e ast.Expr, name string) (string, error) {
	if e.Kind() != ast.LiteralKind {
		return "", fmt.Errorf("%s must be a string literal (e.g. '0xff'), got expression kind %v", name, e.Kind())
	}

	val := e.AsLiteral()
	s, ok := val.Value().(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string literal, got %T", name, val.Value())
	}

	if !strings.HasPrefix(s, "0x") {
		return "", fmt.Errorf("%s must have '0x' prefix, got %q", name, s)
	}

	hex := s[2:]
	if hex == "" {
		return "", fmt.Errorf("%s: hex value is required after '0x'", name)
	}
	if len(hex) > 17 {
		return "", fmt.Errorf("%s: hex value too long (%d chars, max 17): %q", name, len(hex), hex)
	}

	for _, c := range hex {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return "", fmt.Errorf("%s: invalid hex character %q in %q", name, string(c), s)
		}
	}

	// Require exactly 16 hex digits, except for the special case
	// "0x10000000000000000" (2^64) which is the exclusive upper bound.
	if len(hex) == 17 {
		if s != "0x10000000000000000" {
			return "", fmt.Errorf("%s: 17-digit hex value must be '0x10000000000000000' (2^64), got %q", name, s)
		}
	} else if len(hex) != 16 {
		padded := "0x" + strings.Repeat("0", 16-len(hex)) + hex
		return "", fmt.Errorf("%s must be a 0x-prefixed 16-digit hex value, got %q (did you mean %q?)", name, s, padded)
	}

	return s, nil
}
