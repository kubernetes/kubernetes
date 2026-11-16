// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package parser declares an expression parser with support for macro
// expansion.
package parser

import (
	"math"

	"cel.dev/cel-go/common"
	"cel.dev/cel-go/common/ast"
)

// Parser encapsulates the context necessary to perform parsing for different expressions.
type Parser struct {
	options
}

// NewParser builds and returns a new Parser using the provided options.
func NewParser(opts ...Option) (*Parser, error) {
	p := &Parser{}
	p.enableHiddenAccumulatorName = true
	p.enableIdentEscapeSyntax = true
	for _, opt := range opts {
		if err := opt(&p.options); err != nil {
			return nil, err
		}
	}
	if p.errorReportingLimit == 0 {
		p.errorReportingLimit = 100
	}
	if p.maxRecursionDepth == 0 {
		p.maxRecursionDepth = 250
	}
	if p.maxRecursionDepth == -1 {
		p.maxRecursionDepth = math.MaxInt
	}
	if p.errorRecoveryTokenLookaheadLimit == 0 {
		p.errorRecoveryTokenLookaheadLimit = 256
	}
	if p.errorRecoveryLimit == 0 {
		p.errorRecoveryLimit = 30
	}
	if p.errorRecoveryLimit == -1 {
		p.errorRecoveryLimit = math.MaxInt
	}
	if p.expressionSizeCodePointLimit == 0 {
		p.expressionSizeCodePointLimit = 100_000
	}
	if p.expressionSizeCodePointLimit == -1 {
		p.expressionSizeCodePointLimit = math.MaxInt
	}
	if p.maxExpressionNodeCount == 0 {
		p.maxExpressionNodeCount = 100_000
	}
	if p.maxExpressionNodeCount == -1 {
		p.maxExpressionNodeCount = math.MaxInt
	}
	// Bool is false by default, so populateMacroCalls will be false by default
	return p, nil
}

// mustNewParser does the work of NewParser and panics if an error occurs.
//
// This function is only intended for internal use and is for backwards compatibility in Parse and
// ParseWithMacros, where we know the options will result in an error.
func mustNewParser(opts ...Option) *Parser {
	p, err := NewParser(opts...)
	if err != nil {
		panic(err)
	}
	return p
}

// Parse parses the expression represented by source and returns the result.
func (p *Parser) Parse(source common.Source) (*ast.AST, *common.Errors) {
	if p.enablePrattParser {
		return (&prattParser{options: p.options}).Parse(source)
	}
	return (&antlrParser{options: p.options}).Parse(source)
}

// reservedIds are not legal to use as variables.  We exclude them post-parse, as they *are* valid
// field names for protos, and it would complicate the grammar to distinguish the cases.
var reservedIds = map[string]struct{}{
	"as":        {},
	"break":     {},
	"const":     {},
	"continue":  {},
	"else":      {},
	"false":     {},
	"for":       {},
	"function":  {},
	"if":        {},
	"import":    {},
	"in":        {},
	"let":       {},
	"loop":      {},
	"package":   {},
	"namespace": {},
	"null":      {},
	"return":    {},
	"true":      {},
	"var":       {},
	"void":      {},
	"while":     {},
}

// Parse converts a source input a parsed expression.
// This function calls ParseWithMacros with AllMacros.
//
// Deprecated: Use NewParser().Parse() instead.
func Parse(source common.Source) (*ast.AST, *common.Errors) {
	return mustNewParser(Macros(AllMacros...)).Parse(source)
}
