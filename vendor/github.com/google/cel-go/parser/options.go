// Copyright 2021 Google LLC
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

package parser

import "fmt"

type options struct {
	maxRecursionDepth                int
	errorReportingLimit              int
	errorRecoveryTokenLookaheadLimit int
	errorRecoveryLimit               int
	expressionSizeCodePointLimit     int
	macros                           map[string]Macro
	populateMacroCalls               bool
	enableOptionalSyntax             bool
}

// Option configures the behavior of the parser.
type Option func(*options) error

// MaxRecursionDepth limits the maximum depth the parser will attempt to parse the expression before giving up.
func MaxRecursionDepth(limit int) Option {
	return func(opts *options) error {
		if limit < -1 {
			return fmt.Errorf("max recursion depth must be greater than or equal to -1: %d", limit)
		}
		opts.maxRecursionDepth = limit
		return nil
	}
}

// ErrorRecoveryLookaheadTokenLimit limits the number of lexer tokens that may be considered during error recovery.
//
// Error recovery often involves looking ahead in the input to determine if there's a point at which parsing may
// successfully resume. In some pathological cases, the parser can look through quite a large set of input which
// in turn generates a lot of back-tracking and performance degredation.
//
// The limit must be >= 1, and is recommended to be less than the default of 256.
func ErrorRecoveryLookaheadTokenLimit(limit int) Option {
	return func(opts *options) error {
		if limit < 1 {
			return fmt.Errorf("error recovery lookahead token limit must be at least 1: %d", limit)
		}
		opts.errorRecoveryTokenLookaheadLimit = limit
		return nil
	}
}

// ErrorRecoveryLimit limits the number of attempts the parser will perform to recover from an error.
func ErrorRecoveryLimit(limit int) Option {
	return func(opts *options) error {
		if limit < -1 {
			return fmt.Errorf("error recovery limit must be greater than or equal to -1: %d", limit)
		}
		opts.errorRecoveryLimit = limit
		return nil
	}
}

// ErrorReportingLimit limits the number of syntax error reports before terminating parsing.
//
// The limit must be at least 1. If unset, the limit will be 100.
func ErrorReportingLimit(limit int) Option {
	return func(opts *options) error {
		if limit < 1 {
			return fmt.Errorf("error reporting limit must be at least 1: %d", limit)
		}
		opts.errorReportingLimit = limit
		return nil
	}
}

// ExpressionSizeCodePointLimit is an option which limits the maximum code point count of an
// expression.
func ExpressionSizeCodePointLimit(expressionSizeCodePointLimit int) Option {
	return func(opts *options) error {
		if expressionSizeCodePointLimit < -1 {
			return fmt.Errorf("expression size code point limit must be greater than or equal to -1: %d", expressionSizeCodePointLimit)
		}
		opts.expressionSizeCodePointLimit = expressionSizeCodePointLimit
		return nil
	}
}

// Macros adds the given macros to the parser.
func Macros(macros ...Macro) Option {
	return func(opts *options) error {
		for _, m := range macros {
			if m != nil {
				if opts.macros == nil {
					opts.macros = make(map[string]Macro)
				}
				opts.macros[m.MacroKey()] = m
			}
		}
		return nil
	}
}

// PopulateMacroCalls ensures that the original call signatures replaced by expanded macros
// are preserved in the `SourceInfo` of parse result.
func PopulateMacroCalls(populateMacroCalls bool) Option {
	return func(opts *options) error {
		opts.populateMacroCalls = populateMacroCalls
		return nil
	}
}

// EnableOptionalSyntax enables syntax for optional field and index selection.
func EnableOptionalSyntax(optionalSyntax bool) Option {
	return func(opts *options) error {
		opts.enableOptionalSyntax = optionalSyntax
		return nil
	}
}
