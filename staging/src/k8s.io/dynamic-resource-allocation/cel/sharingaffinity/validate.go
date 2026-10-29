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

package sharingaffinity

import (
	"fmt"

	resourceapi "k8s.io/api/resource/v1"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

// CompileResult contains the estimated maximum cost or a compilation error.
// Callers enforce expression length and combine costs at their own API boundary.
type CompileResult struct {
	MaxCost uint64
	Error   error
}

func newCompileError(kind apiservercel.ErrorType, err error) error {
	return &apiservercel.Error{Type: kind, Detail: err.Error(), Cause: err}
}

// ValidateExpression checks syntax, types and program construction in the
// requested compatibility environment, and estimates cost using runtime semantics.
func ValidateExpression(expression string, envType environment.Type) CompileResult {
	compiled, err := compileExpression(expression, envType)
	if err != nil {
		return CompileResult{Error: err}
	}
	return CompileResult{MaxCost: compiled.maxCost}
}

// ValidateExpressions checks CEL compilation and combined estimated cost in the
// stored-expression environment. Extractor names are used only for diagnostics;
// callers validate name uniqueness and other non-CEL constraints separately.
func ValidateExpressions(extractors []resourceapi.SharingAffinityExtractor) error {
	_, err := compileExpressions(extractors, compileExpression)
	return err
}

type compiledExtractor struct {
	name    string
	matcher matcher
	program *compiledExpression
}

func compileExpressions(extractors []resourceapi.SharingAffinityExtractor, compile func(string, environment.Type) (*compiledExpression, error)) ([]compiledExtractor, error) {
	compiled := make([]compiledExtractor, 0, len(extractors))
	remaining := uint64(resourceapi.SharingAffinityCELMaxCost)
	for _, extractor := range extractors {
		program, err := compile(extractor.Expression, environment.StoredExpressions)
		if err != nil {
			return nil, &Error{Kind: ErrInvalidDefinition, Extractor: extractor.Name, ConfigIndex: -1,
				Err: &redactedError{summary: "CEL expression could not be compiled", cause: err}}
		}
		if program.maxCost > remaining {
			return nil, &Error{Kind: ErrInvalidDefinition, Extractor: extractor.Name, ConfigIndex: -1,
				Err: fmt.Errorf("combined estimated cost exceeds %d: %w", resourceapi.SharingAffinityCELMaxCost, ErrCostExceeded)}
		}
		remaining -= program.maxCost
		compiled = append(compiled, compiledExtractor{
			name: extractor.Name, matcher: matcher{apiGroup: extractor.APIGroup, kind: extractor.Kind}, program: program,
		})
	}
	return compiled, nil
}
