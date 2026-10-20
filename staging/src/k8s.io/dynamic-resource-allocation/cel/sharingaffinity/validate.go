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

// CompileError wraps an apiserver CEL Error, preserving its invalid/internal
// classification and compiler diagnostics for admission field errors.
type CompileError struct {
	Err error
}

func (e *CompileError) Error() string { return e.Err.Error() }
func (e *CompileError) Unwrap() error { return e.Err }

func newCompileError(kind apiservercel.ErrorType, err error) *CompileError {
	return &CompileError{Err: &apiservercel.Error{Type: kind, Detail: err.Error(), Cause: err}}
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

// ValidateExtractorSet validates definitions and combined estimated cost, with
// no per-slice entry limit. It can therefore validate complete pool metadata.
// Runtime uses the same validation with its bounded compilation cache.
func ValidateExtractorSet(extractors []resourceapi.SharingAffinityExtractor) error {
	_, err := compileExtractorSet(extractors, storedEnv, compileExpression)
	return err
}

// ValidatePublishedExtractorSet validates complete pool metadata before publication.
// stored marks unchanged expressions on the very object being updated, matching
// admission's per-object ratcheting. nil means all expressions are new.
func ValidatePublishedExtractorSet(extractors []resourceapi.SharingAffinityExtractor, stored []bool) error {
	if stored != nil && len(stored) != len(extractors) {
		return fmt.Errorf("%w: stored markers length %d != extractors %d", ErrInvalidDefinition, len(stored), len(extractors))
	}
	_, err := compileExtractorSet(extractors, publishedEnv(stored), compileExpression)
	return err
}

type envSelector func(int) (environment.Type, bool)

func storedEnv(int) (environment.Type, bool) {
	return environment.StoredExpressions, false
}

func publishedEnv(stored []bool) envSelector {
	return func(i int) (environment.Type, bool) {
		if stored != nil && stored[i] {
			return environment.StoredExpressions, true
		}
		return environment.NewExpressions, false
	}
}

type compiledExtractor struct {
	name    string
	matcher matcher
	program *compiledExpression
}

func compileExtractorSet(extractors []resourceapi.SharingAffinityExtractor, env envSelector, compile func(string, environment.Type) (*compiledExpression, error)) ([]compiledExtractor, error) {
	names := make(map[string]struct{}, len(extractors))
	for _, extractor := range extractors {
		if _, exists := names[extractor.Name]; exists {
			return nil, &Error{Kind: ErrInvalidDefinition, Extractor: extractor.Name, ConfigIndex: -1, Err: fmt.Errorf("extractor names must be unique")}
		}
		names[extractor.Name] = struct{}{}
	}
	compiled := make([]compiledExtractor, 0, len(extractors))
	remaining := uint64(resourceapi.SharingAffinityCELMaxCost)
	for i, extractor := range extractors {
		envType, tolerateCompileError := env(i)
		program, err := compile(extractor.Expression, envType)
		if err != nil {
			// Admission preserves unchanged expressions that no longer compile;
			// they contribute zero estimated cost. Runtime never tolerates this.
			if tolerateCompileError {
				continue
			}
			return nil, &Error{Kind: ErrInvalidDefinition, Extractor: extractor.Name, ConfigIndex: -1,
				Err: &redactedError{summary: "CEL expression could not be compiled", cause: err}}
		}
		// Subtract rather than summing so even saturated CEL estimates cannot
		// overflow and turn an invalid pool into a valid one.
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
