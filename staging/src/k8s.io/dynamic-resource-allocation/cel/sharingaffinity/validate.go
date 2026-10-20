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
