/*
Copyright 2024 The Kubernetes Authors.

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

package cel

import (
	"errors"
	"testing"

	"github.com/google/cel-go/cel"
)

func TestOutOfBudgetError(t *testing.T) {
	err := &Error{
		Type:   ErrorTypeInvalid,
		Detail: "expression out of budget",
		Cause:  ErrOutOfBudget,
	}
	if !errors.Is(err, ErrOutOfBudget) {
		t.Errorf("unexpected %v is not %v", err, ErrOutOfBudget)
	}
	if !errors.Is(err, ErrInvalid) {
		t.Errorf("unexpected %v is not %v", err, ErrInvalid)
	}
}

func TestCompilationError(t *testing.T) {
	if !errors.Is(ErrCompilation, ErrInvalid) {
		t.Errorf("unexpected %v is not %v", ErrCompilation, ErrInvalid)
	}
	issues := &cel.Issues{}
	err := &Error{
		Type:   ErrorTypeInvalid,
		Detail: "fake compilation failed",
		Cause:  NewCompilationError(issues),
	}
	if !errors.Is(err, ErrCompilation) {
		t.Errorf("unexpected %v is not %v", err, ErrCompilation)
	}
	if !errors.Is(err, ErrInvalid) {
		t.Errorf("unexpected %v is not %v", err, ErrInvalid)
	}
	var compilationErr *CompilationError
	if errors.As(err, &compilationErr); compilationErr == nil {
		t.Errorf("unexpected %v cannot be fitted into CompilationError", err)
	}
	if compilationErr.Issues != issues {
		t.Errorf("retrieved issues is not the original")
	}
}
