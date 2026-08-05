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

package cel

import (
	"errors"
	"fmt"
	"strings"
)

// enhancedError wraps an error with additional CEL hint for "no such key" errors.
// It is used to ensure EnhanceRuntimeError is idempotent.
type enhancedError struct {
	error
}

func (e *enhancedError) Unwrap() error { return e.error }

// EnhanceRuntimeError provides a more helpful error message for "no such key" errors.
// It detects CEL runtime errors that occur when accessing non-existent map keys
// and adds a hint about using optional field access patterns.
// For all other errors, it returns the error unchanged.
//
// Note: This function focuses on enhancing the CEL error message itself.
// Additional context (like device ID) should be added by the caller when wrapping the error.
func EnhanceRuntimeError(err error) error {
	if err == nil {
		return nil
	}

	// Check if already enhanced using type assertion for idempotency.
	var enhanced *enhancedError
	if errors.As(err, &enhanced) {
		return err
	}

	errMsg := err.Error()
	// Check if this is a "no such key" error.
	// Using Contains instead of HasPrefix to handle wrapped errors.
	if strings.Contains(errMsg, "no such key:") {
		return &enhancedError{
			error: fmt.Errorf("%w. consider using CEL optional chaining (.? followed by orValue()) or guarding the check with has() for optional fields", err),
		}
	}

	// Not a "no such key" error, return unchanged.
	return err
}
