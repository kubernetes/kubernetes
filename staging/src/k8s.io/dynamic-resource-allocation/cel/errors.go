/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"strings"
)

// EnhanceRuntimeError provides a more helpful error message for "no such key" errors.
// For all other errors, it returns the error unchanged.
func EnhanceRuntimeError(err error) error {
	if err == nil {
		return nil
	}

	if isNoSuchKeyError(err) {
		return fmt.Errorf("%s\n\nSee https://pkg.go.dev/k8s.io/api/resource/v1#CELDeviceSelector for documentation on CEL device selectors and how to handle optional fields.",
			err)
	}

	// Not a "no such key" error, return unchanged.
	return err
}

// isNoSuchKeyError checks if the error is a "no such key" error from cel-go.
//
// The error originates from cel-go's interpreter.resolutionError (an internal type),
// so we cannot use errors.Is() for type checking. Instead, we check the error message format.
//
// The error format is "no such key: <key>" from:
// https://github.com/google/cel-go/blob/v0.26.0/interpreter/attributes.go#L1422
func isNoSuchKeyError(err error) bool {
	if err == nil {
		return false
	}
	return strings.HasPrefix(err.Error(), "no such key:")
}
