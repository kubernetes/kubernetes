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

package util

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"slices"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

// ValidateRuntimeOptions checks administrator-approved option keys. Values stay
// untrusted and must be validated by the selected runtime before it uses them.
func ValidateRuntimeOptions(options map[string]string, allowed []string) error {
	for _, key := range slices.Sorted(maps.Keys(options)) {
		if !slices.Contains(allowed, key) {
			return fmt.Errorf("runtime option %q is not allowed by the RuntimeClass", key)
		}
	}
	return nil
}

// RuntimeOptionPolicyUnavailable identifies temporary lookup failures that
// cannot establish whether options are allowed. Callers must retry before
// executing the checkpoint or restore instead of making that failure terminal.
func RuntimeOptionPolicyUnavailable(err error) bool {
	if apierrors.IsNotFound(err) || apierrors.IsForbidden(err) || apierrors.IsUnauthorized(err) {
		return false
	}
	if apierrors.IsTimeout(err) || apierrors.IsServerTimeout(err) ||
		apierrors.IsServiceUnavailable(err) || apierrors.IsTooManyRequests(err) ||
		apierrors.IsInternalError(err) || errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, context.Canceled) || errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var networkError net.Error
	return errors.As(err, &networkError)
}
