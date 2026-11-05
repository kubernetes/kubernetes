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
	"net"
	"testing"

	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestValidateRuntimeOptions(t *testing.T) {
	for _, tc := range []struct {
		name    string
		options map[string]string
		allowed []string
		wantErr string
	}{
		{name: "nil options"},
		{name: "empty options", options: map[string]string{}},
		{name: "missing allowlist", options: map[string]string{"tcp": "close"}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "empty allowlist", options: map[string]string{"tcp": "close"}, allowed: []string{}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "exact allowed key", options: map[string]string{"tcp": "close"}, allowed: []string{"tcp"}},
		{name: "other key denied", options: map[string]string{"tcp": "close", "device-map": "0"}, allowed: []string{"tcp"}, wantErr: `runtime option "device-map" is not allowed`},
		{name: "no prefix matching", options: map[string]string{"tcp-established": "true"}, allowed: []string{"tcp"}, wantErr: `runtime option "tcp-established" is not allowed`},
		{name: "no wildcard matching", options: map[string]string{"tcp": "close"}, allowed: []string{"*"}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "deterministic rejected key", options: map[string]string{"z": "sensitive-value", "a": "sensitive-value"}, wantErr: `runtime option "a" is not allowed`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateRuntimeOptions(tc.options, tc.allowed)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tc.wantErr)
			require.NotContains(t, err.Error(), "sensitive-value")
		})
	}
}

func TestRuntimeOptionPolicyUnavailable(t *testing.T) {
	resource := schema.GroupResource{Group: "node.k8s.io", Resource: "runtimeclasses"}
	for _, tc := range []struct {
		name      string
		err       error
		retryable bool
	}{
		{name: "nil"},
		{name: "policy denial", err: errors.New("runtime option is not allowed")},
		{name: "timeout", err: apierrors.NewTimeoutError("timeout", 1), retryable: true},
		{name: "server timeout", err: apierrors.NewServerTimeout(resource, "get", 1), retryable: true},
		{name: "service unavailable", err: apierrors.NewServiceUnavailable("unavailable"), retryable: true},
		{name: "too many requests", err: apierrors.NewTooManyRequests("busy", 1), retryable: true},
		{name: "internal server error", err: apierrors.NewInternalError(errors.New("server failure")), retryable: true},
		{name: "context deadline", err: context.DeadlineExceeded, retryable: true},
		{name: "context canceled", err: context.Canceled, retryable: true},
		{name: "network timeout", err: &net.DNSError{Err: "timeout", IsTimeout: true}, retryable: true},
		{name: "connection refused", err: &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}, retryable: true},
		{name: "connection closed", err: io.EOF, retryable: true},
		{name: "unexpected connection closed", err: io.ErrUnexpectedEOF, retryable: true},
		{name: "class not found", err: apierrors.NewNotFound(resource, "runtime")},
		{name: "forbidden", err: apierrors.NewForbidden(resource, "runtime", errors.New("denied"))},
		{name: "unauthorized", err: apierrors.NewUnauthorized("denied")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.retryable, RuntimeOptionPolicyUnavailable(tc.err))
			if tc.err != nil {
				require.Equal(t, tc.retryable, RuntimeOptionPolicyUnavailable(fmt.Errorf("failed to read policy: %w", tc.err)))
			}
		})
	}
}
