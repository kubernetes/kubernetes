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

package webhook

import (
	"context"
	"errors"
	"io"
	"syscall"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestShouldRetryAdmissionWebhookCall(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		wantRetry bool
	}{
		{
			name:      "connection reset",
			err:       syscall.ECONNRESET,
			wantRetry: true,
		},
		{
			name:      "eof",
			err:       io.EOF,
			wantRetry: true,
		},
		{
			name:      "unexpected eof",
			err:       io.ErrUnexpectedEOF,
			wantRetry: true,
		},
		{
			name:      "http2 connection lost",
			err:       errors.New("http2: client connection lost"),
			wantRetry: true,
		},
		{
			name:      "http response error",
			err:       errors.New("the server rejected this request"),
			wantRetry: false,
		},
		{
			name:      "context canceled",
			err:       context.Canceled,
			wantRetry: false,
		},
		{
			name:      "context deadline exceeded",
			err:       context.DeadlineExceeded,
			wantRetry: false,
		},
		{
			name:      "status error",
			err:       apierrors.FromObject(&metav1.Status{Status: metav1.StatusFailure, Code: 500}),
			wantRetry: false,
		},
		{
			name:      "internal server error",
			err:       apierrors.NewInternalError(errors.New("boom")),
			wantRetry: false,
		},
		{
			name:      "too many requests",
			err:       apierrors.NewTooManyRequests("rate limited", 1),
			wantRetry: false,
		},
		{
			name:      "forbidden",
			err:       apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, "pod", errors.New("denied")),
			wantRetry: false,
		},
		{
			name:      "nil",
			err:       nil,
			wantRetry: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldRetryAdmissionWebhookCall(tt.err); got != tt.wantRetry {
				t.Fatalf("expected retry=%t, got %t", tt.wantRetry, got)
			}
		})
	}
}

func TestWithAdmissionWebhookTransportErrorRetryHonorsContextDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	attempts := 0
	err := WithAdmissionWebhookTransportErrorRetry(ctx, func() error {
		attempts++
		return io.EOF
	})
	if err == nil {
		t.Fatal("expected retry to return an error")
	}
	if attempts != 1 {
		t.Fatalf("expected retry to stop after 1 attempt once the context deadline expires, got %d attempts", attempts)
	}
}
