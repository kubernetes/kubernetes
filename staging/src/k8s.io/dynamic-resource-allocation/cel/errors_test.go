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
	"testing"
)

func TestEnhanceRuntimeError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "nil error returns nil",
			err:  nil,
			want: "",
		},
		{
			name: "no such key error gets enhanced with usage hint",
			err:  errors.New("no such key: sharingStrategy"),
			want: "no such key: sharingStrategy. consider using CEL optional chaining (.? followed by orValue()) or guarding the check with has() for optional fields",
		},
		{
			name: "other error is returned unchanged",
			err:  errors.New("some other error"),
			want: "some other error",
		},
		{
			name: "wrapped no such key error gets enhanced",
			err:  errors.New("context: no such key: fieldY"),
			want: "context: no such key: fieldY. consider using CEL optional chaining (.? followed by orValue()) or guarding the check with has() for optional fields",
		},
		{
			name: "error with hint text but not enhancedError type still gets enhanced",
			err:  errors.New("no such key: fieldZ. consider using CEL optional chaining"),
			want: "no such key: fieldZ. consider using CEL optional chaining. consider using CEL optional chaining (.? followed by orValue()) or guarding the check with has() for optional fields",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EnhanceRuntimeError(tt.err)
			if tt.want == "" {
				if got != nil {
					t.Errorf("EnhanceRuntimeError() = %v, want nil", got)
				}
				return
			}
			if got == nil {
				t.Errorf("EnhanceRuntimeError() = nil, want %q", tt.want)
				return
			}
			if got.Error() != tt.want {
				t.Errorf("EnhanceRuntimeError() = %q, want %q", got.Error(), tt.want)
			}
		})
	}
}

func TestEnhanceRuntimeErrorIdempotent(t *testing.T) {
	originalErr := errors.New("no such key: fieldA")
	enhancedOnce := EnhanceRuntimeError(originalErr)
	enhancedTwice := EnhanceRuntimeError(enhancedOnce)

	if enhancedOnce.Error() != enhancedTwice.Error() {
		t.Errorf("EnhanceRuntimeError is not idempotent: first=%q, second=%q", enhancedOnce.Error(), enhancedTwice.Error())
	}
}
