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
	"strings"
	"testing"
)

func TestEnhanceRuntimeError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "no such key error gets enhanced with usage hint",
			err:  errors.New("no such key: sharingStrategy"),
			want: "Consider using CEL optional chaining",
		},
		{
			name: "other error is returned unchanged",
			err:  errors.New("some other error"),
			want: "some other error",
		},
		{
			name: "nil error returns nil",
			err:  nil,
			want: "",
		},
		{
			name: "no such key with domain",
			err:  errors.New("no such key: model"),
			want: "Consider using CEL optional chaining",
		},
		{
			name: "no such key includes original message",
			err:  errors.New("no such key: fieldX"),
			want: "no such key: fieldX",
		},
		{
			name: "non-no-such-key error is unchanged",
			err:  errors.New("compilation failed"),
			want: "compilation failed",
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
				t.Errorf("EnhanceRuntimeError() = nil, want error containing %q", tt.want)
				return
			}
			if !strings.Contains(got.Error(), tt.want) {
				t.Errorf("EnhanceRuntimeError() = %v, want error containing %q", got.Error(), tt.want)
			}
		})
	}
}
