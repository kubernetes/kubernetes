/*
Copyright 2021 The Kubernetes Authors.

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

package request

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/types"
)

func TestAuditIDFrom(t *testing.T) {
	tests := []struct {
		name            string
		auditID         string
		auditIDExpected string
		expected        bool
	}{
		{
			name:            "empty audit ID",
			auditID:         "",
			auditIDExpected: "",
			expected:        false,
		},
		{
			name:            "non empty audit ID",
			auditID:         "foo-bar-baz",
			auditIDExpected: "foo-bar-baz",
			expected:        true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			parent := context.TODO()
			ctx := WithAuditID(parent, types.UID(test.auditID))

			// for an empty audit ID we don't expect a copy of the parent context.
			if len(test.auditID) == 0 && parent != ctx {
				t.Error("expected no copy of the parent context with an empty audit ID")
			}

			value, ok := AuditIDFrom(ctx)
			if test.expected != ok {
				t.Errorf("expected AuditIDFrom to return: %t, but got: %t", test.expected, ok)
			}

			auditIDGot := string(value)
			if test.auditIDExpected != auditIDGot {
				t.Errorf("expected audit ID: %q, but got: %q", test.auditIDExpected, auditIDGot)
			}
		})
	}
}
