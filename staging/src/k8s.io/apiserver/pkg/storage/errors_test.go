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

package storage

import (
	"fmt"
	"testing"
)

func TestIsErrCodeWithWrappedErrors(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		check  func(error) bool
		expect bool
	}{
		{
			name:   "wrapped NotFound",
			err:    fmt.Errorf("context: %w", NewKeyNotFoundError("/foo", 0)),
			check:  IsNotFound,
			expect: true,
		},
		{
			name:   "wrapped Conflict",
			err:    fmt.Errorf("context: %w", NewResourceVersionConflictsError("/foo", 1)),
			check:  IsConflict,
			expect: true,
		},
		{
			name:   "wrapped KeyExists",
			err:    fmt.Errorf("context: %w", NewKeyExistsError("/foo", 0)),
			check:  IsExist,
			expect: true,
		},
		{
			name:   "wrapped error with wrong code does not match",
			err:    fmt.Errorf("context: %w", NewKeyNotFoundError("/foo", 0)),
			check:  IsConflict,
			expect: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.check(tc.err); got != tc.expect {
				t.Errorf("got %v, want %v", got, tc.expect)
			}
		})
	}
}
