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

package value

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

type classifiedError struct{ corrupt bool }

func (e classifiedError) Error() string        { return "classified" }
func (e classifiedError) DataCorruption() bool { return e.corrupt }

func TestIsDataCorruption(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"an error that says nothing counts as corruption", errors.New("decryption failed"), true},
		{"a context error that says nothing counts as corruption", context.Canceled, true},
		{"an error that says it is corruption", classifiedError{corrupt: true}, true},
		{"an error that says it isn't", classifiedError{corrupt: false}, false},
		{"the answer survives wrapping", fmt.Errorf("reading key: %w", classifiedError{corrupt: false}), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsDataCorruption(tc.err); got != tc.want {
				t.Fatalf("IsDataCorruption(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}
