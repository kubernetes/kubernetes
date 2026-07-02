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

package container

import (
	"errors"
	"fmt"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

func TestRestoreErrorReason(t *testing.T) {
	busy := &RestoreError{Reason: "RestoreInProgress", Err: errors.New("busy")}
	failed := &RestoreError{Reason: "PodSpecMismatch", Err: errors.New("mismatch")}
	syncResult := PodSyncResult{}
	syncResult.Fail(busy)
	for _, tc := range []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"unrelated", errors.New("other"), ""},
		{"direct", busy, "RestoreInProgress"},
		{"wrapped", fmt.Errorf("restore: %w", failed), "PodSpecMismatch"},
		{"sync result", syncResult.Error(), "RestoreInProgress"},
		{"nested aggregate", fmt.Errorf("outer: %w", utilerrors.NewAggregate([]error{errors.New("other"), errors.Join(errors.New("first"), utilerrors.NewAggregate([]error{failed}))})), "PodSpecMismatch"},
		{"first reason", errors.Join(busy, failed), "RestoreInProgress"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := RestoreErrorReason(tc.err); got != tc.want {
				t.Fatalf("reason = %q, want %q", got, tc.want)
			}
		})
	}
}
