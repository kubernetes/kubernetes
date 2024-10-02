/*
Copyright 2023 The Kubernetes Authors.

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

package routine

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestExecutionWithRoutine(t *testing.T) {
	var executed bool
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t := TaskFrom(r.Context())
		t.Func = func() {
			executed = true
		}
	})
	ts := httptest.NewServer(WithRoutine(handler, func(_ *http.Request, _ *request.RequestInfo) bool { return true }))
	defer ts.Close()

	_, err := http.Get(ts.URL)
	if err != nil {
		t.Errorf("got unexpected error on request: %v", err)
	}
	if !executed {
		t.Error("expected to execute")
	}
}

func TestAppendTask(t *testing.T) {
	tests := []struct {
		name          string
		existingTask  bool
		taskAppended  bool
		shouldExecute bool
	}{
		{
			name:          "append task when existing",
			existingTask:  true,
			taskAppended:  true,
			shouldExecute: true,
		},
		{
			name:          "not append task when no existing tasks",
			existingTask:  false,
			taskAppended:  false,
			shouldExecute: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var executed, appended bool
			taskToAppend := func() {
				executed = true
			}
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				ctx := r.Context()
				if test.existingTask {
					t := TaskFrom(ctx)
					t.Func = func() {}
				}

				appended = AppendTask(ctx, &Task{taskToAppend})
			})
			ts := httptest.NewServer(WithRoutine(handler, func(_ *http.Request, _ *request.RequestInfo) bool { return true }))
			defer ts.Close()

			_, err := http.Get(ts.URL)
			if err != nil {
				t.Errorf("got unexpected error on request: %v", err)
			}

			if test.taskAppended != appended {
				t.Errorf("expected taskAppended: %t, got: %t", test.taskAppended, executed)
			}

			if test.shouldExecute != executed {
				t.Errorf("expected shouldExecute: %t, got: %t", test.shouldExecute, executed)
			}
		})
	}
}
