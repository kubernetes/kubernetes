/*
Copyright 2025 The Kubernetes Authors.

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

package errors

import (
	"flag"
	"strings"
	"testing"
)

func TestCallStack(t *testing.T) {
	testLeafFunc(t)
}

func testLeafFunc(t *testing.T) {
	e := New("foo")
	s := StackTrace(e)
	if !strings.HasPrefix(s, "k8s.io/kubernetes/cmd/kubeadm/app/util/errors.testLeafFunc") {
		t.Fatalf("unexpected stack trace")
	}
}

func TestHandleError(t *testing.T) {
	_ = flag.CommandLine.String("v", "", "")

	tests := []struct {
		name             string
		vValue           string
		errMessage       string
		expectedContains []string
	}{
		{
			name:             "without stack trace",
			vValue:           "1",
			errMessage:       "foo",
			expectedContains: []string{"foo"},
		},
		{
			name:             "with stack trace",
			vValue:           "5",
			errMessage:       "bar",
			expectedContains: []string{"bar", "errors.TestHandleError"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_ = flag.CommandLine.Set("v", tc.vValue)
			flag.Parse()

			err := New(tc.errMessage)

			handleFunc := func(msg string, code int) {
				if code != defaultErrorExitCode {
					t.Errorf("expected error code: %d", defaultErrorExitCode)
				}
				for _, c := range tc.expectedContains {
					if !strings.Contains(msg, c) {
						t.Errorf("expected error message to contain: %s", c)
					}
				}
			}

			handleError(err, handleFunc)
		})
	}
}
