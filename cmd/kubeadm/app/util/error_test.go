/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

func TestCheckErr(t *testing.T) {
	var codeReturned int
	errHandle := func(err string, code int) {
		codeReturned = code
	}

	var tokenTest = []struct {
		e        error
		expected int
	}{
		{nil, 0},
		{fmt.Errorf(""), DefaultErrorExitCode},
		{&preflight.Error{}, PreFlightExitCode},
	}

	for _, rt := range tokenTest {
		codeReturned = 0
		checkErr("", rt.e, errHandle)
		if codeReturned != rt.expected {
			t.Errorf(
				"failed checkErr:\n\texpected: %d\n\t  actual: %d",
				rt.expected,
				codeReturned,
			)
		}
	}
}
