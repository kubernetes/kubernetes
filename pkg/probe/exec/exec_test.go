/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package exec

import (
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
)

type FakeCmd struct {
	out []byte
	err error
}

func (f *FakeCmd) CombinedOutput() ([]byte, error) {
	return f.out, f.err
}

func (f *FakeCmd) SetDir(dir string) {}

type healthCheckTest struct {
	expectedStatus probe.Result
	expectError    bool
	output         []byte
	err            error
}

func TestExec(t *testing.T) {
	prober := New()
	fake := FakeCmd{}
	tests := []healthCheckTest{
		// Ok
		{probe.Success, false, []byte("OK"), nil},
		// Run returns error
		{probe.Unknown, true, []byte("OK, NOT"), fmt.Errorf("test error")},
		// Unhealthy
		{probe.Failure, false, []byte("Fail"), nil},
	}
	for _, test := range tests {
		fake.out = test.output
		fake.err = test.err
		status, err := prober.Probe(&fake)
		if status != test.expectedStatus {
			t.Errorf("expected %v, got %v", test.expectedStatus, status)
		}
		if err != nil && test.expectError == false {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && test.expectError == true {
			t.Errorf("unexpected non-error")
		}
	}
}
