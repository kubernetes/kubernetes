/*
Copyright 2014 Google Inc. All rights reserved.

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

package health

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

type FakeExec struct {
	cmd []string
	out []byte
	err error
}

func (f *FakeExec) RunInContainer(podFullName string, uid types.UID, container string, cmd []string) ([]byte, error) {
	f.cmd = cmd
	return f.out, f.err
}

type healthCheckTest struct {
	expectedStatus Status
	probe          *api.LivenessProbe
	expectError    bool
	output         []byte
	err            error
}

func TestExec(t *testing.T) {
	fake := FakeExec{}
	checker := ExecHealthChecker{&fake}
	tests := []healthCheckTest{
		// Missing parameters
		{Unknown, &api.LivenessProbe{}, true, nil, nil},
		// Ok
		{Healthy, &api.LivenessProbe{
			Exec: &api.ExecAction{Command: []string{"ls", "-l"}},
		}, false, []byte("OK"), nil},
		// Run returns error
		{Unknown, &api.LivenessProbe{
			Exec: &api.ExecAction{
				Command: []string{"ls", "-l"},
			},
		}, true, []byte("OK, NOT"), fmt.Errorf("test error")},
		// Unhealthy
		{Unhealthy, &api.LivenessProbe{
			Exec: &api.ExecAction{Command: []string{"ls", "-l"}},
		}, false, []byte("Fail"), nil},
	}
	for _, test := range tests {
		fake.out = test.output
		fake.err = test.err
		status, err := checker.HealthCheck("test", "", api.PodStatus{}, api.Container{LivenessProbe: test.probe})
		if status != test.expectedStatus {
			t.Errorf("expected %v, got %v", test.expectedStatus, status)
		}
		if err != nil && test.expectError == false {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && test.expectError == true {
			t.Errorf("unexpected non-error")
		}
		if test.probe.Exec != nil && !reflect.DeepEqual(fake.cmd, test.probe.Exec.Command) {
			t.Errorf("expected: %v, got %v", test.probe.Exec.Command, fake.cmd)
		}
	}
}
