/*
Copyright 2014 The Kubernetes Authors.

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

package componentstatus

import (
	"errors"
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/probe"
)

func matchError(data []byte) error {
	if string(data) != "bar" {
		return errors.New("match error")
	}
	return nil
}

func TestValidate(t *testing.T) {
	tests := []struct {
		probeResult probe.Result
		probeData   string
		probeErr    error

		expectResult probe.Result
		expectData   string
		expectErr    bool

		validator ValidatorFn
	}{
		{probe.Unknown, "", fmt.Errorf("probe error"), probe.Unknown, "", true, nil},
		{probe.Failure, "", nil, probe.Failure, "", false, nil},
		{probe.Success, "foo", nil, probe.Failure, "foo", true, matchError},
		{probe.Success, "foo", nil, probe.Success, "foo", false, nil},
	}

	s := Server{Addr: "foo.com", Port: 8080, Path: "/healthz"}

	for _, test := range tests {
		fakeProber := &fakeHttpProber{
			result: test.probeResult,
			body:   test.probeData,
			err:    test.probeErr,
		}

		s.Validate = test.validator
		result, data, err := s.DoServerCheck(fakeProber)
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if data != test.expectData {
			t.Errorf("expected %s, got %s", test.expectData, data)
		}
		if result != test.expectResult {
			t.Errorf("expected %s, got %s", test.expectResult, result)
		}
	}
}
