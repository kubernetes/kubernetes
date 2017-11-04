// +build linux

/*
Copyright 2017 The Kubernetes Authors.

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

package dockershim

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetSeccompSecurityOpts(t *testing.T) {
	tests := []struct {
		msg            string
		seccompProfile string
		expectedOpts   []string
	}{{
		msg:            "No security annotations",
		seccompProfile: "",
		expectedOpts:   []string{"seccomp=unconfined"},
	}, {
		msg:            "Seccomp unconfined",
		seccompProfile: "unconfined",
		expectedOpts:   []string{"seccomp=unconfined"},
	}, {
		msg:            "Seccomp default",
		seccompProfile: "docker/default",
		expectedOpts:   nil,
	}}

	for i, test := range tests {
		opts, err := getSeccompSecurityOpts(test.seccompProfile, '=')
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}

func TestLoadSeccompLocalhostProfiles(t *testing.T) {
	tests := []struct {
		msg            string
		seccompProfile string
		expectedOpts   []string
		expectErr      bool
	}{{
		msg: "Seccomp localhost/test profile",
		// We are abusing localhost for loading test seccomp profiles.
		// The profile should be an absolute path while we are using a relative one.
		seccompProfile: "localhost/fixtures/seccomp/test",
		expectedOpts:   []string{`seccomp={"foo":"bar"}`},
		expectErr:      false,
	}, {
		msg:            "Seccomp localhost/sub/subtest profile",
		seccompProfile: "localhost/fixtures/seccomp/sub/subtest",
		expectedOpts:   []string{`seccomp={"abc":"def"}`},
		expectErr:      false,
	}, {
		msg:            "Seccomp non-existent",
		seccompProfile: "localhost/fixtures/seccomp/non-existent",
		expectedOpts:   nil,
		expectErr:      true,
	}}

	for i, test := range tests {
		opts, err := getSeccompSecurityOpts(test.seccompProfile, '=')
		if test.expectErr {
			assert.Error(t, err, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
			continue
		}
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}
