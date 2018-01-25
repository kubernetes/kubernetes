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

package selinux

import (
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"reflect"
	"strings"
	"testing"
)

func TestMustRunAsOptions(t *testing.T) {
	tests := map[string]struct {
		opts *extensions.SELinuxStrategyOptions
		pass bool
	}{
		"invalid opts": {
			opts: &extensions.SELinuxStrategyOptions{},
			pass: false,
		},
		"valid opts": {
			opts: &extensions.SELinuxStrategyOptions{SELinuxOptions: &api.SELinuxOptions{}},
			pass: true,
		},
	}
	for name, tc := range tests {
		_, err := NewMustRunAs(tc.opts)
		if err != nil && tc.pass {
			t.Errorf("%s expected to pass but received error %#v", name, err)
		}
		if err == nil && !tc.pass {
			t.Errorf("%s expected to fail but did not receive an error", name)
		}
	}
}

func TestMustRunAsGenerate(t *testing.T) {
	opts := &extensions.SELinuxStrategyOptions{
		SELinuxOptions: &api.SELinuxOptions{
			User:  "user",
			Role:  "role",
			Type:  "type",
			Level: "level",
		},
	}
	mustRunAs, err := NewMustRunAs(opts)
	if err != nil {
		t.Fatalf("unexpected error initializing NewMustRunAs %v", err)
	}
	generated, err := mustRunAs.Generate(nil, nil)
	if err != nil {
		t.Fatalf("unexpected error generating selinux %v", err)
	}
	if !reflect.DeepEqual(generated, opts.SELinuxOptions) {
		t.Errorf("generated selinux does not equal configured selinux")
	}
}

func TestMustRunAsValidate(t *testing.T) {
	newValidOpts := func() *api.SELinuxOptions {
		return &api.SELinuxOptions{
			User:  "user",
			Role:  "role",
			Level: "s0:c0,c6",
			Type:  "type",
		}
	}

	newValidOptsWithLevel := func(level string) *api.SELinuxOptions {
		opts := newValidOpts()
		opts.Level = level
		return opts
	}

	role := newValidOpts()
	role.Role = "invalid"

	user := newValidOpts()
	user.User = "invalid"

	seType := newValidOpts()
	seType.Type = "invalid"

	validOpts := newValidOpts()

	tests := map[string]struct {
		podSeLinux  *api.SELinuxOptions
		pspSeLinux  *api.SELinuxOptions
		expectedMsg string
	}{
		"invalid role": {
			podSeLinux:  role,
			pspSeLinux:  validOpts,
			expectedMsg: "role: Invalid value",
		},
		"invalid user": {
			podSeLinux:  user,
			pspSeLinux:  validOpts,
			expectedMsg: "user: Invalid value",
		},
		"levels are not equal": {
			podSeLinux:  newValidOptsWithLevel("s0"),
			pspSeLinux:  newValidOptsWithLevel("s0:c1,c2"),
			expectedMsg: "level: Invalid value",
		},
		"levels differ by sensitivity": {
			podSeLinux:  newValidOptsWithLevel("s0:c6"),
			pspSeLinux:  newValidOptsWithLevel("s1:c6"),
			expectedMsg: "level: Invalid value",
		},
		"levels differ by categories": {
			podSeLinux:  newValidOptsWithLevel("s0:c0,c8"),
			pspSeLinux:  newValidOptsWithLevel("s0:c1,c7"),
			expectedMsg: "level: Invalid value",
		},
		"valid": {
			podSeLinux:  validOpts,
			pspSeLinux:  validOpts,
			expectedMsg: "",
		},
		"valid with different order of categories": {
			podSeLinux:  newValidOptsWithLevel("s0:c6,c0"),
			pspSeLinux:  validOpts,
			expectedMsg: "",
		},
	}

	for name, tc := range tests {
		opts := &extensions.SELinuxStrategyOptions{
			SELinuxOptions: tc.pspSeLinux,
		}
		mustRunAs, err := NewMustRunAs(opts)
		if err != nil {
			t.Errorf("unexpected error initializing NewMustRunAs for testcase %s: %#v", name, err)
			continue
		}

		errs := mustRunAs.Validate(nil, nil, nil, tc.podSeLinux)
		//should've passed but didn't
		if len(tc.expectedMsg) == 0 && len(errs) > 0 {
			t.Errorf("%s expected no errors but received %v", name, errs)
		}
		//should've failed but didn't
		if len(tc.expectedMsg) != 0 && len(errs) == 0 {
			t.Errorf("%s expected error %s but received no errors", name, tc.expectedMsg)
		}
		//failed with additional messages
		if len(tc.expectedMsg) != 0 && len(errs) > 1 {
			t.Errorf("%s expected error %s but received multiple errors: %v", name, tc.expectedMsg, errs)
		}
		//check that we got the right message
		if len(tc.expectedMsg) != 0 && len(errs) == 1 {
			if !strings.Contains(errs[0].Error(), tc.expectedMsg) {
				t.Errorf("%s expected error to contain %s but it did not: %v", name, tc.expectedMsg, errs)
			}
		}
	}
}
