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
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
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
			Level: "level",
			Type:  "type",
		}
	}

	role := newValidOpts()
	role.Role = "invalid"

	user := newValidOpts()
	user.User = "invalid"

	level := newValidOpts()
	level.Level = "invalid"

	seType := newValidOpts()
	seType.Type = "invalid"

	tests := map[string]struct {
		seLinux     *api.SELinuxOptions
		expectedMsg string
	}{
		"invalid role": {
			seLinux:     role,
			expectedMsg: "does not match required role",
		},
		"invalid user": {
			seLinux:     user,
			expectedMsg: "does not match required user",
		},
		"invalid level": {
			seLinux:     level,
			expectedMsg: "does not match required level",
		},
		"invalid type": {
			seLinux:     seType,
			expectedMsg: "does not match required type",
		},
		"valid": {
			seLinux:     newValidOpts(),
			expectedMsg: "",
		},
	}

	opts := &extensions.SELinuxStrategyOptions{
		SELinuxOptions: newValidOpts(),
	}

	for name, tc := range tests {
		mustRunAs, err := NewMustRunAs(opts)
		if err != nil {
			t.Errorf("unexpected error initializing NewMustRunAs for testcase %s: %#v", name, err)
			continue
		}
		container := &api.Container{
			SecurityContext: &api.SecurityContext{
				SELinuxOptions: tc.seLinux,
			},
		}

		errs := mustRunAs.Validate(nil, container)
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
