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

package user

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestNewMustRunAs(t *testing.T) {
	tests := map[string]struct {
		opts *extensions.RunAsUserStrategyOptions
		pass bool
	}{
		"nil opts": {
			opts: nil,
			pass: false,
		},
		"invalid opts": {
			opts: &extensions.RunAsUserStrategyOptions{},
			pass: false,
		},
		"valid opts": {
			opts: &extensions.RunAsUserStrategyOptions{
				Ranges: []extensions.UserIDRange{
					{Min: 1, Max: 1},
				},
			},
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

func TestGenerate(t *testing.T) {
	opts := &extensions.RunAsUserStrategyOptions{
		Ranges: []extensions.UserIDRange{
			{Min: 1, Max: 1},
		},
	}
	mustRunAs, err := NewMustRunAs(opts)
	if err != nil {
		t.Fatalf("unexpected error initializing NewMustRunAs %v", err)
	}
	generated, err := mustRunAs.Generate(nil, nil)
	if err != nil {
		t.Fatalf("unexpected error generating runAsUser %v", err)
	}
	if *generated != opts.Ranges[0].Min {
		t.Errorf("generated runAsUser does not equal configured runAsUser")
	}
}

func TestValidate(t *testing.T) {
	opts := &extensions.RunAsUserStrategyOptions{
		Ranges: []extensions.UserIDRange{
			{Min: 1, Max: 1},
			{Min: 10, Max: 20},
		},
	}

	validID := int64(15)
	invalidID := int64(21)

	tests := map[string]struct {
		container   *api.Container
		expectedMsg string
	}{
		"good container": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &validID,
				},
			},
		},
		"nil security context": {
			container: &api.Container{
				SecurityContext: nil,
			},
			expectedMsg: "unable to validate nil security context for container",
		},
		"nil run as user": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: nil,
				},
			},
			expectedMsg: "unable to validate nil RunAsUser for container",
		},
		"invalid id": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &invalidID,
				},
			},
			expectedMsg: "does not match required range",
		},
	}

	for name, tc := range tests {
		mustRunAs, err := NewMustRunAs(opts)
		if err != nil {
			t.Errorf("unexpected error initializing NewMustRunAs for testcase %s: %#v", name, err)
			continue
		}
		errs := mustRunAs.Validate(nil, tc.container)
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
