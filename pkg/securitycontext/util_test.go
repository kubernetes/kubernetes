/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package securitycontext

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestParseSELinuxOptions(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected *api.SELinuxOptions
	}{
		{
			name:  "simple",
			input: "user_t:role_t:type_t:s0",
			expected: &api.SELinuxOptions{
				User:  "user_t",
				Role:  "role_t",
				Type:  "type_t",
				Level: "s0",
			},
		},
		{
			name:  "simple + categories",
			input: "user_t:role_t:type_t:s0:c0",
			expected: &api.SELinuxOptions{
				User:  "user_t",
				Role:  "role_t",
				Type:  "type_t",
				Level: "s0:c0",
			},
		},
		{
			name:  "not enough fields",
			input: "type_t:s0:c0",
		},
	}

	for _, tc := range cases {
		result, err := ParseSELinuxOptions(tc.input)

		if err != nil {
			if tc.expected == nil {
				continue
			} else {
				t.Errorf("%v: unexpected error: %v", tc.name, err)
			}
		}

		compareContexts(tc.name, tc.expected, result, t)
	}
}

func compareContexts(name string, ex, ac *api.SELinuxOptions, t *testing.T) {
	if e, a := ex.User, ac.User; e != a {
		t.Errorf("%v: expected user: %v, got: %v", name, e, a)
	}
	if e, a := ex.Role, ac.Role; e != a {
		t.Errorf("%v: expected role: %v, got: %v", name, e, a)
	}
	if e, a := ex.Type, ac.Type; e != a {
		t.Errorf("%v: expected type: %v, got: %v", name, e, a)
	}
	if e, a := ex.Level, ac.Level; e != a {
		t.Errorf("%v: expected level: %v, got: %v", name, e, a)
	}
}

func TestHaRootUID(t *testing.T) {
	var nonRoot int64 = 1
	var root int64 = 0

	tests := map[string]struct {
		container *api.Container
		expect    bool
	}{
		"nil sc": {
			container: &api.Container{SecurityContext: nil},
		},
		"nil runAsuser": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: nil,
				},
			},
		},
		"runAsUser non-root": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &nonRoot,
				},
			},
		},
		"runAsUser root": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &root,
				},
			},
			expect: true,
		},
	}

	for k, v := range tests {
		actual := HasRootUID(v.container)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}

func TestHasRunAsUser(t *testing.T) {
	var runAsUser int64 = 0

	tests := map[string]struct {
		container *api.Container
		expect    bool
	}{
		"nil sc": {
			container: &api.Container{SecurityContext: nil},
		},
		"nil runAsUser": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: nil,
				},
			},
		},
		"valid runAsUser": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &runAsUser,
				},
			},
			expect: true,
		},
	}

	for k, v := range tests {
		actual := HasRunAsUser(v.container)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}

func TestHasRootRunAsUser(t *testing.T) {
	var nonRoot int64 = 1
	var root int64 = 0

	tests := map[string]struct {
		container *api.Container
		expect    bool
	}{
		"nil sc": {
			container: &api.Container{SecurityContext: nil},
		},
		"nil runAsuser": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: nil,
				},
			},
		},
		"runAsUser non-root": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &nonRoot,
				},
			},
		},
		"runAsUser root": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &root,
				},
			},
			expect: true,
		},
	}

	for k, v := range tests {
		actual := HasRootRunAsUser(v.container)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}
