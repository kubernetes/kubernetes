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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
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

func TestProjectSELinuxOptions(t *testing.T) {
	baseContext := func() *api.SELinuxOptions {
		return &api.SELinuxOptions{
			User:  "user",
			Role:  "role",
			Type:  "type",
			Level: "level",
		}
	}

	var (
		userContext  = baseContext()
		roleContext  = baseContext()
		typeContext  = baseContext()
		levelContext = baseContext()
		allContext   = baseContext()
	)

	userContext.User = "user2"
	roleContext.Role = "role2"
	typeContext.Type = "type2"
	levelContext.Level = "level2"
	allContext.User = "user3"
	allContext.Role = "role3"
	allContext.Type = "type3"
	allContext.Level = "level3"

	cases := []struct {
		name     string
		source   *api.SELinuxOptions
		target   *api.SELinuxOptions
		expected *api.SELinuxOptions
	}{
		{
			name:     "project user",
			source:   userContext,
			target:   baseContext(),
			expected: userContext,
		},
		{
			name:     "project role",
			source:   roleContext,
			target:   baseContext(),
			expected: roleContext,
		},
		{
			name:     "project type",
			source:   typeContext,
			target:   baseContext(),
			expected: typeContext,
		},
		{
			name:     "project level",
			source:   levelContext,
			target:   baseContext(),
			expected: levelContext,
		},
		{
			name:     "project all",
			source:   allContext,
			target:   baseContext(),
			expected: allContext,
		},
	}

	for _, tc := range cases {
		result := ProjectSELinuxOptions(tc.source, tc.target)

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
