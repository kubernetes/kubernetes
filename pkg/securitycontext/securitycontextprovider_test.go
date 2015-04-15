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

package securitycontext

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestFilterCapabilities(t *testing.T) {
	testCases := []struct {
		name      string
		allowed   []api.CapabilityType
		requested []api.CapabilityType
		expected  []api.CapabilityType
	}{
		{
			"Empty allowed",
			[]api.CapabilityType{},
			[]api.CapabilityType{},
			[]api.CapabilityType{},
		},
		{
			"Remove disallowed",
			[]api.CapabilityType{"a"},
			[]api.CapabilityType{"a", "b"},
			[]api.CapabilityType{"a"},
		},
		{
			"Filter multiple",
			[]api.CapabilityType{"a", "c"},
			[]api.CapabilityType{"a", "b", "c", "d"},
			[]api.CapabilityType{"a", "c"},
		},
	}

	scp := DefaultSecurityContextProvider{}

	for _, tc := range testCases {
		actual := scp.filterCapabilities(tc.requested, tc.allowed)
		if !reflect.DeepEqual(tc.expected, actual) {
			t.Errorf("Failed to filter caps correctly for tc: %s.  Expected: %v but got %v", tc.name, tc.expected, actual)
		}
	}
}

func TestApplySecurityOption(t *testing.T) {
	testCases := []struct {
		name     string
		config   []string
		optName  string
		optVal   string
		expected []string
	}{
		{
			"Empty name",
			[]string{"a", "b"},
			"",
			"valA",
			[]string{"a", "b"},
		},
		{
			"Empty val",
			[]string{"a", "b"},
			"optA",
			"",
			[]string{"a", "b"},
		},
		{
			"Valid",
			[]string{"a", "b"},
			"c",
			"d",
			[]string{"a", "b", "c:d"},
		},
	}

	for _, tc := range testCases {
		actual := modifySecurityOption(tc.config, tc.optName, tc.optVal)
		if !reflect.DeepEqual(tc.expected, actual) {
			t.Errorf("Failed to apply options correctly for tc: %S.  Expected: %v but got %v", tc.name, tc.expected, actual)
		}
	}
}

func TestApplyPrivileged(t *testing.T) {
	testCases := []struct {
		name              string
		constraintSetting bool
		containerSetting  bool
		expectedResult    bool
	}{
		{"Constraint allowed, pod requested", true, true, true},
		{"Constraint allowed, pod not requested", true, false, false},
		{"Constraint disallowed, pod requested", false, true, false},
		{"Constraint disallowed, pod not requested", false, false, false},
	}

	scp := DefaultSecurityContextProvider{&api.SecurityConstraints{}}

	for _, tc := range testCases {
		scp.SecurityConstraints.AllowPrivileged = tc.constraintSetting
		container := &api.Container{
			SecurityContext: &api.SecurityContext{
				Privileged: tc.containerSetting,
			},
		}
		scp.applyPrivileged(container)

		if container.SecurityContext.Privileged != tc.expectedResult {
			t.Errorf("Failed to set privileged correctly for tc: %s.  Expected %s, got %s", tc.name, tc.expectedResult, container.SecurityContext.Privileged)
		}
	}
}

func TestApplyCapRequests(t *testing.T) {
	testCases := []struct {
		name                     string
		securityContraints       *api.SecurityConstraints
		containerSecurityContext *api.SecurityContext
		expectedSecurityContext  *api.SecurityContext
	}{
		{
			//not testing nil, the entry method (ApplySecurityContext) ensures that if the container has
			//a nil context then the default one is added
			name: "context that doesn't allow, no default caps, no container requests",
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: false,
			},
			containerSecurityContext: &api.SecurityContext{},
			expectedSecurityContext:  &api.SecurityContext{},
		},
		{
			name: "context that doesn't allow, no default caps, container requests",
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: false,
			},
			containerSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
			expectedSecurityContext: &api.SecurityContext{Capabilities: &api.Capabilities{}},
		},
		{
			name: "context that doesn't allow, default caps, no container requests",
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: false,
				DefaultSecurityContext: &api.SecurityContext{
					Capabilities: &api.Capabilities{
						Add:  []api.CapabilityType{"a"},
						Drop: []api.CapabilityType{"b"},
					},
				},
			},
			containerSecurityContext: &api.SecurityContext{},
			expectedSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"a"},
					Drop: []api.CapabilityType{"b"},
				},
			},
		},
		{
			name: "context that doesn't allow, default caps, container requests",
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: false,
				DefaultSecurityContext: &api.SecurityContext{
					Capabilities: &api.Capabilities{
						Add:  []api.CapabilityType{"a"},
						Drop: []api.CapabilityType{"b"},
					},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
			expectedSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"a"},
					Drop: []api.CapabilityType{"b"},
				},
			},
		},
		{
			name: "filter with no whitelists", //everything should be allowed
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: true,
			},
			containerSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
			expectedSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
		},
		{
			name: "filter with empty whitelists", //everything should be filtered out since nothing is allowed
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: true,
				Capabilities:      &api.Capabilities{},
			},
			containerSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
			expectedSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{},
					Drop: []api.CapabilityType{},
				},
			},
		},
		{
			name: "filter add whitelists",
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: true,
				Capabilities: &api.Capabilities{
					Add: []api.CapabilityType{"foo"},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo", "anotherFoo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
			expectedSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{},
				},
			},
		},
		{
			name: "filter drop whitelists",
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: true,
				Capabilities: &api.Capabilities{
					Drop: []api.CapabilityType{"bar"},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar", "anotherBar"},
				},
			},
			expectedSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{},
					Drop: []api.CapabilityType{"bar"},
				},
			},
		},
		{
			name: "filter both whitelists",
			securityContraints: &api.SecurityConstraints{
				AllowCapabilities: true,
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo", "anotherFoo"},
					Drop: []api.CapabilityType{"bar", "anotherBar"},
				},
			},
			expectedSecurityContext: &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add:  []api.CapabilityType{"foo"},
					Drop: []api.CapabilityType{"bar"},
				},
			},
		},
	}

	scp := DefaultSecurityContextProvider{}
	container := api.Container{}
	for _, tc := range testCases {
		scp.SecurityConstraints = tc.securityContraints
		container.SecurityContext = tc.containerSecurityContext
		scp.applyCapRequests(&container)

		if !reflect.DeepEqual(tc.expectedSecurityContext.Capabilities, container.SecurityContext.Capabilities) {
			t.Errorf("Unexpected capabilities result for tc: %s.  Expected %+v, got %+v", tc.name, tc.expectedSecurityContext.Capabilities, container.SecurityContext.Capabilities)
		}
	}
}

func TestApplySELinux(t *testing.T) {
	testCases := []struct {
		name                     string
		securityContraints       *api.SecurityConstraints
		containerSecurityContext *api.SecurityContext
		expectedSecurityContext  *api.SecurityContext
	}{
		{
			name: "context that doesn't allow, no default, no container requests",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, false, false, false, false},
			},
			containerSecurityContext: &api.SecurityContext{},
			expectedSecurityContext:  &api.SecurityContext{},
		},
		{
			//everything should be removed
			name: "context that doesn't allow, no default, container requests",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, false, false, false, false},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"", "", "", "", false},
			},
		},
		{
			name: "context doesn't allow, default, no container requests",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, false, false, false, false},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
			},
		},
		{
			name: "context doesn't allow, default, container requests",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, false, false, false, false},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"a", "b", "c", "d", true},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
			},
		},
		{
			name: "context allows, container requests disable",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, false, false, false, true},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"", "", "", "", true},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", true},
			},
		},
		{
			name: "context allows, container requests level",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, false, false, true, false},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"", "", "", "test", false},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "test", false},
			},
		},
		{
			name: "context allows, container requests role",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, true, false, false, false},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"", "test", "", "", false},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"user", "test", "type", "level", false},
			},
		},
		{
			name: "context allows, container requests type",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{false, false, true, false, false},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"", "", "test", "", false},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"user", "role", "test", "level", false},
			},
		},
		{
			name: "context allows, container requests user",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{true, false, false, false, false},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"test", "", "", "", false},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"test", "role", "type", "level", false},
			},
		},
		{
			name: "context allows, full override",
			securityContraints: &api.SecurityConstraints{
				SELinux: &api.SELinuxSecurityConstraints{true, true, true, true, true},
				DefaultSecurityContext: &api.SecurityContext{
					SELinuxOptions: &api.SELinuxOptions{"user", "role", "type", "level", false},
				},
			},
			containerSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"test", "test", "test", "test", true},
			},
			expectedSecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{"test", "test", "test", "test", true},
			},
		},
	}

	scp := DefaultSecurityContextProvider{}
	container := api.Container{}
	for _, tc := range testCases {
		scp.SecurityConstraints = tc.securityContraints
		container.SecurityContext = tc.containerSecurityContext
		scp.applySELinux(&container)

		if !reflect.DeepEqual(tc.expectedSecurityContext.SELinuxOptions, container.SecurityContext.SELinuxOptions) {
			t.Errorf("Unexpected SELinux result for tc: %s.  Expected %+v, got %+v", tc.name, tc.expectedSecurityContext.SELinuxOptions, container.SecurityContext.SELinuxOptions)
		}
	}
}
