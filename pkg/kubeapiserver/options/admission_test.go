/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	genericoptions "k8s.io/apiserver/pkg/server/options"
)

func TestValidate(t *testing.T) {
	tests := []struct {
		name             string
		pluginNames      []string
		genericAdmission *genericoptions.AdmissionOptions
		wantError        bool
	}{
		{
			name:        "Both `--admission-control` and `--enable-admission-plugins` are specified",
			pluginNames: []string{"ServiceAccount"},
			genericAdmission: &genericoptions.AdmissionOptions{
				Plugins:       admission.NewPlugins(),
				EnablePlugins: []string{"Initializers"},
			},
			wantError: true,
		},
		{
			name:        "Both `--admission-control` and `--disable-admission-plugins` are specified",
			pluginNames: []string{"ServiceAccount"},
			genericAdmission: &genericoptions.AdmissionOptions{
				Plugins:        admission.NewPlugins(),
				DisablePlugins: []string{"Initializers"},
			},
			wantError: true,
		},
		{
			name:        "PluginNames is not registered",
			pluginNames: []string{"pluginA"},
			wantError:   true,
		},
		{
			name:        "PluginNames is not valid",
			pluginNames: []string{"ServiceAccount"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options := NewAdmissionOptions()
			if tt.genericAdmission != nil {
				options.GenericAdmission = tt.genericAdmission
			}
			options.PluginNames = tt.pluginNames
			if tt.wantError {
				if len(options.Validate()) == 0 {
					t.Errorf("Expect error, but got none")
				}
			} else {
				if errs := options.Validate(); len(errs) > 0 {
					t.Errorf("Unexpected err: %v", errs)
				}
			}
		})
	}
}

func TestComputeEnabledAdmission(t *testing.T) {
	tests := []struct {
		name             string
		all              []string
		enabled          []string
		expectedDisabled []string
	}{
		{
			name:             "matches",
			all:              []string{"one", "two"},
			enabled:          []string{"one", "two"},
			expectedDisabled: []string{},
		},
		{
			name:             "choose one",
			all:              []string{"one", "two"},
			enabled:          []string{"one"},
			expectedDisabled: []string{"two"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actualEnabled, actualDisabled := computePluginNames(tc.enabled, tc.all)
			if e, a := tc.enabled, actualEnabled; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, got %v", e, a)
			}
			if e, a := tc.expectedDisabled, actualDisabled; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, got %v", e, a)
			}
		})
	}
}
