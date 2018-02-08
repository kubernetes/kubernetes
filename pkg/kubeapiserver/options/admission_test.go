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
)

func TestValidate(t *testing.T) {
	// 1. Both `--admission-control` and `--enable-admission-plugins` are specified
	options := NewAdmissionOptions()
	options.PluginNames = []string{"ServiceAccount"}
	options.GenericAdmission.EnablePlugins = []string{"Initializers"}
	if len(options.Validate()) == 0 {
		t.Errorf("Expect error, but got none")
	}

	// 2. Both `--admission-control` and `--disable-admission-plugins` are specified
	options = NewAdmissionOptions()
	options.PluginNames = []string{"ServiceAccount"}
	options.GenericAdmission.DisablePlugins = []string{"Initializers"}
	if len(options.Validate()) == 0 {
		t.Errorf("Expect error, but got none")
	}

	// 3. PluginNames is not registered
	options = NewAdmissionOptions()
	options.PluginNames = []string{"pluginA"}
	if len(options.Validate()) == 0 {
		t.Errorf("Expect error, but got none")
	}

	// 4. PluginNames is not valid
	options = NewAdmissionOptions()
	options.PluginNames = []string{"ServiceAccount"}
	if errs := options.Validate(); len(errs) > 0 {
		t.Errorf("Unexpected err: %v", errs)
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
