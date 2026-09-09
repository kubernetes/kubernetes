/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/component-base/config"
	cmconfig "k8s.io/controller-manager/config"
)

func TestValidateGenericControllerManagerConfigurationOptions(t *testing.T) {
	testCases := []struct {
		name                   string
		allControllers         []string
		controllerAliases      map[string]string
		options                *GenericControllerManagerConfigurationOptions
		expectErrors           bool
		expectedErrorSubString string
	}{
		{
			name:              "no controllers defined",
			allControllers:    nil,
			controllerAliases: nil,
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"*",
				},
			}),
		},
		{
			name:              "recognizes empty controllers",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options:           NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{}),
		},
		{
			name:              "recognizes controllers without any aliases",
			allControllers:    getAllControllers(),
			controllerAliases: nil,
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"blue-controller",
				},
			}),
		},
		{
			name:              "recognizes valid controllers",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"*",
					"-red-controller",
					"blue-controller",
				},
			}),
		},
		{
			name:              "recognizes disabled controller",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"green-controller",
				},
			}),
		},
		{
			name:              "recognized aliased controller",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"ultramarine-controller",
					"-pink-controller",
				},
			}),
		},
		{
			name:              "does not recognize controller",
			allControllers:    nil,
			controllerAliases: nil,
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"red-controller",
				},
			}),
			expectErrors:           true,
			expectedErrorSubString: "\"red-controller\" is not in the list of known controllers",
		},
		{
			name:              "does not recognize controller with aliases",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"crimson-controller",
					"grey-controller",
				},
			}),
			expectErrors:           true,
			expectedErrorSubString: "\"grey-controller\" is not in the list of known controllers",
		},
		{
			name:              "leader election accepts only leases",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				LeaderElection: config.LeaderElectionConfiguration{
					LeaderElect:  true,
					ResourceLock: "configmapsleases",
				},
			}),
			expectErrors:           true,
			expectedErrorSubString: "resourceLock value must be \"leases\"",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := tc.options.Validate(tc.allControllers, []string{"green-controller"}, tc.controllerAliases)
			if len(errs) > 0 && !tc.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && tc.expectErrors {
				t.Errorf("expected errors, no errors found")
			}

			if len(errs) > 0 && tc.expectErrors {
				gotErr := utilerrors.NewAggregate(errs).Error()
				if !strings.Contains(gotErr, tc.expectedErrorSubString) {
					t.Errorf("expected error: %s, got err: %v", tc.expectedErrorSubString, gotErr)
				}
			}
		})
	}
}

func TestApplyToGenericControllerManagerConfigurationOptions(t *testing.T) {
	testCases := []struct {
		name                string
		allControllers      []string
		controllerAliases   map[string]string
		options             *GenericControllerManagerConfigurationOptions
		expectedControllers []string
	}{
		{
			name:              "no controllers defined",
			allControllers:    nil,
			controllerAliases: nil,
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"*",
				},
			}),
			expectedControllers: []string{
				"*",
			},
		},
		{
			name:              "empty aliases",
			allControllers:    getAllControllers(),
			controllerAliases: nil,
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"-blue-controller",
				},
			}),
			expectedControllers: []string{
				"-blue-controller",
			},
		},
		{
			name:              "applies valid controllers",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"*",
					"green-controller",
					"-red-controller",
					"blue-controller",
				},
			}),
			expectedControllers: []string{
				"*",
				"green-controller",
				"-red-controller",
				"blue-controller",
			},
		},
		{
			name:              "resolves aliases",
			allControllers:    getAllControllers(),
			controllerAliases: getControllerAliases(),
			options: NewGenericControllerManagerConfigurationOptions(&cmconfig.GenericControllerManagerConfiguration{
				Controllers: []string{
					"green-controller",
					"-crimson-controller",
					"ultramarine-controller",
					"-pink-controller",
				},
			}),
			expectedControllers: []string{
				"green-controller",
				"-red-controller",
				"blue-controller",
				"-red-controller",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &cmconfig.GenericControllerManagerConfiguration{}
			err := tc.options.ApplyTo(cfg, tc.allControllers, []string{"green-controller"}, tc.controllerAliases)
			if err != nil {
				t.Errorf("expected no errors, error found: %v", err)
			}
			if !reflect.DeepEqual(cfg.Controllers, tc.expectedControllers) {
				t.Errorf("applyTo failed, expected controllers %q, got controllers %q", strings.Join(cfg.Controllers, ","), strings.Join(tc.expectedControllers, ","))
			}
		})
	}
}

func getAllControllers() []string {
	return []string{
		"red-controller",
		"green-controller",
		"blue-controller",
	}
}

func getControllerAliases() map[string]string {
	return map[string]string{
		"crimson-controller":     "red-controller",
		"pink-controller":        "red-controller",
		"ultramarine-controller": "blue-controller",
	}
}
