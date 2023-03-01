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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	cmconfig "k8s.io/controller-manager/config"
)

func TestValidateControllers(t *testing.T) {
	allControllers := []string{"defaulton1", "defaulton2", "defaultoff1", "defaultoff2"}
	disabledByDefaultControllers := []string{"defaultoff1", "defaultoff2"}

	grid := []struct {
		Name           string
		Controllers    []string
		ExpectedErrors []string
	}{
		{
			Name:           "exclude-known-controller",
			Controllers:    []string{"-defaulton1"},
			ExpectedErrors: nil,
		},
		{
			Name:           "exclude-unknown-controller",
			Controllers:    []string{"-notacontroller"},
			ExpectedErrors: nil,
		},
		{
			Name:        "include-unknown-controller",
			Controllers: []string{"notacontroller"},
			ExpectedErrors: []string{
				`"notacontroller" is not in the list of known controllers`,
			},
		},
	}

	for _, g := range grid {
		t.Run(g.Name, func(t *testing.T) {
			options := GenericControllerManagerConfigurationOptions{
				GenericControllerManagerConfiguration: &cmconfig.GenericControllerManagerConfiguration{
					Controllers: g.Controllers,
				},
			}

			errors := options.Validate(allControllers, disabledByDefaultControllers)
			var errorStrings []string
			for _, err := range errors {
				errorStrings = append(errorStrings, err.Error())
			}
			got := strings.Join(errorStrings, "\n")
			want := strings.Join(g.ExpectedErrors, "\n")
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("unexpected output from Validate (-want, +got): %v", diff)
			}
		})
	}
}
