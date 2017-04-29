/*
Copyright 2017 The Kubernetes Authors.

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

package plugins

import (
	"strings"
	"testing"
)

func TestPlugin(t *testing.T) {
	tests := []struct {
		plugin        Plugin
		expectedErr   string
		expectedValid bool
	}{
		{
			plugin: Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
				},
			},
			expectedValid: true,
		},
		{
			plugin: Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
				},
			},
			expectedErr: "incomplete",
		},
		{
			plugin:      Plugin{},
			expectedErr: "incomplete",
		},
	}

	for _, test := range tests {
		if is := test.plugin.IsValid(); test.expectedValid != is {
			t.Errorf("%s: expected valid=%v, got %v", test.plugin.Name, test.expectedValid, is)
		}
		err := test.plugin.Validate()
		if len(test.expectedErr) > 0 {
			if err == nil {
				t.Errorf("%s: expected error, got none", test.plugin.Name)
			} else if !strings.Contains(err.Error(), test.expectedErr) {
				t.Errorf("%s: expected error containing %q, got %v", test.plugin.Name, test.expectedErr, err)
			}
		} else if err != nil {
			t.Errorf("%s: expected no error, got %v", test.plugin.Name, err)
		}
	}
}
