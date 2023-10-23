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

package plugin

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRegistrationHandler_ValidatePlugin(t *testing.T) {
	for _, test := range []struct {
		description string
		handler     func() *RegistrationHandler
		pluginName  string
		endpoint    string
		versions    []string
		shouldError bool
	}{
		{
			description: "no versions provided",
			handler:     NewRegistrationHandler,
			shouldError: true,
		},
		{
			description: "unsupported version",
			handler:     NewRegistrationHandler,
			versions:    []string{"v2.0.0"},
			shouldError: true,
		},
		{
			description: "plugin already registered with a higher supported version",
			handler: func() *RegistrationHandler {
				handler := NewRegistrationHandler()
				if err := handler.RegisterPlugin("this-plugin-already-exists-and-has-a-long-name-so-it-doesnt-collide", "", []string{"v1.1.0"}); err != nil {
					t.Fatal(err)
				}
				return handler
			},
			pluginName:  "this-plugin-already-exists-and-has-a-long-name-so-it-doesnt-collide",
			versions:    []string{"v1.0.0"},
			shouldError: true,
		},
		{
			description: "should validate the plugin",
			handler:     NewRegistrationHandler,
			pluginName:  "this-is-a-dummy-plugin-with-a-long-name-so-it-doesnt-collide",
			versions:    []string{"v1.3.0"},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			handler := test.handler()
			err := handler.ValidatePlugin(test.pluginName, test.endpoint, test.versions)
			if test.shouldError {
				assert.Error(t, err)
			} else {
				assert.Nil(t, err)
			}
		})
	}

	t.Cleanup(func() {
		handler := NewRegistrationHandler()
		handler.DeRegisterPlugin("this-plugin-already-exists-and-has-a-long-name-so-it-doesnt-collide")
		handler.DeRegisterPlugin("this-is-a-dummy-plugin-with-a-long-name-so-it-doesnt-collide")
	})
}
