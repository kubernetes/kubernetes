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

package config

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestValidate(t *testing.T) {
	for _, testCase := range []struct {
		config   *Config
		hasError bool
	}{
		{
			config: &Config{Federations: map[string]string{}},
		},
		{
			config: &Config{
				Federations: map[string]string{
					"abc": "d.e.f",
				},
			},
		},
		{
			config: &Config{
				Federations: map[string]string{
					"a.b": "cdef",
				},
			},
			hasError: true,
		},
	} {
		err := testCase.config.Validate()
		if !testCase.hasError {
			assert.Nil(t, err, "should be valid", testCase)
		} else {
			assert.NotNil(t, err, "should not be valid", testCase)
		}
	}
}
