/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidation(t *testing.T) {
	testcases := map[string]struct {
		config       LoggingConfiguration
		path         *field.Path
		expectErrors string
	}{
		"okay": {
			config: LoggingConfiguration{
				Format:    "text",
				Verbosity: 10,
				VModule: VModuleConfiguration{
					{
						FilePattern: "gopher*",
						Verbosity:   100,
					},
				},
			},
		},
		"wrong-format": {
			config: LoggingConfiguration{
				Format: "no-such-format",
			},
			expectErrors: `format: Invalid value: "no-such-format": Unsupported log format`,
		},
		"embedded": {
			config: LoggingConfiguration{
				Format: "no-such-format",
			},
			path:         field.NewPath("config"),
			expectErrors: `config.format: Invalid value: "no-such-format": Unsupported log format`,
		},
		"verbosity-overflow": {
			config: LoggingConfiguration{
				Format:    "text",
				Verbosity: math.MaxInt32 + 1,
			},
			expectErrors: `verbosity: Invalid value: 0x80000000: Must be <= 2147483647`,
		},
		"vmodule-verbosity-overflow": {
			config: LoggingConfiguration{
				Format: "text",
				VModule: VModuleConfiguration{
					{
						FilePattern: "gopher*",
						Verbosity:   math.MaxInt32 + 1,
					},
				},
			},
			expectErrors: `vmodule[0]: Invalid value: 0x80000000: Must be <= 2147483647`,
		},
		"vmodule-empty-pattern": {
			config: LoggingConfiguration{
				Format: "text",
				VModule: VModuleConfiguration{
					{
						FilePattern: "",
						Verbosity:   1,
					},
				},
			},
			expectErrors: `vmodule[0]: Required value: File pattern must not be empty`,
		},
		"vmodule-pattern-with-special-characters": {
			config: LoggingConfiguration{
				Format: "text",
				VModule: VModuleConfiguration{
					{
						FilePattern: "foo,bar",
						Verbosity:   1,
					},
					{
						FilePattern: "foo=bar",
						Verbosity:   1,
					},
				},
			},
			expectErrors: `[vmodule[0]: Invalid value: "foo,bar": File pattern must not contain equal sign or comma, vmodule[1]: Invalid value: "foo=bar": File pattern must not contain equal sign or comma]`,
		},
		"vmodule-unsupported": {
			config: LoggingConfiguration{
				Format: "json",
				VModule: VModuleConfiguration{
					{
						FilePattern: "foo",
						Verbosity:   1,
					},
				},
			},
			expectErrors: `[format: Invalid value: "json": Unsupported log format, vmodule: Forbidden: Only supported for text log format]`,
		},
	}

	for name, test := range testcases {
		t.Run(name, func(t *testing.T) {
			err := test.config.Validate(nil, test.path)
			if len(err) == 0 {
				if test.expectErrors != "" {
					t.Fatalf("did not get expected error(s): %s", test.expectErrors)
				}
			} else {
				assert.Equal(t, test.expectErrors, err.ToAggregate().Error())
			}
		})
	}
}
