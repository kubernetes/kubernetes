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

package logs

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/component-base/config"
)

func TestValidateLoggingConfiguration(t *testing.T) {
	testcases := map[string]struct {
		config       config.LoggingConfiguration
		expectErrors string
	}{
		"okay": {
			config: config.LoggingConfiguration{
				Format:    "text",
				Verbosity: 10,
				VModule: config.VModuleConfiguration{
					{
						FilePattern: "gopher*",
						Verbosity:   100,
					},
				},
			},
		},
		"wrong-format": {
			config: config.LoggingConfiguration{
				Format: "no-such-format",
			},
			expectErrors: `format: Invalid value: "no-such-format": Unsupported log format`,
		},
		"verbosity-overflow": {
			config: config.LoggingConfiguration{
				Format:    "text",
				Verbosity: math.MaxInt32 + 1,
			},
			expectErrors: `verbosity: Invalid value: 0x80000000: Must be <= 2147483647`,
		},
		"vmodule-verbosity-overflow": {
			config: config.LoggingConfiguration{
				Format: "text",
				VModule: config.VModuleConfiguration{
					{
						FilePattern: "gopher*",
						Verbosity:   math.MaxInt32 + 1,
					},
				},
			},
			expectErrors: `vmodule[0]: Invalid value: 0x80000000: Must be <= 2147483647`,
		},
		"vmodule-empty-pattern": {
			config: config.LoggingConfiguration{
				Format: "text",
				VModule: config.VModuleConfiguration{
					{
						FilePattern: "",
						Verbosity:   1,
					},
				},
			},
			expectErrors: `vmodule[0]: Required value: File pattern must not be empty`,
		},
		"vmodule-pattern-with-special-characters": {
			config: config.LoggingConfiguration{
				Format: "text",
				VModule: config.VModuleConfiguration{
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
			config: config.LoggingConfiguration{
				Format: "json",
				VModule: config.VModuleConfiguration{
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
			errs := ValidateLoggingConfiguration(&test.config, nil)
			if len(errs) == 0 {
				if test.expectErrors != "" {
					t.Fatalf("did not get expected error(s): %s", test.expectErrors)
				}
			} else {
				assert.Equal(t, test.expectErrors, errs.ToAggregate().Error())
			}
		})
	}
}
