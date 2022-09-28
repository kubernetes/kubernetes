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
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/featuregate"
)

func TestValidation(t *testing.T) {
	jsonOptionsEnabled := LoggingConfiguration{
		Format: "text",
		Options: FormatOptions{
			JSON: JSONOptions{
				SplitStream: true,
				InfoBufferSize: resource.QuantityValue{
					Quantity: *resource.NewQuantity(1024, resource.DecimalSI),
				},
			},
		},
	}
	jsonErrors := `[options.json.splitStream: Forbidden: Feature LoggingAlphaOptions is disabled, options.json.infoBufferSize: Forbidden: Feature LoggingAlphaOptions is disabled]`
	testcases := map[string]struct {
		config       LoggingConfiguration
		path         *field.Path
		featureGate  featuregate.FeatureGate
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
		"JSON used, default gates": {
			config:       jsonOptionsEnabled,
			featureGate:  defaultFeatureGate,
			expectErrors: jsonErrors,
		},
		"JSON used, disabled gates": {
			config:       jsonOptionsEnabled,
			featureGate:  disabledFeatureGate,
			expectErrors: jsonErrors,
		},
		"JSON used, enabled gates": {
			config:      jsonOptionsEnabled,
			featureGate: enabledFeatureGate,
		},
	}

	for name, test := range testcases {
		t.Run(name, func(t *testing.T) {
			featureGate := test.featureGate
			if featureGate == nil {
				featureGate = defaultFeatureGate
			}
			err := Validate(&test.config, featureGate, test.path)
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
