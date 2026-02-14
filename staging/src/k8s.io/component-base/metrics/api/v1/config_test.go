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
	"testing"

	"github.com/blang/semver/v4"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateShowHiddenMetricsVersion(t *testing.T) {
	currentVersion := semver.MustParse("1.17.0")

	var tests = []struct {
		desc          string
		targetVersion string
		expectedError bool
	}{
		{
			desc:          "invalid version is not allowed",
			targetVersion: "1.invalid",
			expectedError: true,
		},
		{
			desc:          "patch version is not allowed",
			targetVersion: "1.16.0",
			expectedError: true,
		},
		{
			desc:          "old version is not allowed",
			targetVersion: "1.15",
			expectedError: true,
		},
		{
			desc:          "new version is not allowed",
			targetVersion: "1.17",
			expectedError: true,
		},
		{
			desc:          "valid version is allowed",
			targetVersion: "1.16",
			expectedError: false,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.desc, func(t *testing.T) {
			errs := validateShowHiddenMetricsVersion(currentVersion, tc.targetVersion, field.NewPath("showHiddenMetricsForVersion"))

			if tc.expectedError {
				assert.Errorf(t, errs.ToAggregate(), "Failed to test: %s", tc.desc)
			} else {
				assert.NoErrorf(t, errs.ToAggregate(), "Failed to test: %s", tc.desc)
			}
		})
	}
}

func TestValidateDisabledMetrics(t *testing.T) {
	var tests = []struct {
		name          string
		input         []string
		expectedError bool
	}{
		{
			"validated",
			[]string{"metric_name", "another_metric"},
			false,
		},
		{
			"empty input",
			[]string{},
			false,
		},
		{
			name:          "empty metric name",
			input:         []string{"", "another_metric"},
			expectedError: true,
		},
		{
			"invalid metric name",
			[]string{"metric_.name", "another_metric"},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := validateDisabledMetrics(tt.input, field.NewPath("disabledMetrics"))
			if len(errs) == 0 && tt.expectedError {
				t.Error("Got no error, wanted error(s)")
			}
			if len(errs) != 0 && !tt.expectedError {
				t.Errorf("Got error(s): %v, wanted no error", errs.ToAggregate().Error())
			}
		})
	}
}

func TestValidateAllowListMapping(t *testing.T) {
	var tests = []struct {
		name          string
		input         map[string]string
		expectedError bool
	}{
		{
			"validated",
			map[string]string{
				"metric_name,label_name": "labelValue1,labelValue2",
			},
			false,
		},
		{
			"metric name is not valid",
			map[string]string{
				"-metric_name,label_name": "labelValue1,labelValue2",
			},
			true,
		},
		{
			"label name is not valid",
			map[string]string{
				"metric_name,:label_name": "labelValue1,labelValue2",
			},
			true,
		},
		{
			"no label name",
			map[string]string{
				"metric_name": "labelValue1,labelValue2",
			},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := validateAllowListMapping(tt.input, field.NewPath("allowListMapping"))
			if len(errs) == 0 && tt.expectedError {
				t.Error("Got no error, wanted error(s)")
			}
			if len(errs) != 0 && !tt.expectedError {
				t.Errorf("Got error: %v, wanted no error(s)", errs.ToAggregate().Error())
			}
		})
	}
}
