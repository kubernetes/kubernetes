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

package metrics

import "testing"

func TestValidateAllowMetricLabel(t *testing.T) {
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
			err := validateAllowMetricLabel(tt.input)
			if err == nil && tt.expectedError {
				t.Error("Got error is nil, wanted error is not nil")
			}
			if err != nil && !tt.expectedError {
				t.Errorf("Got error is %v, wanted no error", err)
			}
		})
	}
}
