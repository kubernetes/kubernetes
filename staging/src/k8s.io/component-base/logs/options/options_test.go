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

package options

import (
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/config/options"
	"testing"
)

func TestValidate(t *testing.T) {
	logsOptions := NewOptions()
	testcases := []struct {
		name   string
		value  string
		expect field.ErrorList
	}{{
		name:  "Default log format",
		value: options.DefaultLogFormat,
	}, {
		name:  "JSON log format",
		value: options.JSONLogFormat,
	}, {
		name:  "Unsupported log format",
		value: "test",
		expect: field.ErrorList{&field.Error{
			Type:     "FieldValueInvalid",
			Field:    "loggingConfiguration.Format",
			BadValue: "test",
			Detail:   "Unsupported log format",
		}},
	},
	}

	for _, tc := range testcases {
		logsOptions.Config.Format = tc.value
		res := logsOptions.Validate()
		if !assert.ElementsMatch(t, tc.expect, res) {
			t.Errorf("Wrong set of log format for %q. expect %v, got %v", tc.name, tc.expect, res)
		}
	}

}
