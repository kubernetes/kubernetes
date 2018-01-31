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

package podtolerationrestriction

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"

	api "k8s.io/kubernetes/pkg/apis/core"
)

const apiGroup = "podtolerationrestriction.admission.k8s.io"

var supportedConfigAPIVersions []string = []string{"v1alpha1", "v1"}

// TestConfigParse tests config parsing for versioned configuration data
func TestConfigParse(t *testing.T) {
	for _, v := range supportedConfigAPIVersions {
		testName := fmt.Sprintf("Test config parse for api version: %s", v)
		version := apiGroup + "/" + v

		t.Run(testName, func(t *testing.T) {
			content := `apiVersion: ` + version + `
kind: Configuration
default:
- key: key1
  operator: Equal
  value: value1
whitelist:
- key: key2
  operator: Equal
  value: value2`

			parsed, err := loadConfiguration(bytes.NewBufferString(content))
			if err != nil {
				t.Error("encountered unexpected error when loading configuration", err)
			}

			expectedDefault := []api.Toleration{{Key: "key1", Operator: "Equal", Value: "value1"}}
			expectedWhitelist := []api.Toleration{{Key: "key2", Operator: "Equal", Value: "value2"}}

			if !reflect.DeepEqual(parsed.Default, expectedDefault) {
				t.Errorf("failed to parse expected default tolerations. expected: %v. got: %v.", expectedDefault, parsed.Default)
			}

			if !reflect.DeepEqual(parsed.Whitelist, expectedWhitelist) {
				t.Errorf("failed to parse expected tolerations whitelist. expected: %v. got: %v.", expectedWhitelist, parsed.Whitelist)
			}
		})
	}
}
