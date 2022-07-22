/*
Copyright 2022 The Kubernetes Authors.

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

package webhook

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	"testing"
)

func TestValidateWebhookURL(t *testing.T) {
	testCases := []struct {
		title          string
		f              *field.Path
		URL            string
		forceHttps     bool
		hasError       bool
		expectError    string
		expectErrorNum int
	}{
		{
			title:      "URL without error",
			f:          nil,
			URL:        "https://127.0.0.1:8708",
			forceHttps: true,
			hasError:   false,
		},
		{
			title:          "URL with wrong pattern",
			f:              nil,
			URL:            "wrong url pattern",
			forceHttps:     true,
			hasError:       true,
			expectError:    "'https' is the only allowed URL scheme; desired format: https://host[/path]",
			expectErrorNum: 0,
		},
		{
			title:          "URL with fragments",
			f:              nil,
			URL:            "https://127.0.0.1:8708#seree",
			forceHttps:     true,
			hasError:       true,
			expectError:    "fragments are not permitted in the URL",
			expectErrorNum: 0,
		},
		{
			title:          "URL query parameters",
			f:              nil,
			URL:            "https://127.0.0.1:8708?seree",
			forceHttps:     true,
			hasError:       true,
			expectError:    "query parameters are not permitted in the URL",
			expectErrorNum: 0,
		},
	}
	for _, testCase := range testCases {
		result := ValidateWebhookURL(testCase.f, testCase.URL, testCase.forceHttps)
		if !testCase.hasError && result != nil {
			t.Errorf("it should not has error,but now it has error.")
		}
		if testCase.hasError && testCase.expectError != result[testCase.expectErrorNum].Detail {
			t.Errorf("the exepect error is: %v, but now the result is: %v", testCase.expectError, result[testCase.expectErrorNum].Detail)
		}
	}
}
