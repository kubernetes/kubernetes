/*
Copyright 2018 The Kubernetes Authors.

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

package templates

import (
	"testing"
)

func TestExamples(t *testing.T) {
	tests := []struct {
		name     string
		example  string
		expected string
	}{
		{
			name:     "empty string",
			example:  "",
			expected: "",
		},
		{
			name:     "single line",
			example:  "example",
			expected: "  example",
		},
		{
			name: "no indent",
			example: "" +
				"first line\n" +
				"second line",
			expected: "" +
				"  first line\n" +
				"  second line",
		},
		{
			name: "deep indent using tabs",
			example: "" +
				"			first line\n" +
				"			second line",
			expected: "" +
				"  first line\n" +
				"  second line",
		},
		{
			name: "multi-level indent",
			example: "" +
				"first level\n" +
				"  second level\n" +
				"    third level",
			expected: "" +
				"  first level\n" +
				"    second level\n" +
				"      third level",
		},
		{
			name: "multi-level with base indent",
			example: "" +
				"    first level\n" +
				"      second level\n" +
				"        third level",
			expected: "" +
				"  first level\n" +
				"    second level\n" +
				"      third level",
		},
		{
			name: "multi-level with base indent and starting newline",
			example: "\n" +
				"  first level\n" +
				"    second level\n" +
				"      third level",
			expected: "" +
				"  first level\n" +
				"    second level\n" +
				"      third level",
		},
		{
			name: "bad original indentation",
			example: "" +
				"  this line has a base indent\n" +
				"this line doesn't",
			expected: "" +
				"  this line has a base indent\n" +
				"  this line doesn't",
		},
		{
			name:     "remove leading and trailing whitespace",
			example:  "\n        example   \n   ",
			expected: "  example",
		},
		{
			name: "preserve empty lines in the middle",
			example: "" +
				"first line\n" +
				"\n" +
				"third line",
			expected: "" +
				"  first line\n" +
				"  \n" +
				"  third line",
		},
		{
			name: "actual example from pkg/kubectl/cmd/apply.go",
			example: "\n" +
				"		# Apply the configuration in pod.json to a pod.\n" +
				"		kubectl apply -f ./pod.json\n" +
				"\n" +
				"		# Apply the JSON passed into stdin to a pod.\n" +
				"		cat pod.json | kubectl apply -f -\n",
			expected: "" +
				"  # Apply the configuration in pod.json to a pod.\n" +
				"  kubectl apply -f ./pod.json\n" +
				"  \n" +
				"  # Apply the JSON passed into stdin to a pod.\n" +
				"  cat pod.json | kubectl apply -f -",
		},
		{
			name: "actual example from service catalog's svcat CLI tool",
			example: `
  svcat provision wordpress-mysql-instance --class mysqldb --plan free -p location=eastus -p sslEnforcement=disabled
  svcat provision secure-instance --class mysqldb --plan secureDB --params-json '{
    "encrypt" : true,
    "firewallRules" : [
        {
            "name": "AllowSome",
            "startIPAddress": "75.70.113.50",
            "endIPAddress" : "75.70.113.131"
        }
    ]
}
'
`,
			expected: `  svcat provision wordpress-mysql-instance --class mysqldb --plan free -p location=eastus -p sslEnforcement=disabled
  svcat provision secure-instance --class mysqldb --plan secureDB --params-json '{
    "encrypt" : true,
    "firewallRules" : [
        {
            "name": "AllowSome",
            "startIPAddress": "75.70.113.50",
            "endIPAddress" : "75.70.113.131"
        }
    ]
  }
  '`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := Examples(test.example)
			if actual != test.expected {
				t.Errorf("Test case '%s': Unexpected indentation.\nExpected:\n%s\nActual:\n%s", test.name, test.expected, actual)
			}
		})
	}
}
