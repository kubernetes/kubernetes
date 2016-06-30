/*
Copyright 2015 The Kubernetes Authors.

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

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_syncExamples(t *testing.T) {
	var podExample = `apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
`
	var textExample = `some text
`
	var cases = []struct {
		in       string
		expected string
	}{
		{"", ""},
		{
			"<!-- BEGIN MUNGE: EXAMPLE testdata/pod.yaml -->\n<!-- END MUNGE: EXAMPLE testdata/pod.yaml -->\n",
			"<!-- BEGIN MUNGE: EXAMPLE testdata/pod.yaml -->\n\n```yaml\n" + podExample + "```\n\n[Download example](testdata/pod.yaml?raw=true)\n<!-- END MUNGE: EXAMPLE testdata/pod.yaml -->\n",
		},
		{
			"<!-- BEGIN MUNGE: EXAMPLE ../mungedocs/testdata/pod.yaml -->\n<!-- END MUNGE: EXAMPLE ../mungedocs/testdata/pod.yaml -->\n",
			"<!-- BEGIN MUNGE: EXAMPLE ../mungedocs/testdata/pod.yaml -->\n\n```yaml\n" + podExample + "```\n\n[Download example](../mungedocs/testdata/pod.yaml?raw=true)\n<!-- END MUNGE: EXAMPLE ../mungedocs/testdata/pod.yaml -->\n",
		},
		{
			"<!-- BEGIN MUNGE: EXAMPLE testdata/example.txt -->\n<!-- END MUNGE: EXAMPLE testdata/example.txt -->\n",
			"<!-- BEGIN MUNGE: EXAMPLE testdata/example.txt -->\n\n```\n" + textExample + "```\n\n[Download example](testdata/example.txt?raw=true)\n<!-- END MUNGE: EXAMPLE testdata/example.txt -->\n",
		},
	}
	repoRoot = ""
	for _, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.expected)
		actual, err := syncExamples("filename.md", in)
		assert.NoError(t, err)
		if !expected.Equal(actual) {
			t.Errorf("Expected example \n'%q' but got \n'%q'", expected.String(), actual.String())
		}
	}
}
