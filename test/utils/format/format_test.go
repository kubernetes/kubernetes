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

package format_test

import (
	"fmt"
	"regexp"
	"testing"

	"github.com/onsi/gomega/format"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
)

func TestGomegaFormatObject(t *testing.T) {
	for name, test := range map[string]struct {
		value       interface{}
		expected    string
		indentation uint
	}{
		"int":            {value: 1, expected: `<int>: 1`},
		"string":         {value: "hello world", expected: `<string>: "hello world"`},
		"struct":         {value: myStruct{a: 1, b: 2}, expected: `<format_test.myStruct>: {a: 1, b: 2}`},
		"gomegastringer": {value: typeWithGomegaStringer(2), expected: `<format_test.typeWithGomegaStringer>: my stringer 2`},
		"pod": {value: v1.Pod{}, expected: `<v1.Pod>: 
    metadata: {}
    spec:
      containers: null
    status: {}`},
		"pod-indented": {value: v1.Pod{}, indentation: 1, expected: `    <v1.Pod>: 
        metadata: {}
        spec:
          containers: null
        status: {}`},
		"pod-ptr": {value: &v1.Pod{}, expected: `<*v1.Pod | <hex>>: 
    metadata: {}
    spec:
      containers: null
    status: {}`},
		"pod-hash": {value: map[string]v1.Pod{}, expected: `<map[string]v1.Pod | len:0>: 
    {}`},
		"podlist": {value: v1.PodList{}, expected: `<v1.PodList>: 
    items: null
    metadata: {}`},
	} {
		t.Run(name, func(t *testing.T) {
			actual := format.Object(test.value, test.indentation)
			actual = regexp.MustCompile(`\| 0x[a-z0-9]+`).ReplaceAllString(actual, `| <hex>`)
			assert.Equal(t, test.expected, actual)
		})
	}

}

type typeWithGomegaStringer int

func (v typeWithGomegaStringer) GomegaString() string {
	return fmt.Sprintf("my stringer %d", v)
}

type myStruct struct {
	a, b int
}
