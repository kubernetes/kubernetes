/*
Copyright 2023 The Kubernetes Authors.

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

package generators

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_extractTag(t *testing.T) {
	testCases := map[string]struct {
		comments []string
		expect   []string
	}{
		"no comments": {
			comments: []string{},
			expect:   nil,
		},
		"no tag": {
			comments: []string{
				"+",
			},
			expect: nil,
		},
		"wrong tag": {
			comments: []string{
				"+k8s:wrong-tag",
			},
			expect: nil,
		},
		"no value": {
			comments: []string{
				"+k8s:conversion-gen=",
			},
			expect: []string{
				"",
			},
		},
		"value==false": {
			comments: []string{
				"+k8s:conversion-gen=false",
			},
			expect: []string{
				"false",
			},
		},
		"one comment": {
			comments: []string{
				"+k8s:conversion-gen=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
			},
		},
		"two different comments": {
			comments: []string{
				"+k8s:conversion-gen=k8s.io/kubernetes/runtime.Object",
				"+k8s:conversion-gen=k8s.io/kubernetes/runtime.List",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
				"k8s.io/kubernetes/runtime.List",
			},
		},
		"two same comments": {
			comments: []string{
				"+k8s:conversion-gen=k8s.io/kubernetes/runtime.Object",
				"+k8s:conversion-gen=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
				"k8s.io/kubernetes/runtime.Object",
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			r := extractTag(tc.comments)
			assert.Equal(t, tc.expect, r)
		})
	}
}

func Test_extractExplicitFromTag(t *testing.T) {
	testCases := map[string]struct {
		comments []string
		expect   []string
	}{
		"no comments": {
			comments: []string{},
			expect:   nil,
		},
		"no tag": {
			comments: []string{"+"},
			expect:   nil,
		},
		"wrong tag": {
			comments: []string{"+k8s:wrong-tag"},
			expect:   nil,
		},
		"no value": {
			comments: []string{"+k8s:conversion-gen:explicit-from="},
			expect:   []string{""},
		},
		"value==false": {
			comments: []string{"+k8s:conversion-gen:explicit-from=false"},
			expect:   []string{"false"},
		},
		"one comment": {
			comments: []string{
				"+k8s:conversion-gen:explicit-from=url.Values",
			},
			expect: []string{
				"url.Values",
			},
		},
		"two different comments": {
			comments: []string{
				"+k8s:conversion-gen:explicit-from=url.Values.Object",
				"+k8s:conversion-gen:explicit-from=url.Values.List",
			},
			expect: []string{
				"url.Values.Object",
				"url.Values.List",
			},
		},
		"two same comments": {
			comments: []string{
				"+k8s:conversion-gen:explicit-from=url.Values.Object",
				"+k8s:conversion-gen:explicit-from=url.Values.Object",
			},
			expect: []string{
				"url.Values.Object",
				"url.Values.Object",
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			r := extractExplicitFromTag(tc.comments)
			assert.Equal(t, tc.expect, r)
		})
	}
}

func Test_extractExternalTypesTag(t *testing.T) {
	testCases := map[string]struct {
		comments []string
		expect   []string
	}{
		"no comments": {
			comments: []string{},
			expect:   nil,
		},
		"no tag": {
			comments: []string{"+"},
			expect:   nil,
		},
		"wrong tag": {
			comments: []string{"+k8s:wrong-tag"},
			expect:   nil,
		},
		"no value": {
			comments: []string{"+k8s:conversion-gen-external-types="},
			expect:   []string{""},
		},
		"value==false": {
			comments: []string{"+k8s:conversion-gen-external-types=false"},
			expect:   []string{"false"},
		},
		"one comment": {
			comments: []string{
				"+k8s:conversion-gen-external-types=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
			},
		},
		"two different comments": {
			comments: []string{
				"+k8s:conversion-gen-external-types=k8s.io/kubernetes/runtime.Object",
				"+k8s:conversion-gen-external-types=k8s.io/kubernetes/runtime.List",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
				"k8s.io/kubernetes/runtime.List",
			},
		},
		"two same comments": {
			comments: []string{
				"+k8s:conversion-gen-external-types=k8s.io/kubernetes/runtime.Object",
				"+k8s:conversion-gen-external-types=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
				"k8s.io/kubernetes/runtime.Object",
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			r := extractExternalTypesTag(tc.comments)
			assert.Equal(t, tc.expect, r)
		})
	}
}

func Test_isCopyOnly(t *testing.T) {
	testCases := map[string]struct {
		comments []string
		expect   bool
	}{
		"no comments": {
			comments: []string{},
			expect:   false,
		},
		"no tag": {
			comments: []string{"+"},
			expect:   false,
		},
		"wrong tag": {
			comments: []string{"+k8s:wrong-tag"},
			expect:   false,
		},
		"no value": {
			comments: []string{"+k8s:conversion-fn="},
			expect:   false,
		},
		"value==false": {
			comments: []string{"+k8s:conversion-fn=false"},
			expect:   false,
		},
		"one copy-only": {
			comments: []string{
				"+k8s:conversion-fn=copy-only",
			},
			expect: true,
		},
		"two different comments": {
			comments: []string{
				"+k8s:conversion-fn=copy-only",
				"+k8s:conversion-fn=k8s.io/kubernetes/runtime.List",
			},
			expect: false,
		},
		"two same copy-only": {
			comments: []string{
				"+k8s:conversion-fn=copy-only",
				"+k8s:conversion-fn=copy-only",
			},
			expect: false,
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			r := isCopyOnly(tc.comments)
			assert.Equal(t, tc.expect, r)
		})
	}
}

func Test_isDrop(t *testing.T) {
	testCases := map[string]struct {
		comments []string
		expect   bool
	}{
		"no comments": {
			comments: []string{},
			expect:   false,
		},
		"no tag": {
			comments: []string{"+"},
			expect:   false,
		},
		"wrong tag": {
			comments: []string{"+k8s:wrong-tag"},
			expect:   false,
		},
		"no value": {
			comments: []string{"+k8s:conversion-fn="},
			expect:   false,
		},
		"value==false": {
			comments: []string{"+k8s:conversion-fn=false"},
			expect:   false,
		},
		"one isDrop": {
			comments: []string{
				"+k8s:conversion-fn=drop",
			},
			expect: true,
		},
		"two different comments": {
			comments: []string{
				"+k8s:conversion-fn=drop",
				"+k8s:conversion-fn=k8s.io/kubernetes/runtime.List",
			},
			expect: false,
		},
		"two same isDrop": {
			comments: []string{
				"+k8s:conversion-fn=drop",
				"+k8s:conversion-fn=drop",
			},
			expect: false,
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			r := isDrop(tc.comments)
			assert.Equal(t, tc.expect, r)
		})
	}
}
