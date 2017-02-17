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

package fieldpath

import (
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestExtractFieldPathAsString(t *testing.T) {
	cases := []struct {
		name                    string
		fieldPath               string
		obj                     interface{}
		expectedValue           string
		expectedMessageFragment string
	}{
		{
			name:      "not an API object",
			fieldPath: "metadata.name",
			obj:       "",
			expectedMessageFragment: "expected struct",
		},
		{
			name:      "ok - namespace",
			fieldPath: "metadata.namespace",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "object-namespace",
				},
			},
			expectedValue: "object-namespace",
		},
		{
			name:      "ok - name",
			fieldPath: "metadata.name",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "object-name",
				},
			},
			expectedValue: "object-name",
		},
		{
			name:      "ok - labels",
			fieldPath: "metadata.labels",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"key": "value"},
				},
			},
			expectedValue: "key=\"value\"",
		},
		{
			name:      "ok - labels bslash n",
			fieldPath: "metadata.labels",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"key": "value\n"},
				},
			},
			expectedValue: "key=\"value\\n\"",
		},
		{
			name:      "ok - annotations",
			fieldPath: "metadata.annotations",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"builder": "john-doe"},
				},
			},
			expectedValue: "builder=\"john-doe\"",
		},
		{
			name:      "ok - annotation",
			fieldPath: "metadata.annotations['spec.pod.beta.kubernetes.io/statefulset-index']",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"spec.pod.beta.kubernetes.io/statefulset-index": "1"},
				},
			},
			expectedValue: "1",
		},

		{
			name:      "invalid expression",
			fieldPath: "metadata.whoops",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "object-namespace",
				},
			},
			expectedMessageFragment: "unsupported fieldPath",
		},
		{
			name:      "invalid annotation",
			fieldPath: "metadata.annotations['unknown.key']",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"foo": "bar"},
				},
			},
			expectedMessageFragment: "unsupported fieldPath",
		},
	}

	for _, tc := range cases {
		actual, err := ExtractFieldPathAsString(tc.obj, tc.fieldPath)
		if err != nil {
			if tc.expectedMessageFragment != "" {
				if !strings.Contains(err.Error(), tc.expectedMessageFragment) {
					t.Errorf("%v: unexpected error message: %q, expected to contain %q", tc.name, err, tc.expectedMessageFragment)
				}
			} else {
				t.Errorf("%v: unexpected error: %v", tc.name, err)
			}
		} else if e := tc.expectedValue; e != "" && e != actual {
			t.Errorf("%v: unexpected result; got %q, expected %q", tc.name, actual, e)
		}
	}
}

func TestSplitMaybeSubscriptedPath(t *testing.T) {
	cases := []struct {
		fieldPath         string
		expectedPath      string
		expectedSubscript string
	}{
		{
			fieldPath:         "metadata.annotations['key']",
			expectedPath:      "metadata.annotations",
			expectedSubscript: "key",
		},
		{
			fieldPath:         "metadata.annotations['a[b']c']",
			expectedPath:      "metadata.annotations",
			expectedSubscript: "a[b']c",
		},
		{
			fieldPath:         "metadata.labels['['key']",
			expectedPath:      "metadata.labels",
			expectedSubscript: "['key",
		},
		{
			fieldPath:         "metadata.labels['key']']",
			expectedPath:      "metadata.labels",
			expectedSubscript: "key']",
		},
		{
			fieldPath:         "metadata.labels[ 'key' ]",
			expectedPath:      "metadata.labels[ 'key' ]",
			expectedSubscript: "",
		},
		{
			fieldPath:         "metadata.labels['']",
			expectedPath:      "metadata.labels['']",
			expectedSubscript: "",
		},
		{
			fieldPath:         "metadata.labels[' ']",
			expectedPath:      "metadata.labels",
			expectedSubscript: " ",
		},
		{
			fieldPath:         "metadata.labels[]",
			expectedPath:      "metadata.labels[]",
			expectedSubscript: "",
		},
		{
			fieldPath:         "metadata.labels[']",
			expectedPath:      "metadata.labels[']",
			expectedSubscript: "",
		},
		{
			fieldPath:         "metadata.labels['key']foo",
			expectedPath:      "metadata.labels['key']foo",
			expectedSubscript: "",
		},
		{
			fieldPath:         "['key']",
			expectedPath:      "['key']",
			expectedSubscript: "",
		},
		{
			fieldPath:         "metadata.labels",
			expectedPath:      "metadata.labels",
			expectedSubscript: "",
		},
	}
	for _, tc := range cases {
		path, subscript := SplitMaybeSubscriptedPath(tc.fieldPath)
		if path != tc.expectedPath || subscript != tc.expectedSubscript {
			t.Errorf("SplitMaybeSubscriptedPath(%q) = (%q, %q), expect (%q, %q)",
				tc.fieldPath, path, subscript, tc.expectedPath, tc.expectedSubscript)
		}
	}
}
