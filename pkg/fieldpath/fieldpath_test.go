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
			name:      "ok - annotation",
			fieldPath: "metadata.annotations['Www.k8s.io/test']",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"Www.k8s.io/test": "1"},
				},
			},
			expectedValue: "1",
		},
		{
			name:      "ok - uid",
			fieldPath: "metadata.uid",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID: "b70b3269-858e-12a8-9cf2-1232a194038a",
				},
			},
			expectedValue: "b70b3269-858e-12a8-9cf2-1232a194038a",
		},
		{
			name:      "ok - label",
			fieldPath: "metadata.labels['something']",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"something": "label value",
					},
				},
			},
			expectedValue: "label value",
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
			name:      "invalid annotation key",
			fieldPath: "metadata.annotations['invalid~key']",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"foo": "bar"},
				},
			},
			expectedMessageFragment: "invalid key subscript in metadata.annotations",
		},
		{
			name:      "invalid label key",
			fieldPath: "metadata.labels['Www.k8s.io/test']",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"foo": "bar"},
				},
			},
			expectedMessageFragment: "invalid key subscript in metadata.labels",
		},
		{
			name:                    "invalid subscript",
			fieldPath:               "metadata.notexisting['something']",
			obj:                     &v1.Pod{},
			expectedMessageFragment: "fieldPath \"metadata.notexisting['something']\" does not support subscript",
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
		} else if tc.expectedMessageFragment != "" {
			t.Errorf("%v: expected error: %v", tc.name, tc.expectedMessageFragment)
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
		expectedOK        bool
	}{
		{
			fieldPath:         "metadata.annotations['key']",
			expectedPath:      "metadata.annotations",
			expectedSubscript: "key",
			expectedOK:        true,
		},
		{
			fieldPath:         "metadata.annotations['a[b']c']",
			expectedPath:      "metadata.annotations",
			expectedSubscript: "a[b']c",
			expectedOK:        true,
		},
		{
			fieldPath:         "metadata.labels['['key']",
			expectedPath:      "metadata.labels",
			expectedSubscript: "['key",
			expectedOK:        true,
		},
		{
			fieldPath:         "metadata.labels['key']']",
			expectedPath:      "metadata.labels",
			expectedSubscript: "key']",
			expectedOK:        true,
		},
		{
			fieldPath:         "metadata.labels['']",
			expectedPath:      "metadata.labels",
			expectedSubscript: "",
			expectedOK:        true,
		},
		{
			fieldPath:         "metadata.labels[' ']",
			expectedPath:      "metadata.labels",
			expectedSubscript: " ",
			expectedOK:        true,
		},
		{
			fieldPath:  "metadata.labels[ 'key' ]",
			expectedOK: false,
		},
		{
			fieldPath:  "metadata.labels[]",
			expectedOK: false,
		},
		{
			fieldPath:  "metadata.labels[']",
			expectedOK: false,
		},
		{
			fieldPath:  "metadata.labels['key']foo",
			expectedOK: false,
		},
		{
			fieldPath:  "['key']",
			expectedOK: false,
		},
		{
			fieldPath:  "metadata.labels",
			expectedOK: false,
		},
	}
	for _, tc := range cases {
		path, subscript, ok := SplitMaybeSubscriptedPath(tc.fieldPath)
		if !ok {
			if tc.expectedOK {
				t.Errorf("SplitMaybeSubscriptedPath(%q) expected to return (_, _, true)", tc.fieldPath)
			}
			continue
		}
		if path != tc.expectedPath || subscript != tc.expectedSubscript {
			t.Errorf("SplitMaybeSubscriptedPath(%q) = (%q, %q, true), expect (%q, %q, true)",
				tc.fieldPath, path, subscript, tc.expectedPath, tc.expectedSubscript)
		}
	}
}
