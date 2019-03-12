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

package fake

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1unstructured "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
)

const (
	testGroup      = "testgroup"
	testVersion    = "testversion"
	testResource   = "testkinds"
	testNamespace  = "testns"
	testName       = "testname"
	testKind       = "TestKind"
	testAPIVersion = "testgroup/testversion"
)

func newUnstructuredWithRV(apiVersion, kind, namespace, name string, rv string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"namespace":       namespace,
				"name":            name,
				"resourceVersion": rv,
			},
		},
	}
}

func newUnstructured(apiVersion, kind, namespace, name string) *unstructured.Unstructured {
	return newUnstructuredWithRV(apiVersion, kind, namespace, name, "1")
}

func newUnstructuredWithSpec(spec map[string]interface{}) *unstructured.Unstructured {
	u := newUnstructured(testAPIVersion, testKind, testNamespace, testName)
	u.Object["spec"] = spec
	return u
}

func TestList(t *testing.T) {
	scheme := runtime.NewScheme()

	tt := []struct {
		name     string
		objs     []runtime.Object
		expected *metav1unstructured.UnstructuredList
	}{
		{
			name: "unstructured input",
			objs: []runtime.Object{
				newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo", "42"),
				newUnstructuredWithRV("group2/version", "TheKind", "ns-foo", "name2-foo", "6"),
				newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-bar", "1"),
				newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-baz", "2"),
				newUnstructuredWithRV("group2/version", "TheKind", "ns-foo", "name2-baz", "1"),
			},
			expected: &metav1unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"resourceVersion": "42",
					},
				},
				Items: []metav1unstructured.Unstructured{
					*newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo", "42"),
					*newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-bar", "1"),
					*newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-baz", "2"),
				},
			},
		},
		{
			name: "unstructured input RV test",
			objs: []runtime.Object{
				newUnstructuredWithRV("group2/version", "TheKind", "ns-foo", "name2-foo1", "99"),
				newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo1", "1"),
				newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo3", "3"),
				newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo2", "2"),
				newUnstructuredWithRV("group2/version", "TheKind", "ns-foo", "name2-foo2", "100"),
			},
			expected: &metav1unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"resourceVersion": "3",
					},
				},
				Items: []metav1unstructured.Unstructured{
					*newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo1", "1"),
					*newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo3", "3"),
					*newUnstructuredWithRV("group/version", "TheKind", "ns-foo", "name-foo2", "2"),
				},
			},
		},
		{
			name: "RV is set even with empty list",
			objs: []runtime.Object{},
			expected: &metav1unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						// We use RV "1" to simulate any RV in the real word but " " and "0" which are reserved values
						"resourceVersion": "1",
					},
				},
				Items: []metav1unstructured.Unstructured{},
			},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			client := NewSimpleDynamicClient(scheme, tc.objs...)
			list, err := client.Resource(schema.GroupVersionResource{Group: "group", Version: "version", Resource: "thekinds"}).List(metav1.ListOptions{})
			if err != nil {
				t.Fatal(err)
			}

			if !equality.Semantic.DeepEqual(tc.expected, list) {
				t.Errorf("\nexpected: %#v\ngot:      %#v\ndiff: %s", tc.expected, list, diff.ObjectReflectDiff(tc.expected, list))
			}
		})
	}
}

type patchTestCase struct {
	name                  string
	object                runtime.Object
	patchType             types.PatchType
	patchBytes            []byte
	wantErrMsg            string
	expectedPatchedObject runtime.Object
}

func (tc *patchTestCase) runner(t *testing.T) {
	client := NewSimpleDynamicClient(runtime.NewScheme(), tc.object)
	resourceInterface := client.Resource(schema.GroupVersionResource{Group: testGroup, Version: testVersion, Resource: testResource}).Namespace(testNamespace)

	got, recErr := resourceInterface.Patch(testName, tc.patchType, tc.patchBytes, metav1.PatchOptions{})

	if err := tc.verifyErr(recErr); err != nil {
		t.Error(err)
	}

	if err := tc.verifyResult(got); err != nil {
		t.Error(err)
	}

}

// verifyErr verifies that the given error returned from Patch is the error
// expected by the test case.
func (tc *patchTestCase) verifyErr(err error) error {
	if tc.wantErrMsg != "" && err == nil {
		return fmt.Errorf("want error, got nil")
	}

	if tc.wantErrMsg == "" && err != nil {
		return fmt.Errorf("want no error, got %v", err)
	}

	if err != nil {
		if want, got := tc.wantErrMsg, err.Error(); want != got {
			return fmt.Errorf("incorrect error: want: %q got: %q", want, got)
		}
	}
	return nil
}

func (tc *patchTestCase) verifyResult(result *unstructured.Unstructured) error {
	if tc.expectedPatchedObject == nil && result == nil {
		return nil
	}
	if !equality.Semantic.DeepEqual(result, tc.expectedPatchedObject) {
		return fmt.Errorf("unexpected diff in received object: %s", diff.ObjectGoPrintDiff(tc.expectedPatchedObject, result))
	}
	return nil
}

func TestPatch(t *testing.T) {
	testCases := []patchTestCase{
		{
			name:       "jsonpatch fails with merge type",
			object:     newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
			patchType:  types.StrategicMergePatchType,
			patchBytes: []byte(`[]`),
			wantErrMsg: "invalid JSON document",
		}, {
			name:      "jsonpatch works with empty patch",
			object:    newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
			patchType: types.JSONPatchType,
			// No-op
			patchBytes:            []byte(`[]`),
			expectedPatchedObject: newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
		}, {
			name:      "jsonpatch works with simple change patch",
			object:    newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
			patchType: types.JSONPatchType,
			// change spec.foo from bar to foobar
			patchBytes:            []byte(`[{"op": "replace", "path": "/spec/foo", "value": "foobar"}]`),
			expectedPatchedObject: newUnstructuredWithSpec(map[string]interface{}{"foo": "foobar"}),
		}, {
			name:      "jsonpatch works with simple addition",
			object:    newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
			patchType: types.JSONPatchType,
			// add spec.newvalue = dummy
			patchBytes:            []byte(`[{"op": "add", "path": "/spec/newvalue", "value": "dummy"}]`),
			expectedPatchedObject: newUnstructuredWithSpec(map[string]interface{}{"foo": "bar", "newvalue": "dummy"}),
		}, {
			name:      "jsonpatch works with simple deletion",
			object:    newUnstructuredWithSpec(map[string]interface{}{"foo": "bar", "toremove": "shouldnotbehere"}),
			patchType: types.JSONPatchType,
			// remove spec.newvalue = dummy
			patchBytes:            []byte(`[{"op": "remove", "path": "/spec/toremove"}]`),
			expectedPatchedObject: newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
		}, {
			name:      "strategic merge patch fails with JSONPatch",
			object:    newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
			patchType: types.StrategicMergePatchType,
			// add spec.newvalue = dummy
			patchBytes: []byte(`[{"op": "add", "path": "/spec/newvalue", "value": "dummy"}]`),
			wantErrMsg: "invalid JSON document",
		}, {
			name:       "merge patch fails as unsupported",
			object:     newUnstructured(testAPIVersion, testKind, testNamespace, testName),
			patchType:  types.MergePatchType,
			patchBytes: []byte(`{}`),
			wantErrMsg: "PatchType is not supported",
		},
		// TODO: Add tests for strategic merge using v1.Pod for example to ensure the test cases
		// demonstrate expected use cases.
	}

	for _, tc := range testCases {
		t.Run(tc.name, tc.runner)
	}
}
