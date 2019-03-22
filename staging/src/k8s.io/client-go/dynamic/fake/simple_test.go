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

func newUnstructured(apiVersion, kind, namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
		},
	}
}

func newUnstructuredWithSpec(spec map[string]interface{}) *unstructured.Unstructured {
	u := newUnstructured(testAPIVersion, testKind, testNamespace, testName)
	u.Object["spec"] = spec
	return u
}

func TestList(t *testing.T) {
	scheme := runtime.NewScheme()

	client := NewSimpleDynamicClient(scheme,
		newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
		newUnstructured("group2/version", "TheKind", "ns-foo", "name2-foo"),
		newUnstructured("group/version", "TheKind", "ns-foo", "name-bar"),
		newUnstructured("group/version", "TheKind", "ns-foo", "name-baz"),
		newUnstructured("group2/version", "TheKind", "ns-foo", "name2-baz"),
	)
	listFirst, err := client.Resource(schema.GroupVersionResource{Group: "group", Version: "version", Resource: "thekinds"}).List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}

	expected := []unstructured.Unstructured{
		*newUnstructured("group/version", "TheKind", "ns-foo", "name-foo"),
		*newUnstructured("group/version", "TheKind", "ns-foo", "name-bar"),
		*newUnstructured("group/version", "TheKind", "ns-foo", "name-baz"),
	}
	if !equality.Semantic.DeepEqual(listFirst.Items, expected) {
		t.Fatal(diff.ObjectGoPrintDiff(expected, listFirst.Items))
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
			name:                  "merge patch works with simple replacement",
			object:                newUnstructuredWithSpec(map[string]interface{}{"foo": "bar"}),
			patchType:             types.MergePatchType,
			patchBytes:            []byte(`{ "spec": { "foo": "baz" } }`),
			expectedPatchedObject: newUnstructuredWithSpec(map[string]interface{}{"foo": "baz"}),
		},
		// TODO: Add tests for strategic merge using v1.Pod for example to ensure the test cases
		// demonstrate expected use cases.
	}

	for _, tc := range testCases {
		t.Run(tc.name, tc.runner)
	}
}
