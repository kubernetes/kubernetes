/*
Copyright 2014 The Kubernetes Authors.

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

package handlers

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/evanphx/json-patch"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/apiserver/pkg/registry/rest"
)

type testPatchType struct {
	metav1.TypeMeta `json:",inline"`

	TestPatchSubType `json:",inline"`
}

// We explicitly make it public as private types doesn't
// work correctly with json inlined types.
type TestPatchSubType struct {
	StringField string `json:"theField"`
}

func (obj *testPatchType) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }

func TestPatchAnonymousField(t *testing.T) {
	testGV := schema.GroupVersion{Group: "", Version: "v"}
	api.Scheme.AddKnownTypes(testGV, &testPatchType{})
	codec := api.Codecs.LegacyCodec(testGV)

	original := &testPatchType{
		TypeMeta:         metav1.TypeMeta{Kind: "testPatchType", APIVersion: "v"},
		TestPatchSubType: TestPatchSubType{StringField: "my-value"},
	}
	patch := `{"theField": "changed!"}`
	expected := &testPatchType{
		TypeMeta:         metav1.TypeMeta{Kind: "testPatchType", APIVersion: "v"},
		TestPatchSubType: TestPatchSubType{StringField: "changed!"},
	}

	actual := &testPatchType{}
	_, _, err := strategicPatchObject(codec, original, []byte(patch), actual, &testPatchType{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !apiequality.Semantic.DeepEqual(actual, expected) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}

type testPatcher struct {
	t *testing.T

	// startingPod is used for the first Update
	startingPod *api.Pod

	// updatePod is the pod that is used for conflict comparison and used for subsequent Update calls
	updatePod *api.Pod

	numUpdates int
}

func (p *testPatcher) New() runtime.Object {
	return &api.Pod{}
}

func (p *testPatcher) Update(ctx request.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	currentPod := p.startingPod
	if p.numUpdates > 0 {
		currentPod = p.updatePod
	}
	p.numUpdates++

	obj, err := objInfo.UpdatedObject(ctx, currentPod)
	if err != nil {
		return nil, false, err
	}
	inPod := obj.(*api.Pod)
	if inPod.ResourceVersion != p.updatePod.ResourceVersion {
		return nil, false, apierrors.NewConflict(api.Resource("pods"), inPod.Name, fmt.Errorf("existing %v, new %v", p.updatePod.ResourceVersion, inPod.ResourceVersion))
	}

	return inPod, false, nil
}

func (p *testPatcher) Get(ctx request.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	p.t.Fatal("Unexpected call to testPatcher.Get")
	return nil, errors.New("Unexpected call to testPatcher.Get")
}

type testNamer struct {
	namespace string
	name      string
}

func (p *testNamer) Namespace(req *restful.Request) (namespace string, err error) {
	return p.namespace, nil
}

// Name returns the name from the request, and an optional namespace value if this is a namespace
// scoped call. An error is returned if the name is not available.
func (p *testNamer) Name(req *restful.Request) (namespace, name string, err error) {
	return p.namespace, p.name, nil
}

// ObjectName returns the namespace and name from an object if they exist, or an error if the object
// does not support names.
func (p *testNamer) ObjectName(obj runtime.Object) (namespace, name string, err error) {
	return p.namespace, p.name, nil
}

// SetSelfLink sets the provided URL onto the object. The method should return nil if the object
// does not support selfLinks.
func (p *testNamer) SetSelfLink(obj runtime.Object, url string) error {
	return errors.New("not implemented")
}

// GenerateLink creates a path and query for a given runtime object that represents the canonical path.
func (p *testNamer) GenerateLink(req *restful.Request, obj runtime.Object) (uri string, err error) {
	return "", errors.New("not implemented")
}

// GenerateLink creates a path and query for a list that represents the canonical path.
func (p *testNamer) GenerateListLink(req *restful.Request) (uri string, err error) {
	return "", errors.New("not implemented")
}

type patchTestCase struct {
	name string

	// admission chain to use, nil is fine
	admit updateAdmissionFunc

	// startingPod is used as the starting point for the first Update
	startingPod *api.Pod
	// changedPod is the "destination" pod for the patch.  The test will create a patch from the startingPod to the changedPod
	// to use when calling the patch operation
	changedPod *api.Pod
	// updatePod is the pod that is used for conflict comparison and as the starting point for the second Update
	updatePod *api.Pod

	// expectedPod is the pod that you expect to get back after the patch is complete
	expectedPod   *api.Pod
	expectedError string
}

func (tc *patchTestCase) Run(t *testing.T) {
	t.Logf("Starting test %s", tc.name)

	namespace := tc.startingPod.Namespace
	name := tc.startingPod.Name

	codec := testapi.Default.Codec()
	admit := tc.admit
	if admit == nil {
		admit = func(updatedObject runtime.Object, currentObject runtime.Object) error {
			return nil
		}
	}

	testPatcher := &testPatcher{}
	testPatcher.t = t
	testPatcher.startingPod = tc.startingPod
	testPatcher.updatePod = tc.updatePod

	ctx := request.NewDefaultContext()
	ctx = request.WithNamespace(ctx, namespace)

	namer := &testNamer{namespace, name}
	copier := runtime.ObjectCopier(api.Scheme)
	resource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	versionedObj := &v1.Pod{}

	for _, patchType := range []types.PatchType{types.JSONPatchType, types.MergePatchType, types.StrategicMergePatchType} {
		// TODO SUPPORT THIS!
		if patchType == types.JSONPatchType {
			continue
		}
		t.Logf("Working with patchType %v", patchType)

		originalObjJS, err := runtime.Encode(codec, tc.startingPod)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			return
		}
		changedJS, err := runtime.Encode(codec, tc.changedPod)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			return
		}

		patch := []byte{}
		switch patchType {
		case types.JSONPatchType:
			continue

		case types.StrategicMergePatchType:
			patch, err = strategicpatch.CreateTwoWayMergePatch(originalObjJS, changedJS, versionedObj)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				return
			}

		case types.MergePatchType:
			patch, err = jsonpatch.CreateMergePatch(originalObjJS, changedJS)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				return
			}

		}

		resultObj, err := patchResource(ctx, admit, 1*time.Second, versionedObj, testPatcher, name, patchType, patch, namer, copier, resource, codec)
		if len(tc.expectedError) != 0 {
			if err == nil || err.Error() != tc.expectedError {
				t.Errorf("%s: expected error %v, but got %v", tc.name, tc.expectedError, err)
				return
			}
		} else {
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				return
			}
		}

		if tc.expectedPod == nil {
			if resultObj != nil {
				t.Errorf("%s: unexpected result: %v", tc.name, resultObj)
			}
			return
		}

		resultPod := resultObj.(*api.Pod)

		// roundtrip to get defaulting
		expectedJS, err := runtime.Encode(codec, tc.expectedPod)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			return
		}
		expectedObj, err := runtime.Decode(codec, expectedJS)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			return
		}
		reallyExpectedPod := expectedObj.(*api.Pod)

		if !reflect.DeepEqual(*reallyExpectedPod, *resultPod) {
			t.Errorf("%s mismatch: %v\n", tc.name, diff.ObjectGoPrintDiff(reallyExpectedPod, resultPod))
			return
		}
	}

}

func TestPatchResourceWithVersionConflict(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")
	fifteen := int64(15)
	thirty := int64(30)

	tc := &patchTestCase{
		name: "TestPatchResourceWithVersionConflict",

		startingPod: &api.Pod{},
		changedPod:  &api.Pod{},
		updatePod:   &api.Pod{},

		expectedPod: &api.Pod{},
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = "v1"
	tc.updatePod.Spec.ActiveDeadlineSeconds = &fifteen
	tc.updatePod.Spec.NodeName = "anywhere"

	tc.expectedPod.Name = name
	tc.expectedPod.Namespace = namespace
	tc.expectedPod.UID = uid
	tc.expectedPod.ResourceVersion = "2"
	tc.expectedPod.Spec.ActiveDeadlineSeconds = &thirty
	tc.expectedPod.Spec.NodeName = "anywhere"

	tc.Run(t)
}

func TestPatchResourceWithConflict(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")

	tc := &patchTestCase{
		name: "TestPatchResourceWithConflict",

		startingPod: &api.Pod{},
		changedPod:  &api.Pod{},
		updatePod:   &api.Pod{},

		expectedError: `Operation cannot be fulfilled on pods "foo": existing 2, new 1`,
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.NodeName = "here"

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.NodeName = "there"

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = "v1"
	tc.updatePod.Spec.NodeName = "anywhere"

	tc.Run(t)
}

func TestPatchWithAdmissionRejection(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")
	fifteen := int64(15)
	thirty := int64(30)

	tc := &patchTestCase{
		name: "TestPatchWithAdmissionRejection",

		admit: func(updatedObject runtime.Object, currentObject runtime.Object) error {
			return errors.New("admission failure")
		},

		startingPod: &api.Pod{},
		changedPod:  &api.Pod{},
		updatePod:   &api.Pod{},

		expectedError: "admission failure",
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.Run(t)
}

func TestPatchWithVersionConflictThenAdmissionFailure(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")
	fifteen := int64(15)
	thirty := int64(30)
	seen := false

	tc := &patchTestCase{
		name: "TestPatchWithVersionConflictThenAdmissionFailure",

		admit: func(updatedObject runtime.Object, currentObject runtime.Object) error {
			if seen {
				return errors.New("admission failure")
			}

			seen = true
			return nil
		},

		startingPod: &api.Pod{},
		changedPod:  &api.Pod{},
		updatePod:   &api.Pod{},

		expectedError: "admission failure",
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = "v1"
	tc.updatePod.Spec.ActiveDeadlineSeconds = &fifteen
	tc.updatePod.Spec.NodeName = "anywhere"

	tc.Run(t)
}

func TestHasUID(t *testing.T) {
	testcases := []struct {
		obj    runtime.Object
		hasUID bool
	}{
		{obj: nil, hasUID: false},
		{obj: &api.Pod{}, hasUID: false},
		{obj: nil, hasUID: false},
		{obj: runtime.Object(nil), hasUID: false},
		{obj: &api.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("A")}}, hasUID: true},
	}
	for i, tc := range testcases {
		actual, err := hasUID(tc.obj)
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if tc.hasUID != actual {
			t.Errorf("%d: expected %v, got %v", i, tc.hasUID, actual)
		}
	}
}

func TestParseTimeout(t *testing.T) {
	if d := parseTimeout(""); d != 30*time.Second {
		t.Errorf("blank timeout produces %v", d)
	}
	if d := parseTimeout("not a timeout"); d != 30*time.Second {
		t.Errorf("bad timeout produces %v", d)
	}
	if d := parseTimeout("10s"); d != 10*time.Second {
		t.Errorf("10s timeout produced: %v", d)
	}
}
