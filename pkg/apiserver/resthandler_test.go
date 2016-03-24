/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/evanphx/json-patch"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
)

type testPatchType struct {
	unversioned.TypeMeta `json:",inline"`

	testPatchSubType `json:",inline"`
}

type testPatchSubType struct {
	StringField string `json:"theField"`
}

func (obj *testPatchType) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }

func TestPatchAnonymousField(t *testing.T) {
	originalJS := `{"kind":"testPatchType","theField":"my-value"}`
	patch := `{"theField": "changed!"}`
	expectedJS := `{"kind":"testPatchType","theField":"changed!"}`

	actualBytes, err := getPatchedJS(api.StrategicMergePatchType, []byte(originalJS), []byte(patch), &testPatchType{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(actualBytes) != expectedJS {
		t.Errorf("expected %v, got %v", expectedJS, string(actualBytes))
	}
}

type testPatcher struct {
	// startingPod is used for the first Get
	startingPod *api.Pod

	// updatePod is the pod that is used for conflict comparison and returned for the SECOND Get
	updatePod *api.Pod

	numGets int
}

func (p *testPatcher) New() runtime.Object {
	return &api.Pod{}
}

func (p *testPatcher) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	inPod := obj.(*api.Pod)
	if inPod.ResourceVersion != p.updatePod.ResourceVersion {
		return nil, false, apierrors.NewConflict(api.Resource("pods"), inPod.Name, fmt.Errorf("existing %v, new %v", p.updatePod.ResourceVersion, inPod.ResourceVersion))
	}

	return inPod, false, nil
}

func (p *testPatcher) Get(ctx api.Context, name string) (runtime.Object, error) {
	if p.numGets > 0 {
		return p.updatePod, nil
	}
	p.numGets++

	return p.startingPod, nil
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
func (p *testNamer) GenerateLink(req *restful.Request, obj runtime.Object) (path, query string, err error) {
	return "", "", errors.New("not implemented")
}

// GenerateLink creates a path and query for a list that represents the canonical path.
func (p *testNamer) GenerateListLink(req *restful.Request) (path, query string, err error) {
	return "", "", errors.New("not implemented")
}

type patchTestCase struct {
	name string

	// admission chain to use, nil is fine
	admit updateAdmissionFunc

	// startingPod is used for the first Get
	startingPod *api.Pod
	// changedPod is the "destination" pod for the patch.  The test will create a patch from the startingPod to the changedPod
	// to use when calling the patch operation
	changedPod *api.Pod
	// updatePod is the pod that is used for conflict comparison and returned for the SECOND Get
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
		admit = func(updatedObject runtime.Object) error {
			return nil
		}
	}

	testPatcher := &testPatcher{}
	testPatcher.startingPod = tc.startingPod
	testPatcher.updatePod = tc.updatePod

	ctx := api.NewDefaultContext()
	ctx = api.WithNamespace(ctx, namespace)

	namer := &testNamer{namespace, name}

	versionedObj, err := api.Scheme.ConvertToVersion(&api.Pod{}, "v1")
	if err != nil {
		t.Errorf("%s: unexpected error: %v", tc.name, err)
		return
	}

	for _, patchType := range []api.PatchType{api.JSONPatchType, api.MergePatchType, api.StrategicMergePatchType} {
		// TODO SUPPORT THIS!
		if patchType == api.JSONPatchType {
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
		case api.JSONPatchType:
			continue

		case api.StrategicMergePatchType:
			patch, err = strategicpatch.CreateStrategicMergePatch(originalObjJS, changedJS, versionedObj)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				return
			}

		case api.MergePatchType:
			patch, err = jsonpatch.CreateMergePatch(originalObjJS, changedJS)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				return
			}

		}

		resultObj, err := patchResource(ctx, admit, 1*time.Second, versionedObj, testPatcher, name, patchType, patch, namer, codec)
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
			t.Errorf("%s mismatch: %v\n", tc.name, util.ObjectGoPrintDiff(reallyExpectedPod, resultPod))
			return
		}
	}

}

func TestPatchResourceWithVersionConflict(t *testing.T) {
	namespace := "bar"
	name := "foo"
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
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = "v1"
	tc.updatePod.Spec.ActiveDeadlineSeconds = &fifteen
	tc.updatePod.Spec.NodeName = "anywhere"

	tc.expectedPod.Name = name
	tc.expectedPod.Namespace = namespace
	tc.expectedPod.ResourceVersion = "2"
	tc.expectedPod.Spec.ActiveDeadlineSeconds = &thirty
	tc.expectedPod.Spec.NodeName = "anywhere"

	tc.Run(t)
}

func TestPatchResourceWithConflict(t *testing.T) {
	namespace := "bar"
	name := "foo"

	tc := &patchTestCase{
		name: "TestPatchResourceWithConflict",

		startingPod: &api.Pod{},
		changedPod:  &api.Pod{},
		updatePod:   &api.Pod{},

		expectedError: `pods "foo" cannot be updated: existing 2, new 1`,
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.NodeName = "here"

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.NodeName = "there"

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = "v1"
	tc.updatePod.Spec.NodeName = "anywhere"

	tc.Run(t)
}

func TestPatchWithAdmissionRejection(t *testing.T) {
	namespace := "bar"
	name := "foo"
	fifteen := int64(15)
	thirty := int64(30)

	tc := &patchTestCase{
		name: "TestPatchWithAdmissionRejection",

		admit: func(updatedObject runtime.Object) error {
			return errors.New("admission failure")
		},

		startingPod: &api.Pod{},
		changedPod:  &api.Pod{},
		updatePod:   &api.Pod{},

		expectedError: "admission failure",
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.Run(t)
}

func TestPatchWithVersionConflictThenAdmissionFailure(t *testing.T) {
	namespace := "bar"
	name := "foo"
	fifteen := int64(15)
	thirty := int64(30)
	seen := false

	tc := &patchTestCase{
		name: "TestPatchWithVersionConflictThenAdmissionFailure",

		admit: func(updatedObject runtime.Object) error {
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
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = "v1"
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = "v1"
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = "v1"
	tc.updatePod.Spec.ActiveDeadlineSeconds = &fifteen
	tc.updatePod.Spec.NodeName = "anywhere"

	tc.Run(t)
}
