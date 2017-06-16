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
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/evanphx/json-patch"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	example.AddToScheme(scheme)
	examplev1.AddToScheme(scheme)
}

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
	scheme.AddKnownTypes(testGV, &testPatchType{})
	codec := codecs.LegacyCodec(testGV)
	defaulter := runtime.ObjectDefaulter(scheme)

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
	err := strategicPatchObject(codec, defaulter, original, []byte(patch), actual, &testPatchType{})
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
	startingPod *example.Pod

	// updatePod is the pod that is used for conflict comparison and used for subsequent Update calls
	updatePod *example.Pod

	numUpdates int
}

func (p *testPatcher) New() runtime.Object {
	return &example.Pod{}
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
	inPod := obj.(*example.Pod)
	if inPod.ResourceVersion != p.updatePod.ResourceVersion {
		return nil, false, apierrors.NewConflict(example.Resource("pods"), inPod.Name, fmt.Errorf("existing %v, new %v", p.updatePod.ResourceVersion, inPod.ResourceVersion))
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

func (p *testNamer) Namespace(req *http.Request) (namespace string, err error) {
	return p.namespace, nil
}

// Name returns the name from the request, and an optional namespace value if this is a namespace
// scoped call. An error is returned if the name is not available.
func (p *testNamer) Name(req *http.Request) (namespace, name string, err error) {
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
func (p *testNamer) GenerateLink(requestInfo *request.RequestInfo, obj runtime.Object) (uri string, err error) {
	return "", errors.New("not implemented")
}

// GenerateListLink creates a path and query for a list that represents the canonical path.
func (p *testNamer) GenerateListLink(req *http.Request) (uri string, err error) {
	return "", errors.New("not implemented")
}

type patchTestCase struct {
	name string

	// admission chain to use, nil is fine
	admit updateAdmissionFunc

	// startingPod is used as the starting point for the first Update
	startingPod *example.Pod
	// changedPod is the "destination" pod for the patch.  The test will create a patch from the startingPod to the changedPod
	// to use when calling the patch operation
	changedPod *example.Pod
	// updatePod is the pod that is used for conflict comparison and as the starting point for the second Update
	updatePod *example.Pod

	// expectedPod is the pod that you expect to get back after the patch is complete
	expectedPod   *example.Pod
	expectedError string
}

func (tc *patchTestCase) Run(t *testing.T) {
	t.Logf("Starting test %s", tc.name)

	namespace := tc.startingPod.Namespace
	name := tc.startingPod.Name

	codec := codecs.LegacyCodec(examplev1.SchemeGroupVersion)
	admit := tc.admit
	if admit == nil {
		admit = func(updatedObject runtime.Object, currentObject runtime.Object) error {
			return nil
		}
	}

	ctx := request.NewDefaultContext()
	ctx = request.WithNamespace(ctx, namespace)

	namer := &testNamer{namespace, name}
	copier := runtime.ObjectCopier(scheme)
	creater := runtime.ObjectCreater(scheme)
	defaulter := runtime.ObjectDefaulter(scheme)
	convertor := runtime.UnsafeObjectConvertor(scheme)
	kind := examplev1.SchemeGroupVersion.WithKind("Pod")
	resource := examplev1.SchemeGroupVersion.WithResource("pods")
	versionedObj := &examplev1.Pod{}

	for _, patchType := range []types.PatchType{types.JSONPatchType, types.MergePatchType, types.StrategicMergePatchType} {
		// This needs to be reset on each iteration.
		testPatcher := &testPatcher{
			t:           t,
			startingPod: tc.startingPod,
			updatePod:   tc.updatePod,
		}

		// TODO SUPPORT THIS!
		if patchType == types.JSONPatchType {
			continue
		}
		t.Logf("Working with patchType %v", patchType)

		originalObjJS, err := runtime.Encode(codec, tc.startingPod)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}
		changedJS, err := runtime.Encode(codec, tc.changedPod)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}

		patch := []byte{}
		switch patchType {
		case types.JSONPatchType:
			continue

		case types.StrategicMergePatchType:
			patch, err = strategicpatch.CreateTwoWayMergePatch(originalObjJS, changedJS, versionedObj)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				continue
			}

		case types.MergePatchType:
			patch, err = jsonpatch.CreateMergePatch(originalObjJS, changedJS)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				continue
			}

		}

		resultObj, err := patchResource(ctx, admit, 1*time.Second, versionedObj, testPatcher, name, patchType, patch, namer, copier, creater, defaulter, convertor, kind, resource, codec)
		if len(tc.expectedError) != 0 {
			if err == nil || err.Error() != tc.expectedError {
				t.Errorf("%s: expected error %v, but got %v", tc.name, tc.expectedError, err)
				continue
			}
		} else {
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
				continue
			}
		}

		if tc.expectedPod == nil {
			if resultObj != nil {
				t.Errorf("%s: unexpected result: %v", tc.name, resultObj)
			}
			continue
		}

		resultPod := resultObj.(*example.Pod)

		// roundtrip to get defaulting
		expectedJS, err := runtime.Encode(codec, tc.expectedPod)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}
		expectedObj, err := runtime.Decode(codec, expectedJS)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}
		reallyExpectedPod := expectedObj.(*example.Pod)

		if !reflect.DeepEqual(*reallyExpectedPod, *resultPod) {
			t.Errorf("%s mismatch: %v\n", tc.name, diff.ObjectGoPrintDiff(reallyExpectedPod, resultPod))
			continue
		}
	}

}

func TestNumberConversion(t *testing.T) {
	codec := codecs.LegacyCodec(examplev1.SchemeGroupVersion)
	defaulter := runtime.ObjectDefaulter(scheme)

	terminationGracePeriodSeconds := int64(42)
	activeDeadlineSeconds := int64(42)
	currentVersionedObject := &examplev1.Pod{
		TypeMeta:   metav1.TypeMeta{Kind: "Example", APIVersion: examplev1.SchemeGroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{Name: "test-example"},
		Spec: examplev1.PodSpec{
			TerminationGracePeriodSeconds: &terminationGracePeriodSeconds,
			ActiveDeadlineSeconds:         &activeDeadlineSeconds,
		},
	}
	versionedObjToUpdate := &examplev1.Pod{}
	versionedObj := &examplev1.Pod{}

	patchJS := []byte(`{"spec":{"terminationGracePeriodSeconds":42,"activeDeadlineSeconds":120}}`)

	err := strategicPatchObject(codec, defaulter, currentVersionedObject, patchJS, versionedObjToUpdate, versionedObj)
	if err != nil {
		t.Fatal(err)
	}
	if versionedObjToUpdate.Spec.TerminationGracePeriodSeconds == nil || *versionedObjToUpdate.Spec.TerminationGracePeriodSeconds != 42 ||
		versionedObjToUpdate.Spec.ActiveDeadlineSeconds == nil || *versionedObjToUpdate.Spec.ActiveDeadlineSeconds != 120 {
		t.Fatal(errors.New("Ports failed to merge because of number conversion issue"))
	}
}

func TestPatchResourceNumberConversion(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")
	fifteen := int64(15)
	thirty := int64(30)

	tc := &patchTestCase{
		name: "TestPatchResourceNumberConversion",

		startingPod: &example.Pod{},
		changedPod:  &example.Pod{},
		updatePod:   &example.Pod{},

		expectedPod: &example.Pod{},
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	// Patch tries to change to 30.
	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	// Someone else already changed it to 30.
	// This should be fine since it's not a "meaningful conflict".
	// Previously this was detected as a meaningful conflict because int64(30) != float64(30).
	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.updatePod.Spec.ActiveDeadlineSeconds = &thirty
	tc.updatePod.Spec.NodeName = "anywhere"

	tc.expectedPod.Name = name
	tc.expectedPod.Namespace = namespace
	tc.expectedPod.UID = uid
	tc.expectedPod.ResourceVersion = "2"
	tc.expectedPod.Spec.ActiveDeadlineSeconds = &thirty
	tc.expectedPod.Spec.NodeName = "anywhere"

	tc.Run(t)
}

func TestPatchResourceWithVersionConflict(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")
	fifteen := int64(15)
	thirty := int64(30)

	tc := &patchTestCase{
		name: "TestPatchResourceWithVersionConflict",

		startingPod: &example.Pod{},
		changedPod:  &example.Pod{},
		updatePod:   &example.Pod{},

		expectedPod: &example.Pod{},
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = examplev1.SchemeGroupVersion.String()
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

		startingPod: &example.Pod{},
		changedPod:  &example.Pod{},
		updatePod:   &example.Pod{},

		expectedError: `Operation cannot be fulfilled on pods.example.apiserver.k8s.io "foo": existing 2, new 1`,
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.startingPod.Spec.NodeName = "here"

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.changedPod.Spec.NodeName = "there"

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = examplev1.SchemeGroupVersion.String()
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

		startingPod: &example.Pod{},
		changedPod:  &example.Pod{},
		updatePod:   &example.Pod{},

		expectedError: "admission failure",
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = examplev1.SchemeGroupVersion.String()
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

		startingPod: &example.Pod{},
		changedPod:  &example.Pod{},
		updatePod:   &example.Pod{},

		expectedError: "admission failure",
	}

	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "1"
	tc.startingPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.startingPod.Spec.ActiveDeadlineSeconds = &fifteen

	tc.changedPod.Name = name
	tc.changedPod.Namespace = namespace
	tc.changedPod.UID = uid
	tc.changedPod.ResourceVersion = "1"
	tc.changedPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.changedPod.Spec.ActiveDeadlineSeconds = &thirty

	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "2"
	tc.updatePod.APIVersion = examplev1.SchemeGroupVersion.String()
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
		{obj: &example.Pod{}, hasUID: false},
		{obj: nil, hasUID: false},
		{obj: runtime.Object(nil), hasUID: false},
		{obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("A")}}, hasUID: true},
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

func TestFinishRequest(t *testing.T) {
	exampleObj := &example.Pod{}
	exampleErr := fmt.Errorf("error")
	successStatusObj := &metav1.Status{Status: metav1.StatusSuccess, Message: "success message"}
	errorStatusObj := &metav1.Status{Status: metav1.StatusFailure, Message: "error message"}
	testcases := []struct {
		timeout     time.Duration
		fn          resultFunc
		expectedObj runtime.Object
		expectedErr error
	}{
		{
			// Expected obj is returned.
			timeout: time.Second,
			fn: func() (runtime.Object, error) {
				return exampleObj, nil
			},
			expectedObj: exampleObj,
			expectedErr: nil,
		},
		{
			// Expected error is returned.
			timeout: time.Second,
			fn: func() (runtime.Object, error) {
				return nil, exampleErr
			},
			expectedObj: nil,
			expectedErr: exampleErr,
		},
		{
			// Successful status object is returned as expected.
			timeout: time.Second,
			fn: func() (runtime.Object, error) {
				return successStatusObj, nil
			},
			expectedObj: successStatusObj,
			expectedErr: nil,
		},
		{
			// Error status object is converted to StatusError.
			timeout: time.Second,
			fn: func() (runtime.Object, error) {
				return errorStatusObj, nil
			},
			expectedObj: nil,
			expectedErr: apierrors.FromObject(errorStatusObj),
		},
	}
	for i, tc := range testcases {
		obj, err := finishRequest(tc.timeout, tc.fn)
		if (err == nil && tc.expectedErr != nil) || (err != nil && tc.expectedErr == nil) || (err != nil && tc.expectedErr != nil && err.Error() != tc.expectedErr.Error()) {
			t.Errorf("%d: unexpected err. expected: %v, got: %v", i, tc.expectedErr, err)
		}
		if !apiequality.Semantic.DeepEqual(obj, tc.expectedObj) {
			t.Errorf("%d: unexpected obj. expected %#v, got %#v", i, tc.expectedObj, obj)
		}
	}
}
