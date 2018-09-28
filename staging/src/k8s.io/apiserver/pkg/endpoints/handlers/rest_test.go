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
	"context"
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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
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

func (obj *testPatchType) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	clone := *obj
	return &clone
}

func TestPatchAnonymousField(t *testing.T) {
	testGV := schema.GroupVersion{Group: "", Version: "v"}
	scheme.AddKnownTypes(testGV, &testPatchType{})
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
	err := strategicPatchObject(defaulter, original, []byte(patch), actual, &testPatchType{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !apiequality.Semantic.DeepEqual(actual, expected) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}

func TestStrategicMergePatchInvalid(t *testing.T) {
	testGV := schema.GroupVersion{Group: "", Version: "v"}
	scheme.AddKnownTypes(testGV, &testPatchType{})
	defaulter := runtime.ObjectDefaulter(scheme)

	original := &testPatchType{
		TypeMeta:         metav1.TypeMeta{Kind: "testPatchType", APIVersion: "v"},
		TestPatchSubType: TestPatchSubType{StringField: "my-value"},
	}
	patch := `barbaz`
	expectedError := "invalid character 'b' looking for beginning of value"

	actual := &testPatchType{}
	err := strategicPatchObject(defaulter, original, []byte(patch), actual, &testPatchType{})
	if !apierrors.IsBadRequest(err) {
		t.Errorf("expected HTTP status: BadRequest, got: %#v", apierrors.ReasonForError(err))
	}
	if err.Error() != expectedError {
		t.Errorf("expected %#v, got %#v", expectedError, err.Error())
	}
}

func TestJSONPatch(t *testing.T) {
	for _, test := range []struct {
		name              string
		patch             string
		expectedError     string
		expectedErrorType metav1.StatusReason
	}{
		{
			name:  "valid",
			patch: `[{"op": "test", "value": "podA", "path": "/metadata/name"}]`,
		},
		{
			name:              "invalid-syntax",
			patch:             `invalid json patch`,
			expectedError:     "invalid character 'i' looking for beginning of value",
			expectedErrorType: metav1.StatusReasonBadRequest,
		},
		{
			name:              "invalid-semantics",
			patch:             `[{"op": "test", "value": "podA", "path": "/invalid/path"}]`,
			expectedError:     "the server rejected our request due to an error in our request",
			expectedErrorType: metav1.StatusReasonInvalid,
		},
	} {
		p := &patcher{
			patchType: types.JSONPatchType,
			patchJS:   []byte(test.patch),
		}
		jp := jsonPatcher{p}
		codec := codecs.LegacyCodec(examplev1.SchemeGroupVersion)
		pod := &examplev1.Pod{}
		pod.Name = "podA"
		versionedJS, err := runtime.Encode(codec, pod)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.name, err)
			continue
		}
		_, err = jp.applyJSPatch(versionedJS)
		if err != nil {
			if len(test.expectedError) == 0 {
				t.Errorf("%s: expect no error when applying json patch, but got %v", test.name, err)
				continue
			}
			if err.Error() != test.expectedError {
				t.Errorf("%s: expected error %v, but got %v", test.name, test.expectedError, err)
			}
			if test.expectedErrorType != apierrors.ReasonForError(err) {
				t.Errorf("%s: expected error type %v, but got %v", test.name, test.expectedErrorType, apierrors.ReasonForError(err))
			}
		} else if len(test.expectedError) > 0 {
			t.Errorf("%s: expected err %s", test.name, test.expectedError)
		}
	}
}

func TestPatchCustomResource(t *testing.T) {
	testGV := schema.GroupVersion{Group: "mygroup.example.com", Version: "v1beta1"}
	scheme.AddKnownTypes(testGV, &unstructured.Unstructured{})
	defaulter := runtime.ObjectDefaulter(scheme)

	original := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1beta1",
			"kind":       "Noxu",
			"metadata": map[string]interface{}{
				"namespace": "Namespaced",
				"name":      "foo",
			},
			"spec": map[string]interface{}{
				"num": "10",
			},
		},
	}
	patch := `{"spec":{"num":"20"}}`
	expectedError := "strategic merge patch format is not supported"

	actual := &unstructured.Unstructured{}
	err := strategicPatchObject(defaulter, original, []byte(patch), actual, &unstructured.Unstructured{})
	if !apierrors.IsBadRequest(err) {
		t.Errorf("expected HTTP status: BadRequest, got: %#v", apierrors.ReasonForError(err))
	}
	if err.Error() != expectedError {
		t.Errorf("expected %#v, got %#v", expectedError, err.Error())
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

func (p *testPatcher) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// Simulate GuaranteedUpdate behavior (retries internally on etcd changes if the incoming resource doesn't pin resourceVersion)
	for {
		currentPod := p.startingPod
		if p.numUpdates > 0 {
			currentPod = p.updatePod
		}
		p.numUpdates++

		// Remember the current resource version
		currentResourceVersion := currentPod.ResourceVersion

		obj, err := objInfo.UpdatedObject(ctx, currentPod)
		if err != nil {
			return nil, false, err
		}
		inPod := obj.(*example.Pod)
		if inPod.ResourceVersion == "" || inPod.ResourceVersion == "0" {
			inPod.ResourceVersion = p.updatePod.ResourceVersion
		}
		if inPod.ResourceVersion != p.updatePod.ResourceVersion {
			// If the patch didn't have an opinion on the resource version, retry like GuaranteedUpdate does
			if inPod.ResourceVersion == currentResourceVersion {
				continue
			}
			// If the patch changed the resource version and it mismatches, conflict
			return nil, false, apierrors.NewConflict(example.Resource("pods"), inPod.Name, fmt.Errorf("existing %v, new %v", p.updatePod.ResourceVersion, inPod.ResourceVersion))
		}

		if currentPod == nil {
			if err := createValidation(currentPod); err != nil {
				return nil, false, err
			}
		} else {
			if err := updateValidation(currentPod, inPod); err != nil {
				return nil, false, err
			}
		}

		return inPod, false, nil
	}
}

func (p *testPatcher) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
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
	admissionMutation   mutateObjectUpdateFunc
	admissionValidation rest.ValidateObjectUpdateFunc

	// startingPod is used as the starting point for the first Update
	startingPod *example.Pod
	// changedPod can be set as the "destination" pod for the patch, and the test will compute a patch from the startingPod to the changedPod,
	// or patches can be set directly using strategicMergePatch, mergePatch, and jsonPatch
	changedPod          *example.Pod
	strategicMergePatch string
	mergePatch          string
	jsonPatch           string

	// updatePod is the pod that is used for conflict comparison and as the starting point for the second Update
	updatePod *example.Pod

	// expectedPod is the pod that you expect to get back after the patch is complete
	expectedPod   *example.Pod
	expectedError string
	// if set, indicates the number of times patching was expected to be attempted
	expectedTries int
}

func (tc *patchTestCase) Run(t *testing.T) {
	t.Logf("Starting test %s", tc.name)

	namespace := tc.startingPod.Namespace
	name := tc.startingPod.Name

	codec := codecs.LegacyCodec(examplev1.SchemeGroupVersion)

	admissionMutation := tc.admissionMutation
	if admissionMutation == nil {
		admissionMutation = func(updatedObject runtime.Object, currentObject runtime.Object) error {
			return nil
		}
	}
	admissionValidation := tc.admissionValidation
	if admissionValidation == nil {
		admissionValidation = func(updatedObject runtime.Object, currentObject runtime.Object) error {
			return nil
		}
	}

	ctx := request.NewDefaultContext()
	ctx = request.WithNamespace(ctx, namespace)

	namer := &testNamer{namespace, name}
	creater := runtime.ObjectCreater(scheme)
	defaulter := runtime.ObjectDefaulter(scheme)
	convertor := runtime.UnsafeObjectConvertor(scheme)
	kind := examplev1.SchemeGroupVersion.WithKind("Pod")
	resource := examplev1.SchemeGroupVersion.WithResource("pods")
	schemaReferenceObj := &examplev1.Pod{}

	for _, patchType := range []types.PatchType{types.JSONPatchType, types.MergePatchType, types.StrategicMergePatchType} {
		// This needs to be reset on each iteration.
		testPatcher := &testPatcher{
			t:           t,
			startingPod: tc.startingPod,
			updatePod:   tc.updatePod,
		}

		t.Logf("Working with patchType %v", patchType)

		patch := []byte{}
		switch patchType {
		case types.StrategicMergePatchType:
			patch = []byte(tc.strategicMergePatch)
			if len(patch) == 0 {
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
				patch, err = strategicpatch.CreateTwoWayMergePatch(originalObjJS, changedJS, schemaReferenceObj)
				if err != nil {
					t.Errorf("%s: unexpected error: %v", tc.name, err)
					continue
				}
			}

		case types.MergePatchType:
			patch = []byte(tc.mergePatch)
			if len(patch) == 0 {
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
				patch, err = jsonpatch.CreateMergePatch(originalObjJS, changedJS)
				if err != nil {
					t.Errorf("%s: unexpected error: %v", tc.name, err)
					continue
				}
			}

		case types.JSONPatchType:
			patch = []byte(tc.jsonPatch)
			if len(patch) == 0 {
				// TODO SUPPORT THIS!
				continue
			}

		default:
			t.Error("unsupported patch type")
		}

		p := patcher{
			namer:           namer,
			creater:         creater,
			defaulter:       defaulter,
			unsafeConvertor: convertor,
			kind:            kind,
			resource:        resource,

			createValidation: rest.ValidateAllObjectFunc,
			updateValidation: admissionValidation,
			admissionCheck:   admissionMutation,

			codec: codec,

			timeout: 1 * time.Second,

			restPatcher: testPatcher,
			name:        name,
			patchType:   patchType,
			patchJS:     patch,

			trace: utiltrace.New("Patch" + name),
		}

		resultObj, err := p.patchResource(ctx)
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

		if tc.expectedTries > 0 {
			if tc.expectedTries != testPatcher.numUpdates {
				t.Errorf("%s: expected %d tries, got %d", tc.name, tc.expectedTries, testPatcher.numUpdates)
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
	schemaReferenceObj := &examplev1.Pod{}

	patchJS := []byte(`{"spec":{"terminationGracePeriodSeconds":42,"activeDeadlineSeconds":120}}`)

	err := strategicPatchObject(defaulter, currentVersionedObject, patchJS, versionedObjToUpdate, schemaReferenceObj)
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

	setTcPod(tc.startingPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &fifteen, "")

	// Patch tries to change to 30.
	setTcPod(tc.changedPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &thirty, "")

	// Someone else already changed it to 30.
	// This should be fine since it's not a "meaningful conflict".
	// Previously this was detected as a meaningful conflict because int64(30) != float64(30).
	setTcPod(tc.updatePod, name, namespace, uid, "2", examplev1.SchemeGroupVersion.String(), &thirty, "anywhere")

	setTcPod(tc.expectedPod, name, namespace, uid, "2", "", &thirty, "anywhere")

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

	setTcPod(tc.startingPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &fifteen, "")

	setTcPod(tc.changedPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &thirty, "")

	setTcPod(tc.updatePod, name, namespace, uid, "2", examplev1.SchemeGroupVersion.String(), &fifteen, "anywhere")

	setTcPod(tc.expectedPod, name, namespace, uid, "2", "", &thirty, "anywhere")

	tc.Run(t)
}

func TestPatchResourceWithStaleVersionConflict(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")

	tc := &patchTestCase{
		name: "TestPatchResourceWithStaleVersionConflict",

		startingPod: &example.Pod{},
		updatePod:   &example.Pod{},

		expectedError: `Operation cannot be fulfilled on pods.example.apiserver.k8s.io "foo": existing 2, new 1`,
		expectedTries: 1,
	}

	// starting pod is at rv=2
	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "2"
	tc.startingPod.APIVersion = examplev1.SchemeGroupVersion.String()
	// same pod is still in place when attempting to persist the update
	tc.updatePod = tc.startingPod

	// patches are submitted with a stale rv=1
	tc.mergePatch = `{"metadata":{"resourceVersion":"1"},"spec":{"nodeName":"foo"}}`
	tc.strategicMergePatch = `{"metadata":{"resourceVersion":"1"},"spec":{"nodeName":"foo"}}`

	tc.Run(t)
}

func TestPatchResourceWithRacingVersionConflict(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")

	tc := &patchTestCase{
		name: "TestPatchResourceWithRacingVersionConflict",

		startingPod: &example.Pod{},
		updatePod:   &example.Pod{},

		expectedError: `Operation cannot be fulfilled on pods.example.apiserver.k8s.io "foo": existing 3, new 2`,
		expectedTries: 2,
	}

	// starting pod is at rv=2
	tc.startingPod.Name = name
	tc.startingPod.Namespace = namespace
	tc.startingPod.UID = uid
	tc.startingPod.ResourceVersion = "2"
	tc.startingPod.APIVersion = examplev1.SchemeGroupVersion.String()

	// pod with rv=3 is found when attempting to persist the update
	tc.updatePod.Name = name
	tc.updatePod.Namespace = namespace
	tc.updatePod.UID = uid
	tc.updatePod.ResourceVersion = "3"
	tc.updatePod.APIVersion = examplev1.SchemeGroupVersion.String()

	// patches are submitted with a rv=2
	tc.mergePatch = `{"metadata":{"resourceVersion":"2"},"spec":{"nodeName":"foo"}}`
	tc.strategicMergePatch = `{"metadata":{"resourceVersion":"2"},"spec":{"nodeName":"foo"}}`

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
		expectedPod: &example.Pod{},
	}

	// See issue #63104 for discussion of how much sense this makes.

	setTcPod(tc.startingPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), nil, "here")

	setTcPod(tc.changedPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), nil, "there")

	setTcPod(tc.updatePod, name, namespace, uid, "2", examplev1.SchemeGroupVersion.String(), nil, "anywhere")

	tc.expectedPod.Name = name
	tc.expectedPod.Namespace = namespace
	tc.expectedPod.UID = uid
	tc.expectedPod.ResourceVersion = "2"
	tc.expectedPod.APIVersion = examplev1.SchemeGroupVersion.String()
	tc.expectedPod.Spec.NodeName = "there"

	tc.Run(t)
}

func TestPatchWithAdmissionRejection(t *testing.T) {
	namespace := "bar"
	name := "foo"
	uid := types.UID("uid")
	fifteen := int64(15)
	thirty := int64(30)

	type Test struct {
		name                string
		admissionMutation   mutateObjectUpdateFunc
		admissionValidation rest.ValidateObjectUpdateFunc
		expectedError       string
	}
	for _, test := range []Test{
		{
			name: "TestPatchWithMutatingAdmissionRejection",
			admissionMutation: func(updatedObject runtime.Object, currentObject runtime.Object) error {
				return errors.New("mutating admission failure")
			},
			admissionValidation: rest.ValidateAllObjectUpdateFunc,
			expectedError:       "mutating admission failure",
		},
		{
			name:              "TestPatchWithValidatingAdmissionRejection",
			admissionMutation: rest.ValidateAllObjectUpdateFunc,
			admissionValidation: func(updatedObject runtime.Object, currentObject runtime.Object) error {
				return errors.New("validating admission failure")
			},
			expectedError: "validating admission failure",
		},
		{
			name: "TestPatchWithBothAdmissionRejections",
			admissionMutation: func(updatedObject runtime.Object, currentObject runtime.Object) error {
				return errors.New("mutating admission failure")
			},
			admissionValidation: func(updatedObject runtime.Object, currentObject runtime.Object) error {
				return errors.New("validating admission failure")
			},
			expectedError: "mutating admission failure",
		},
	} {
		tc := &patchTestCase{
			name: test.name,

			admissionMutation:   test.admissionMutation,
			admissionValidation: test.admissionValidation,

			startingPod: &example.Pod{},
			changedPod:  &example.Pod{},
			updatePod:   &example.Pod{},

			expectedError: test.expectedError,
		}

		setTcPod(tc.startingPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &fifteen, "")

		setTcPod(tc.changedPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &thirty, "")

		setTcPod(tc.updatePod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &fifteen, "")

		tc.Run(t)
	}
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

		admissionMutation: func(updatedObject runtime.Object, currentObject runtime.Object) error {
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

	setTcPod(tc.startingPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &fifteen, "")

	setTcPod(tc.changedPod, name, namespace, uid, "1", examplev1.SchemeGroupVersion.String(), &thirty, "")

	setTcPod(tc.updatePod, name, namespace, uid, "2", examplev1.SchemeGroupVersion.String(), &fifteen, "anywhere")

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

func setTcPod(tcPod *example.Pod, name string, namespace string, uid types.UID, resourceVersion string, apiVersion string, activeDeadlineSeconds *int64, nodeName string) {
	tcPod.Name = name
	tcPod.Namespace = namespace
	tcPod.UID = uid
	tcPod.ResourceVersion = resourceVersion
	if len(apiVersion) != 0 {
		tcPod.APIVersion = apiVersion
	}
	if activeDeadlineSeconds != nil {
		tcPod.Spec.ActiveDeadlineSeconds = activeDeadlineSeconds
	}
	if len(nodeName) != 0 {
		tcPod.Spec.NodeName = nodeName
	}
}
