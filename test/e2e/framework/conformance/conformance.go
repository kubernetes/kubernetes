/*
Copyright 2025 The Kubernetes Authors.

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

package architecture

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	gtypes "github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/utils/ptr"
	k8sjson "sigs.k8s.io/json"
)

// ResourceTestcaseInterface describes how to test one particular API endpoint
// by executing different operations against it.
//
// The content of created or patched objects is verified by ensuring that
// all fields are set as in the sent object. Extra fields or map entries
// are ignored.
//
// Basic create/read/update/delete (CRUD) semantic is covered, which
// is the minimum that is required for conformance testing of a
// GA feature. Actual functional testing is desirable, but not
// required.
//
// See [ResourceTestcase] for an implementation of this interface
// where test data is provided as Go objects and patch strings.
type ResourceTestcaseInterface interface {
	// GetGroupVersionResource returns the API group, version, and resource (plural form, lower case).
	GetGroupVersionResource() schema.GroupVersionResource

	// IsNamespaced defines whether the object must be created inside a namespace.
	IsNamespaced() bool

	// HasStatus defines whether the resource has a "status" sub-resource.
	//
	// Other sub-resources are not supported by this common test code.
	HasStatus() bool

	// VerifyContent defines whether the content of objects returned by
	// the apiserver gets compared against the content that was sent.
	//
	// If enabled, all field values that were sent must also be included
	// in the returned object. Additional fields and list or map entries
	// may get added (for example, because of defaulting or mutating
	// admission).
	//
	// This should not be enabled in conformance tests because admission
	// is allowed to modify what is being stored.
	VerifyContent() bool

	// GetInitialObject returns the data which is going to be used in a Create call.
	//
	// For cluster-scoped resources the test namespace can be used
	// to create a name which does not conflict with other objects
	// because it is unique while the test runs.
	//
	// It does not need to be set for namespaced resources because the
	// caller will ensure that. The caller cannot do that for the name because
	// different resources have different rules for what names are valid
	GetInitialObject(namespace string) *unstructured.Unstructured

	// GetUpdateSpec modifies an existing object.
	// It gets called for the result of creating the initial object.
	//
	// Ideally it should  change the spec (hence the name).
	// If that is impossible, then adding some label is also okay.
	// The goal is to add some fields that can be checked for
	// after an Update.
	//
	// May modify and return the input object.
	GetUpdateSpec(object *unstructured.Unstructured) *unstructured.Unstructured

	// GetUpdateStatus modifies the status of an existing object.
	// It gets called for the result of creating the initial object
	// and then updating its spec.
	//
	// May modify and return the input object.
	GetUpdateStatus(object *unstructured.Unstructured) *unstructured.Unstructured

	// GetPatchSpec describes how to generate patches.
	//
	// Each patch is applied to the initial object by itself, without the other patches.
	// An empty slice is valid and disables testing of patching. This may not be sufficient
	// for full conformance testing of the resource.
	//
	// If content verification is enabled, then this must cause the same change as GetUpdateSpec
	// because verification of the patch result uses the GetUpdateSpec result as reference.
	GetPatchSpec() []Patch

	// GetPatchStatus is like GetPatchSpec for the status.
	//
	// The initial object with the updated spec gets patched,
	// so the result must match the result of GetUpdateStatus
	// applied to GetInitialObject if content verification is
	// enabled.
	GetPatchStatus(object *unstructured.Unstructured) []Patch
}

// Patch contains the parameters for a Patch API call.
//
// The data must match an existing object.
//
// There's no retry loop because of conflicts, so the patch should not include
// a check of the ResourceVersion. Checking the UID in the patch is encouraged to prevent
// patching a replaced resource.
type Patch struct {
	GetData func(object *unstructured.Unstructured) []byte
	Type    types.PatchType
}

// ResourceTestcase provides test data for testing operations for a resource.
// Test data is based on the native Go type of the resource.
// The template parameter must be a pointer to the native Go type.
//
// The data is used like this:
// - create InitialSpec -> update with UpdateSpec -> update status with UpdateStatus
// - create InitialSpec -> apply StrategicMergePatchSpec and compare against UpdateSpec -> apply StrategicMergePatchStatus and compare against UpdateStatus
type ResourceTestcase[T runtime.Object] struct {
	// GroupResourceVersion identifies the API group, version and resource (plural form, lower case)
	// within that API which is to be tested.
	GVR schema.GroupVersionResource

	// Namespaced must be true if the resource must be created in a
	// namespace, false if it is cluster-scoped. Leaving it unset is
	// an error.
	//
	// Namespaced resources get created in the test namespace.
	//
	// The name of cluster-scoped resources gets extended with
	// `-<test namespace name>` to make it unique.
	Namespaced *bool

	// ContentVerificationEnabled defines whether the content of objects returned by
	// the apiserver gets compared against the content that was sent.
	//
	// If enabled, all field values that were sent must also be included
	// in the returned object. Additional fields and list or map entries
	// may get added (for example, because of defaulting or mutating
	// admission).
	//
	// This should not be enabled in conformance tests because admission
	// is allowed to modify what is being stored.
	ContentVerificationEnabled bool

	// InitialSpec must contain the initial state of a valid resource, without a status.
	InitialSpec T

	// UpdateSpec gets called for the created initial object
	// and must update something, ideally the spec (hence the name).
	// If that is not possible, then adding some label also works
	// for the sake of testing an update.
	//
	// May modify and return the input object.
	UpdateSpec func(T) T

	// UpdateStatus gets called for the updated object
	// and must add a status.
	//
	// May be nil if no status is supported.
	//
	// May modify and return the input object.
	UpdateStatus func(T) T

	// StrategicMergePatchSpec may modify fields in InitialSpec
	// with a strategic merge patch
	// (https://github.com/kubernetes/community/blob/master/contributors/devel/sig-api-machinery/strategic-merge-patch.md).
	// Muse use JSON encoding.
	//
	// If content verification is enabled, then this must contain the same change as UpdateSpec
	// because verification of the patch result uses UpdateSpec as reference.
	StrategicMergePatchSpec string

	// StrategicMergePatchStatus may add status fields
	// with a strategic merge patch
	// (https://github.com/kubernetes/community/blob/master/contributors/devel/sig-api-machinery/strategic-merge-patch.md)
	// Must use JSON encoding.
	//
	// The initial object with the updated spec gets patched,
	// so the result must match GetUpdateStatus
	// applied to GetInitialObject if content verification is
	// enabled.
	//
	// If empty, the status sub-resource is not getting tested.
	// May contain the name, but that's not required.
	StrategicMergePatchStatus string

	// ApplyPatchSpec corresponds to StrategicMergePatchSpec,
	// using the JSON encoding of a server-side-apply (SSA) patch
	// (https://kubernetes.io/docs/reference/using-api/server-side-apply).
	ApplyPatchSpec string

	// ApplyPatchStatus corresponds to StrategicMergePatchStatus,
	// using the JSON encoding of a server-side-apply (SSA) patch
	// (https://kubernetes.io/docs/reference/using-api/server-side-apply).
	ApplyPatchStatus string

	// JSONPatchSpec corresponds to StrategicMergePatchSpec,
	// using a JSON patch (https://tools.ietf.org/html/rfc6902).
	JSONPatchSpec string

	// JSONPatchStatus corresponds to StrategicMergePatchStatus,
	// using a JSON patch (https://tools.ietf.org/html/rfc6902).
	JSONPatchStatus string

	// JSONMergePatchSpec corresponds to StrategicMergePatchSpec,
	// using a JSON merge patch (https://tools.ietf.org/html/rfc7386).
	JSONMergePatchSpec string

	// JSONMergePatchStatus corresponds to StrategicMergePatchStatus,
	// using a JSON merge patch (https://tools.ietf.org/html/rfc7386).
	JSONMergePatchStatus string
}

var _ ResourceTestcaseInterface = &ResourceTestcase[*v1.Pod]{}

func (tc *ResourceTestcase[T]) GetGroupVersionResource() schema.GroupVersionResource {
	return tc.GVR
}

func (tc *ResourceTestcase[T]) IsNamespaced() bool {
	if tc.Namespaced == nil {
		framework.Fail("Test case error: Namespaced must be set")
	}

	return *tc.Namespaced
}

func (tc *ResourceTestcase[T]) HasStatus() bool {
	return tc.UpdateStatus != nil
}

func (tc *ResourceTestcase[T]) VerifyContent() bool {
	return tc.ContentVerificationEnabled
}

func (tc *ResourceTestcase[T]) GetInitialObject(namespace string) *unstructured.Unstructured {
	object := tc.toUnstructured("InitialSpec", tc.InitialSpec)
	if object.GetName() == "" {
		object.SetName("test")
	}
	if !tc.IsNamespaced() {
		object.SetName(object.GetName() + "-" + namespace)
	}

	return object
}

func (tc *ResourceTestcase[T]) GetUpdateSpec(in *unstructured.Unstructured) *unstructured.Unstructured {
	out := tc.fromUnstructured("existing object", in)
	out = tc.UpdateSpec(out)
	return tc.toUnstructured("updated object", out)
}

func (tc *ResourceTestcase[T]) GetUpdateStatus(in *unstructured.Unstructured) *unstructured.Unstructured {
	out := tc.fromUnstructured("updated object", in)
	out = tc.UpdateStatus(out)
	return tc.toUnstructured("updated object with status", out)
}

func (tc *ResourceTestcase[T]) GetPatchSpec() []Patch {
	var patches []Patch

	if tc.StrategicMergePatchSpec != "" {
		patches = append(patches, Patch{
			Type: types.StrategicMergePatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				patch := tc.createPatchObject("StrategicMergePatchSpec", tc.StrategicMergePatchSpec, existingObject)

				jsonData, err := patch.MarshalJSON()
				framework.ExpectNoError(err, "re-encode spec patch as JSON")

				return jsonData
			},
		})
	}

	if tc.ApplyPatchSpec != "" {
		patches = append(patches, Patch{
			Type: types.ApplyPatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				patch := tc.createPatchObject("ApplyPatchSpec", tc.ApplyPatchSpec, existingObject)

				jsonData, err := patch.MarshalJSON()
				framework.ExpectNoError(err, "re-encode spec patch as JSON")

				return jsonData
			},
		})
	}

	if tc.JSONMergePatchSpec != "" {
		patches = append(patches, Patch{
			Type: types.MergePatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				patch := tc.createPatchObject("JSONMergePatchSpec", tc.JSONMergePatchSpec, existingObject)

				jsonData, err := patch.MarshalJSON()
				framework.ExpectNoError(err, "re-encode spec patch as JSON")

				return jsonData
			},
		})
	}

	if tc.JSONPatchSpec != "" {
		patches = append(patches, Patch{
			Type: types.JSONPatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				return []byte(tc.JSONPatchSpec)
			},
		})
	}

	return patches
}

func (tc *ResourceTestcase[T]) GetPatchStatus(object *unstructured.Unstructured) []Patch {
	var patches []Patch

	if tc.StrategicMergePatchStatus != "" {
		patches = append(patches, Patch{
			Type: types.StrategicMergePatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				patch := tc.createPatchObject("StrategicMergePatchStatus", tc.StrategicMergePatchStatus, existingObject)

				jsonData, err := patch.MarshalJSON()
				framework.ExpectNoError(err, "re-encode status patch as JSON")

				return jsonData
			},
		})
	}

	if tc.ApplyPatchStatus != "" {
		patches = append(patches, Patch{
			Type: types.ApplyPatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				patch := tc.createPatchObject("ApplyPatchStatus", tc.ApplyPatchStatus, existingObject)

				jsonData, err := patch.MarshalJSON()
				framework.ExpectNoError(err, "re-encode status patch as JSON")

				return jsonData
			},
		})
	}

	if tc.JSONMergePatchStatus != "" {
		patches = append(patches, Patch{
			Type: types.MergePatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				patch := tc.createPatchObject("JSONMergePatchStatus", tc.JSONMergePatchStatus, existingObject)

				jsonData, err := patch.MarshalJSON()
				framework.ExpectNoError(err, "re-encode status patch as JSON")

				return jsonData
			},
		})
	}

	if tc.JSONPatchStatus != "" {
		patches = append(patches, Patch{
			Type: types.JSONPatchType,
			GetData: func(existingObject *unstructured.Unstructured) []byte {
				return []byte(tc.JSONPatchStatus)
			},
		})
	}

	return patches
}

func (tc *ResourceTestcase[T]) toUnstructured(what string, in T) *unstructured.Unstructured {
	data, err := json.Marshal(in)
	framework.ExpectNoError(err, "encode %s as JSON", what)

	out := tc.toUnstructuredFromJSON(what, data)

	return out
}

func (tc *ResourceTestcase[T]) toUnstructuredFromJSON(what string, in []byte) *unstructured.Unstructured {
	// UnmarshalCaseSensitivePreserveInts does not need kind (in contrast to unstructured.Unstructured.UnmarshalJSON)
	// and matches the behavior of preserving ints that we get when receiving from the apiserver (in contrast to plain json.Unmarshal).
	var out unstructured.Unstructured
	err := k8sjson.UnmarshalCaseSensitivePreserveInts(in, &out.Object)
	framework.ExpectNoError(err, "decode %s from JSON", what)

	return &out

}

func (tc *ResourceTestcase[T]) fromUnstructured(what string, in *unstructured.Unstructured) T {
	data, err := in.MarshalJSON()
	framework.ExpectNoError(err, "encode %s as JSON", what)

	var out T
	err = k8sjson.UnmarshalCaseSensitivePreserveInts(data, &out)
	framework.ExpectNoError(err, "decode %s from JSON", what)

	return out

}

// createPatchObject parses JSON data and then copies namespace/name/uid/kind/apiVersion from the existing object
// to make the patch complete. This works for strategic merge patches, apply patches and JSON merge patches.
func (tc *ResourceTestcase[T]) createPatchObject(what string, data string, existingObject *unstructured.Unstructured) *unstructured.Unstructured {
	object := tc.toUnstructuredFromJSON(what, []byte(data))
	object.SetNamespace(existingObject.GetNamespace())
	object.SetName(existingObject.GetName())
	object.SetUID(existingObject.GetUID())
	object.SetAPIVersion(existingObject.GetAPIVersion())
	object.SetKind(existingObject.GetKind())
	return object
}

// TestResource covers all the typical endpoints for a resource through
// dynamic client calls.
func TestResource(ctx context.Context, f *framework.Framework, tc ResourceTestcaseInterface) {
	// Set up clients.
	gvr := tc.GetGroupVersionResource()
	gv := gvr.GroupVersion()
	resource := gvr.Resource
	resourceClient := f.DynamicClient.Resource(gvr)
	var client dynamic.ResourceInterface
	var resourceType string
	if tc.IsNamespaced() {
		client = resourceClient.Namespace(f.Namespace.Name)
		resourceType = "namespaced resource"
	} else {
		client = resourceClient
		resourceType = "cluster-scoped resource"
	}
	// e.g. `cluster-scoped resource "deviceclasses"`
	// gvr.String() is too long and includes a comma ("resource.k8s.io/v1, Resource=deviceclasses").
	resourceType = fmt.Sprintf("%s %q", resourceType, gvr.Resource)
	config := dynamic.ConfigFor(f.ClientConfig())
	httpClient, err := rest.HTTPClientFor(config)
	framework.ExpectNoError(err, "construct HTTP client")
	restClient, err := rest.UnversionedRESTClientForConfigAndClient(config, httpClient)
	framework.ExpectNoError(err, "construct REST client")

	// All objects get one label added by the test for List and DeleteCollection.
	// The label must get added to all objects returned by ResourceTestcase
	// because the implementation of that interface is unaware of the extra label.
	labelName := "e2e-test.kubernetes.io"
	labelValue := f.UniqueName
	listOptions := metav1.ListOptions{LabelSelector: labelName + "=" + labelValue}
	addLabel := func(obj *unstructured.Unstructured) *unstructured.Unstructured {
		obj = obj.DeepCopy()
		labels := obj.GetLabels()
		if labels == nil {
			labels = make(map[string]string)
		}
		labels[labelName] = labelValue
		obj.SetLabels(labels)
		return obj
	}

	// Prepare for Create, Get and List.
	desiredInitialObject := addLabel(tc.GetInitialObject(f.Namespace.Name))
	if tc.IsNamespaced() {
		desiredInitialObject.SetNamespace(f.Namespace.Name)
	}

	getResource := func(ctx context.Context) (*unstructured.Unstructured, error) {
		return client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{})
	}
	desiredUpdatedObject := tc.GetUpdateSpec(desiredInitialObject.DeepCopy())
	var desiredUpdatedObjectWithStatus *unstructured.Unstructured
	if tc.HasStatus() {
		desiredUpdatedObjectWithStatus = addLabel(tc.GetUpdateStatus(desiredUpdatedObject))
	}

	// Get all resources in the API. The resulting list of resources must include what we are testing.
	ginkgo.By(fmt.Sprintf("Get %s", gv))
	path := "/apis/" + gv.String()
	var api unstructured.Unstructured
	err = restClient.
		Get().
		AbsPath(path).
		Do(ctx).
		Into(&api)
	framework.ExpectNoError(err, "get resource API")
	resources := api.Object["resources"].([]any)
	index := slices.IndexFunc(resources, func(entry any) bool {
		return entry.(map[string]any)["name"].(string) == resource
	})
	if index < 0 {
		framework.Failf("API for %s does not include entry for %s, got:\n%s", gv, resource, format.Object(api, 1))
	}

	// Set up informers, optionally also in the namespace.
	// After each step we check that the informers catch up
	// and what events they received in the meantime.
	// They get stopped through test context cancellation.
	var resourceEvents, namespaceEvents eventRecorder
	resourceInformer := dynamicinformer.NewFilteredDynamicInformer(f.DynamicClient, gvr, "", 0, nil, func(options *metav1.ListOptions) {
		options.LabelSelector = listOptions.LabelSelector
	})
	_, err = resourceInformer.Informer().AddEventHandler(&resourceEvents)
	framework.ExpectNoError(err, "register resource event handler")
	listResource := func(_ context.Context) ([]runtime.Object, error) {
		return resourceInformer.Lister().List(labels.Everything())
	}
	go resourceInformer.Informer().RunWithContext(ctx)
	informersHaveSynced := []cache.InformerSynced{resourceInformer.Informer().HasSynced}
	var namespaceInformer informers.GenericInformer
	var listNamespace func(_ context.Context) ([]runtime.Object, error)
	if tc.IsNamespaced() {
		namespaceInformer = dynamicinformer.NewFilteredDynamicInformer(f.DynamicClient, gvr, f.Namespace.Name, 0, nil, func(options *metav1.ListOptions) {
			options.LabelSelector = listOptions.LabelSelector
		})
		_, err = namespaceInformer.Informer().AddEventHandler(&namespaceEvents)
		framework.ExpectNoError(err, "register namespace event handler")
		listNamespace = func(_ context.Context) ([]runtime.Object, error) {
			return namespaceInformer.Lister().List(labels.Everything())
		}
		go namespaceInformer.Informer().RunWithContext(ctx)
		informersHaveSynced = append(informersHaveSynced, namespaceInformer.Informer().HasSynced)
	}
	if !cache.WaitForNamedCacheSyncWithContext(ctx, informersHaveSynced...) {
		ginkgo.Fail("informers should have synced and didn't")
	}

	// matchObject generates a matcher which checks the result of a list operation
	// against the expected object. Content verification is optional. Without it,
	// only the namespace and name are checked.
	matchObject := func(expectedObject *unstructured.Unstructured) gtypes.GomegaMatcher {
		return &matchObjectList{expectedObject: expectedObject, verifyContent: tc.VerifyContent()}
	}
	gomega.Expect(listResource(ctx)).Should(matchObject(nil), "initial list of resources from informer cache")
	gomega.Expect(resourceEvents.list()).To(gomega.HaveField("Events", gomega.BeEmpty()), "no events from resource informer yet")
	if listNamespace != nil {
		gomega.Expect(listNamespace(ctx)).Should(matchObject(nil), "initial list of namespace from informer cache")
		gomega.Expect(resourceEvents.list()).To(gomega.HaveField("Events", gomega.BeEmpty()), "no events from namespace informer yet")
	}

	// matchEvents generates a matcher which checks the informer event list.
	//
	// The events are expected to involve only the given object.
	// The ResourceVersion must not decrease.
	// The sequence of valid events is given as a regular expression
	// which is applied to the string returned by [EventList.Types].
	matchEvents := func(obj *unstructured.Unstructured, regexp string) gtypes.GomegaMatcher {
		// Verify namespace/name/uid of ids event.
		ids := gomega.HaveField("Events", gomega.Or(gomega.BeEmpty(), gomega.HaveEach(gomega.HaveField("ID()", gomega.Equal(fmt.Sprintf("%s, %s", klog.KObj(obj), obj.GetUID()))))))

		// Match the regexp against the result of Types().
		order := gomega.HaveField("Types()", gomega.MatchRegexp(regexp))

		// Include full object dump, HaveField itself doesn't.
		return framework.GomegaObject(gomega.And(ids, order))
	}

	// ResourceVersion must not decrease. Delete events reset the sequence.
	var resourceRV, namespaceRV string
	nextEvents := func(rv *string, events eventList) {
		ginkgo.GinkgoHelper()

		checkNext := func(obj any) {
			ginkgo.GinkgoHelper()
			if obj == nil {
				return
			}
			if tomb, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tomb.Obj
			}
			metaData, err := meta.Accessor(obj)
			framework.ExpectNoError(err, "access meta data")
			gomega.Expect(metaData).To(apimachineryutils.HaveValidResourceVersion())

			nextRV := metaData.GetResourceVersion()
			if *rv == "" {
				// Nothing to compare yet, initial version.
				*rv = nextRV
				return
			}

			cmpResult, err := resourceversion.CompareResourceVersion(nextRV, *rv)
			framework.ExpectNoError(err, "compare ResourceVersions")
			if cmpResult < 0 {
				framework.Failf("ResourceVersion %s in %s with UID %s is smaller than previous %s, should be equal or larger", nextRV, klog.KObj(metaData), metaData.GetUID(), *rv)
			}
			*rv = nextRV
		}

		for _, e := range events.Events {
			checkNext(e.oldObj)
			checkNext(e.newObj)
		}
	}

	// Verification of the result after each step (= what).
	//
	// Can check the content of an object or be limited to just name/namespace.
	//
	// Also checks the informers. This is a bit redundant after read-only steps,
	// but then it's also fast and thus can be done more often than strictly necessary.
	// Collected informer events get reset, so each verify call must match
	// events since the previous one.
	verify := func(what string, expected, actual *unstructured.Unstructured, haveExpectedEvents gtypes.GomegaMatcher) {
		ginkgo.GinkgoHelper()

		// This captures several different failures before handing them to Ginkgo.
		var failures gomegaFailures

		if expected != nil {
			if tc.VerifyContent() {
				diff := compareObjects(expected, actual)
				failures.Add(fmt.Sprintf("%s: unexpected actual object (- expected, + actual):\n%s", what, diff))
			} else {
				failures.G().Expect(actual.GetName()).Should(gomega.Equal(expected.GetName()), "%s: name in returned object", what)
				failures.G().Expect(actual.GetNamespace()).Should(gomega.Equal(expected.GetNamespace()), "%s: namespace in returned object", what)
			}
			failures.G().Expect(actual).To(apimachineryutils.HaveValidResourceVersion())
		}

		// Abort checking now if there were failures, otherwise we just risk timining out slowly.
		failures.Check()

		failures.G().Eventually(ctx, listResource).Should(matchObject(expected), "list of resources from informer cache after %s", what)
		if haveExpectedEvents != nil {
			// Even if the cache is up-to-date we still need to wait for event delivery.
			failures.G().Eventually(resourceEvents.list).
				WithTimeout(5*time.Second).
				Should(haveExpectedEvents, "list of resource informer events after %s", what)
		}
		nextEvents(&resourceRV, resourceEvents.reset())
		if listNamespace != nil {
			failures.G().Eventually(ctx, listNamespace).Should(matchObject(expected), "list of namespace from informer cache after %s", what)
			if haveExpectedEvents != nil {
				failures.G().Eventually(namespaceEvents.list).
					WithTimeout(5*time.Second).
					Should(haveExpectedEvents, "list of namespace informer events after %s", what)
			}
			nextEvents(&namespaceRV, namespaceEvents.reset())
		}
		failures.Check()
	}

	// Create the initial resource.
	ginkgo.By(fmt.Sprintf("Creating:\n%s", format.Object(desiredInitialObject, 1)))
	existingObject, err := client.Create(ctx, desiredInitialObject, metav1.CreateOptions{FieldValidation: "Strict"})
	framework.ExpectNoError(err, "create initial %s", resourceType)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		// Always clean up.
		err = client.Delete(ctx, desiredInitialObject.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			return
		}
		framework.ExpectNoError(err, "delete %s", resourceType)
		ensureNotFound(ctx, getResource)
	})
	verify("create", desiredInitialObject, existingObject,
		// Initial creation of the object followed by some optional updates by cluster components.
		matchEvents(existingObject, "^add,(update,)*$"),
	)
	createdResourceVersion := existingObject.GetResourceVersion()

	// Get to check for existence.
	ginkgo.By(fmt.Sprintf("Getting %s", klog.KObj(desiredInitialObject)))
	existingObject, err = client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err, "get updated %s", resourceType)
	verify("get", desiredInitialObject, existingObject,
		// Optional updates by cluster components.
		matchEvents(existingObject, "^(update,)*$"),
	)

	// Update the resource. Retry because the existing object might have been updated in the meantime.
	mustGet := false
	gomega.Eventually(ctx, func(ctx context.Context) error {
		if mustGet {
			ginkgo.By(fmt.Sprintf("Getting updated %s", klog.KObj(desiredInitialObject)))
			existingObject, err = getResource(ctx)
			if err != nil {
				return fmt.Errorf("get existing %s: %w", resourceType, err)
			}
		}
		object := tc.GetUpdateSpec(existingObject.DeepCopy())
		ginkgo.By(fmt.Sprintf("Updating:\n%s", format.Object(object, 1)))
		existingObject, err = client.Update(ctx, object, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		mustGet = apierrors.IsConflict(err)
		if mustGet {
			// Retry immediately.
			return fmt.Errorf("update %s: %w", resourceType, err)
		}
		if retry, retryAfter := framework.ShouldRetry(err); retry {
			// Retry with a delay.
			return gomega.TryAgainAfter(retryAfter)
		}
		// Give up, some other error occurred.
		return gomega.StopTrying(fmt.Sprintf("update %s", resourceType)).Wrap(err)
	}).Should(gomega.Succeed())
	verify("update", desiredUpdatedObject, existingObject,
		// At least one update.
		matchEvents(existingObject, "^(update,)+$"),
	)
	updatedResourceVersion := existingObject.GetResourceVersion()
	cmpResult, err := resourceversion.CompareResourceVersion(createdResourceVersion, updatedResourceVersion)
	framework.ExpectNoError(err, "compare ResourceVersion after create against ResourceVersion after update")
	if cmpResult >= 0 {
		framework.Failf("ResourceVersion should have increased during update and didn't (before: %s, after: %s)", createdResourceVersion, updatedResourceVersion)
	}

	// Same for the status. In addition, read the status (same result, but different endpoint+method).
	if tc.HasStatus() {
		mustGet := false
		gomega.Eventually(ctx, func(ctx context.Context) error {
			if mustGet {
				ginkgo.By(fmt.Sprintf("Getting updated %s", klog.KObj(desiredInitialObject)))
				existingObject, err = client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{}, "status")
				if err != nil {
					return fmt.Errorf("get existing %s: %w", resourceType, err)
				}
			}
			object := tc.GetUpdateStatus(existingObject)
			ginkgo.By(fmt.Sprintf("Updating status:\n%s", format.Object(object, 1)))
			existingObject, err = client.Update(ctx, object, metav1.UpdateOptions{}, "status")
			if err == nil {
				return nil
			}
			mustGet = apierrors.IsConflict(err)
			if mustGet {
				// Retry immediately.
				return fmt.Errorf("update %s status: %w", resourceType, err)
			}
			if retry, retryAfter := framework.ShouldRetry(err); retry {
				// Retry with a delay.
				return gomega.TryAgainAfter(retryAfter)
			}
			// Give up, some other error occurred.
			return gomega.StopTrying(fmt.Sprintf("update %s status", resourceType)).Wrap(err)
		}).Should(gomega.Succeed())
		verify("update status", desiredUpdatedObjectWithStatus, existingObject,
			// At least one update.
			matchEvents(existingObject, "^(update,)+$"),
		)

		ginkgo.By(fmt.Sprintf("Getting %s status", klog.KObj(desiredInitialObject)))
		existingObject, err = client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{}, "status")
		framework.ExpectNoError(err, "get updated %s", resourceType)
		verify("get", desiredUpdatedObjectWithStatus, existingObject,
			// Optional updates by cluster components.
			matchEvents(existingObject, "^(update,)*$"),
		)
	}

	// Patch the resource, potentially using multiple different patch types.
	// The result must be the same each time if content verification is enabled.
	for _, patch := range tc.GetPatchSpec() {
		// Delete the resource to start anew.
		ginkgo.By(fmt.Sprintf("Deleting %s", klog.KObj(desiredInitialObject)))
		err = client.Delete(ctx, desiredInitialObject.GetName(), metav1.DeleteOptions{})
		framework.ExpectNoError(err, "delete updated %s", resourceType)
		ensureNotFound(ctx, getResource)
		verify("delete", nil, nil,
			// Optional updates, deletion.
			//
			// We have to verify this here because
			// otherwise we have no guarantee that we see the delete event.
			matchEvents(existingObject, "^(update,)*delete,$"),
		)

		// Recreate for patching.
		ginkgo.By(fmt.Sprintf("Creating again:\n%s", format.Object(desiredInitialObject, 1)))
		existingObject, err = client.Create(ctx, desiredInitialObject, metav1.CreateOptions{})
		framework.ExpectNoError(err, "create %s again", resourceType)
		patchData := patch.GetData(existingObject)

		ginkgo.By(fmt.Sprintf("Patching with %s:\n%s", patch.Type, string(patchData)))
		options := metav1.PatchOptions{FieldValidation: "Strict"}
		switch patch.Type {
		case types.ApplyYAMLPatchType, types.ApplyCBORPatchType:
			options.FieldManager = "test-apply"
			options.Force = ptr.To(true)
		}
		existingObject, err = client.Patch(ctx, desiredInitialObject.GetName(), patch.Type, patchData, options)
		framework.ExpectNoError(err, "patch %s %s", patch.Type, resourceType)
		verify(fmt.Sprintf("patch %s", patch.Type), desiredUpdatedObject, existingObject,
			// Recreation and then at least one update.
			matchEvents(existingObject, "^add,(update,)+$"),
		)
	}

	// Same for status. The patches apply on top of the updated object.
	for _, patch := range tc.GetPatchStatus(existingObject) {
		// Delete the resource to start anew.
		ginkgo.By(fmt.Sprintf("Deleting %s", klog.KObj(desiredInitialObject)))
		err = client.Delete(ctx, desiredInitialObject.GetName(), metav1.DeleteOptions{})
		framework.ExpectNoError(err, "delete updated %s", resourceType)
		ensureNotFound(ctx, getResource)
		verify("delete", nil, nil,
			// Optional updates, deletion.
			//
			// We have to verify this here because
			// otherwise we have no guarantee that we see the delete event.
			matchEvents(existingObject, "^(update,)*delete,$"),
		)

		// Recreate for patching.
		ginkgo.By(fmt.Sprintf("Creating again:\n%s", format.Object(desiredInitialObject, 1)))
		existingObject, err = client.Create(ctx, desiredInitialObject, metav1.CreateOptions{})
		framework.ExpectNoError(err, "create %s again", resourceType)

		// Update again.
		mustGet := false
		gomega.Eventually(ctx, func(ctx context.Context) error {
			if mustGet {
				ginkgo.By(fmt.Sprintf("Getting updated %s", klog.KObj(desiredInitialObject)))
				existingObject, err = getResource(ctx)
				if err != nil {
					return fmt.Errorf("get existing %s: %w", resourceType, err)
				}
			}
			object := tc.GetUpdateSpec(existingObject.DeepCopy())
			ginkgo.By(fmt.Sprintf("Updating:\n%s", format.Object(object, 1)))
			existingObject, err = client.Update(ctx, object, metav1.UpdateOptions{})
			if err == nil {
				return nil
			}
			mustGet = apierrors.IsConflict(err)
			if mustGet {
				// Retry immediately.
				return fmt.Errorf("update %s: %w", resourceType, err)
			}
			if retry, retryAfter := framework.ShouldRetry(err); retry {
				// Retry with a delay.
				return gomega.TryAgainAfter(retryAfter)
			}
			// Give up, some other error occurred.
			return gomega.StopTrying(fmt.Sprintf("update %s", resourceType)).Wrap(err)
		}).Should(gomega.Succeed())
		patchData := patch.GetData(existingObject)

		ginkgo.By(fmt.Sprintf("Patching status with %s:\n%s", patch.Type, string(patchData)))
		options := metav1.PatchOptions{FieldValidation: "Strict"}
		switch patch.Type {
		case types.ApplyYAMLPatchType, types.ApplyCBORPatchType:
			options.FieldManager = "test-apply"
			options.Force = ptr.To(true)
		}
		existingObject, err = client.Patch(ctx, desiredInitialObject.GetName(), patch.Type, patchData, options, "status")
		framework.ExpectNoError(err, "patch %s %s status", patch.Type, resourceType)
		verify(fmt.Sprintf("patch %s status", patch.Type), desiredUpdatedObject, existingObject,
			// Recreation and then at least one update.
			matchEvents(existingObject, "^add,(update,)+$"),
		)
	}

	// Use the label as selector in List and DeleteCollection calls.
	ginkgo.By(fmt.Sprintf("Listing %s collection with label selector %s", gvr, listOptions.LabelSelector))
	items, err := client.List(ctx, listOptions)
	framework.ExpectNoError(err, "list %s", resourceType)
	gomega.Expect(items.Items).Should(gomega.HaveLen(1), "Should have listed exactly the test resource.")
	verify("list", desiredUpdatedObject, &items.Items[0],
		// Optional updates by cluster components.
		matchEvents(existingObject, "^(update,)*$"),
	)

	if tc.IsNamespaced() {
		ginkgo.By(fmt.Sprintf("Listing %s without namespace and with label selector %s", gvr, listOptions.LabelSelector))
		items, err := resourceClient.List(ctx, listOptions)
		framework.ExpectNoError(err, "list %s in all namespaces", resourceType)
		gomega.Expect(items.Items).Should(gomega.HaveLen(1), "Should have listed exactly the test resource in all namespaces.")
		verify("list all namespaces", desiredUpdatedObject, &items.Items[0],
			// Optional updates by cluster components.
			matchEvents(existingObject, "^(update,)*$"),
		)
	}

	ginkgo.By(fmt.Sprintf("Deleting %s collection with label selector %s", gvr, listOptions.LabelSelector))
	err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, listOptions)
	framework.ExpectNoError(err, "delete collection of %s", resourceType)
	ensureNotFound(ctx, getResource)
	verify("list", nil, nil,
		// Optional updates by cluster components, then deletion.
		matchEvents(existingObject, "^(update,)*delete,$"),
	)
}

// ensureNotFound ensures that the error returned by the get function is NotFound.
// This can be called after deleting an object to ensure that it really got removed
// and not just marked for deletion with a DeletionTimestamp. Deletion is not
// necessarily instantaneous e.g. because some cluster component might add its
// own finalizer.
func ensureNotFound(ctx context.Context, get func(context.Context) (*unstructured.Unstructured, error)) {
	ginkgo.GinkgoHelper()
	ginkgo.By("Checking for existence")
	gomega.Eventually(ctx, func(ctx context.Context) error {
		obj, err := framework.HandleRetry(get)(ctx)
		switch {
		case apierrors.IsNotFound(err):
			return nil
		case err != nil:
			return fmt.Errorf("unexpected error after GET: %w", err)
		default:
			return fmt.Errorf("resource not removed yet:\n%s", format.Object(obj, 1))
		}
	}).WithTimeout(30 * time.Second /* From prior conformance tests, e.g. https://github.com/kubernetes/kubernetes/blame/be361a18dda0f2fab1f5e25f8067a9ed43fc3b89/test/e2e/storage/storageclass.go#L152 */).
		Should(gomega.Succeed())
}
