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
	"bytes"
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	"sigs.k8s.io/yaml"
)

// ResourceTestcase describes how to test one particular API endpoint
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
// See [YAMLResourceTestcase] for an implementation of this interface
// where test data is provided as YAML fragments.
type ResourceTestcase interface {
	// GetGroupVersionResource returns the API group, version, and resource (plural form, lower case).
	GetGroupVersionResource() schema.GroupVersionResource

	// IsNamespaced() defines whether the object must be created inside a namespace.
	IsNamespaced() bool

	// HasStatus() defines whether the resource has a "status" sub-resource.
	//
	// Other sub-resources are not supported by this common test code.
	HasStatus() bool

	// GetInitialObject returns the data which is going to be used in a Create call.
	//
	// Must return a new instance each time that it is called.
	// The result must include the labels supplied by the caller.
	// If the resource is namespaced, then the namespace must be set.
	// Otherwise the namespace can be used to create a name which
	// does not conflict with other objects because it is unique
	// while the test runs.
	GetInitialObject(namespace string, additionalLabels map[string]string) *unstructured.Unstructured

	// UpdateSpec modifies an existing object.
	//
	// Ideally it should  change the spec (hence the name).
	// If that is impossible, then adding some label is also okay.
	// The goal is to add some fields that can be checked for
	// after an Update.
	//
	// Must not modify the existing object.
	UpdateSpec(object *unstructured.Unstructured) *unstructured.Unstructured

	// UpdateStatus modifies the status of an existing object.
	//
	// Must not modify the existing object.
	UpdateStatus(object *unstructured.Unstructured) *unstructured.Unstructured

	// GetPatchSpec returns the patch data and type for an existing object (= UID is set).
	//
	// This must cause the same change as UpdateSpec. Validation of the patch result uses
	// the UpdateSpec result as reference.
	//
	// There's no retry loop because of conflicts, so the patch should not include
	// a check of the ResourceVersion. Checking the UID in the patch is encouraged to prevent
	// patching a replaced resource.
	GetPatchSpec(object *unstructured.Unstructured) (patchData string, patchType types.PatchType)

	// GetPatchStatus is like GetPatchSpec for the status.
	GetPatchStatus(object *unstructured.Unstructured) (patchData string, patchType types.PatchType)
}

type YAMLResourceTestcase struct {
	// GroupResourceVersion identifies the API group, version and resource (plural form, lower case)
	// within that API which is to be tested.
	GVR schema.GroupVersionResource

	// Namespaced must be true if the resource must be created in a
	// namespace, false if it is cluster-scoped.
	//
	// Namespaced resources get created in the test namespace.
	//
	// The name of cluster-scoped resources gets extended with
	// `-<test namespace name>` to make it unique.
	Namespaced bool

	// InitialSpec must contain the JSON or YAML representation
	// of a valid resource, without a status.
	//
	// Must include name, apiVersion and kind.
	InitialSpec string

	// PatchSpec must contain modified fields for InitialSpec for
	// use in a strategic merge patch.
	//
	// Must include apiVersion and kind. May contain the name,
	// but that's not required.
	PatchSpec string

	// PatchStatus may contains some status fields for use in a strategic
	// merge patch.
	//
	// If empty, the status sub-resource is not getting tested.
	// Otherwise it must include apiVersion and kind.
	// May contain the name, but that's not required.
	PatchStatus string
}

var _ ResourceTestcase = &YAMLResourceTestcase{}

func (tc *YAMLResourceTestcase) GetGroupVersionResource() schema.GroupVersionResource {
	return tc.GVR
}

func (tc *YAMLResourceTestcase) IsNamespaced() bool {
	return tc.Namespaced
}

func (tc *YAMLResourceTestcase) HasStatus() bool {
	return tc.PatchStatus != ""
}

func (tc *YAMLResourceTestcase) GetInitialObject(namespace string, additionalLabels map[string]string) *unstructured.Unstructured {
	var initialObject *unstructured.Unstructured
	err := yaml.Unmarshal([]byte(tc.InitialSpec), &initialObject)
	framework.ExpectNoError(err, "decode InitialSpec")

	labels := initialObject.GetLabels()
	if labels == nil {
		labels = make(map[string]string)
	}
	for k, v := range additionalLabels {
		labels[k] = v
	}
	initialObject.SetLabels(labels)

	if tc.Namespaced {
		initialObject.SetNamespace(namespace)
	} else {
		initialObject.SetName(initialObject.GetName() + "-" + namespace)
	}

	return initialObject
}

func (tc *YAMLResourceTestcase) UpdateSpec(object *unstructured.Unstructured) *unstructured.Unstructured {
	patchObject := createObject("PatchSpec", tc.PatchSpec, object)
	updateObject := mergeObjects(object, patchObject)
	return updateObject
}

func (tc *YAMLResourceTestcase) UpdateStatus(object *unstructured.Unstructured) *unstructured.Unstructured {
	patchObject := createObject("PatchStatus", tc.PatchStatus, object)
	updateObject := mergeObjects(object, patchObject)
	return updateObject
}

func (tc *YAMLResourceTestcase) GetPatchSpec(object *unstructured.Unstructured) (string, types.PatchType) {
	patchObject := createObject("PatchSpec", tc.PatchSpec, object)

	jsonData, err := patchObject.MarshalJSON()
	framework.ExpectNoError(err, "re-encode spec patch as JSON")

	return string(jsonData), types.StrategicMergePatchType
}

func (tc *YAMLResourceTestcase) GetPatchStatus(object *unstructured.Unstructured) (string, types.PatchType) {
	patchObject := createObject("PatchStatus", tc.PatchStatus, object)

	jsonData, err := patchObject.MarshalJSON()
	framework.ExpectNoError(err, "re-encode status patch as JSON")

	return string(jsonData), types.StrategicMergePatchType
}

// createObject parses YAML data and then copies namespace/name/uid from the existing object.
func createObject(what string, data string, existingObject *unstructured.Unstructured) *unstructured.Unstructured {
	var object *unstructured.Unstructured
	err := yaml.Unmarshal([]byte(data), &object)
	if err != nil || object == nil {
		framework.Failf("Unable to decode %s, got object %v, err %v\n\nData:\n%s", what, object, err, data)
	}

	object.SetNamespace(existingObject.GetNamespace())
	object.SetName(existingObject.GetName())
	object.SetUID(existingObject.GetUID())

	return object
}

// TestResource covers all the typical endpoints for a resource through
// dynamic client calls.
func TestResource(ctx context.Context, f *framework.Framework, tc ResourceTestcase) {
	// Set up clients.
	gvr := tc.GetGroupVersionResource()
	gv := gvr.GroupVersion()
	resource := gvr.Resource
	resourceClient := f.DynamicClient.Resource(gvr)
	var client dynamic.ResourceInterface
	if tc.IsNamespaced() {
		client = resourceClient.Namespace(f.Namespace.Name)
	} else {
		client = resourceClient
	}
	config := dynamic.ConfigFor(f.ClientConfig())
	httpClient, err := rest.HTTPClientFor(config)
	framework.ExpectNoError(err, "construct HTTP client")
	restClient, err := rest.UnversionedRESTClientForConfigAndClient(config, httpClient)
	framework.ExpectNoError(err, "construct REST client")

	// Prepare for Create, Get and List.
	labelName := "e2e-test.kubernetes.io"
	labelValue := f.UniqueName
	listOptions := metav1.ListOptions{LabelSelector: labelName + "=" + labelValue}
	desiredInitialObject := tc.GetInitialObject(f.Namespace.Name, map[string]string{labelName: labelValue})
	getResource := func(ctx context.Context) (*unstructured.Unstructured, error) {
		return client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{})
	}
	desiredUpdatedObject := tc.UpdateSpec(desiredInitialObject)
	var desiredUpdatedObjectWithStatus *unstructured.Unstructured
	if tc.HasStatus() {
		desiredUpdatedObjectWithStatus = tc.UpdateStatus(desiredUpdatedObject)
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
		framework.Failf("API for %s does not include entry for %s, got:\n%s", gv, resource, marshalYAML(&api, 1))
	}

	// Create the initial resource.
	ginkgo.By(fmt.Sprintf("Creating:\n%s", marshalYAML(desiredInitialObject, 1)))
	existingObject, err := client.Create(ctx, desiredInitialObject, metav1.CreateOptions{FieldValidation: "Strict"})
	framework.ExpectNoError(err, "create initial resource")
	ginkgo.DeferCleanup(func(ctx context.Context) {
		// Always clean up.
		err = client.Delete(ctx, desiredInitialObject.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			return
		}
		framework.ExpectNoError(err, "delete resource")
		ensureNotFound(ctx, getResource)
	})
	verifyObject("create", desiredInitialObject, existingObject)

	// Get to check for existence.
	ginkgo.By(fmt.Sprintf("Getting %s", klog.KObj(desiredInitialObject)))
	existingObject, err = client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err, "get updated resource")
	verifyObject("get", desiredInitialObject, existingObject)

	// Update the resource. Retry because the existing object might have been updated in the meantime.
	mustGet := false
	gomega.Eventually(ctx, func(ctx context.Context) error {
		if mustGet {
			ginkgo.By(fmt.Sprintf("Getting updated %s", klog.KObj(desiredInitialObject)))
			existingObject, err = getResource(ctx)
			if err != nil {
				return fmt.Errorf("get existing resource: %w", err)
			}
		}
		object := tc.UpdateSpec(existingObject)
		ginkgo.By(fmt.Sprintf("Updating:\n%s", marshalYAML(object, 1)))
		existingObject, err = client.Update(ctx, object, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		mustGet = apierrors.IsConflict(err)
		if mustGet {
			// Retry immediately.
			return fmt.Errorf("update resource: %w", err)
		}
		if retry, retryAfter := framework.ShouldRetry(err); retry {
			// Retry with a delay.
			return gomega.TryAgainAfter(retryAfter)
		}
		// Give up, some other error occurred.
		return gomega.StopTrying("update resource").Wrap(err)
	}).Should(gomega.Succeed())
	verifyObject("update", desiredUpdatedObject, existingObject)

	// Same for the status. In addition, read the status (same result, but different endpoint+method).
	if tc.HasStatus() {
		mustGet := false
		gomega.Eventually(ctx, func(ctx context.Context) error {
			if mustGet {
				ginkgo.By(fmt.Sprintf("Getting updated %s", klog.KObj(desiredInitialObject)))
				existingObject, err = client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("get existing resource: %w", err)
				}
			}
			object := tc.UpdateStatus(existingObject)
			ginkgo.By(fmt.Sprintf("Updating status:\n%s", marshalYAML(object, 1)))
			existingObject, err = client.Update(ctx, object, metav1.UpdateOptions{}, "status")
			if err == nil {
				return nil
			}
			mustGet = apierrors.IsConflict(err)
			if mustGet {
				// Retry immediately.
				return fmt.Errorf("update resource status: %w", err)
			}
			if retry, retryAfter := framework.ShouldRetry(err); retry {
				// Retry with a delay.
				return gomega.TryAgainAfter(retryAfter)
			}
			// Give up, some other error occurred.
			return gomega.StopTrying("update resource status").Wrap(err)
		}).Should(gomega.Succeed())
		verifyObject("update status", desiredUpdatedObjectWithStatus, existingObject)

		ginkgo.By(fmt.Sprintf("Getting %s status", klog.KObj(desiredInitialObject)))
		existingObject, err = client.Get(ctx, desiredInitialObject.GetName(), metav1.GetOptions{}, "status")
		framework.ExpectNoError(err, "get updated resource")
		verifyObject("get", desiredUpdatedObjectWithStatus, existingObject)
	}

	// Delete the resource.
	ginkgo.By(fmt.Sprintf("Deleting %s", klog.KObj(desiredInitialObject)))
	err = client.Delete(ctx, desiredInitialObject.GetName(), metav1.DeleteOptions{})
	framework.ExpectNoError(err, "delete updated resource")
	ensureNotFound(ctx, getResource)

	// Recreate for patching and label-filtered List/DeleteCollection.
	ginkgo.By(fmt.Sprintf("Creating again with label:\n%s", marshalYAML(desiredInitialObject, 1)))
	existingObject, err = client.Create(ctx, desiredInitialObject, metav1.CreateOptions{})
	framework.ExpectNoError(err, "create resource again")

	// Patch the resource.
	patchData, patchType := tc.GetPatchSpec(existingObject)
	ginkgo.By(fmt.Sprintf("Patching:\n%s", patchData))
	existingObject, err = client.Patch(ctx, desiredInitialObject.GetName(), patchType, []byte(patchData), metav1.PatchOptions{FieldValidation: "Strict"})
	framework.ExpectNoError(err, "patch resource")
	verifyObject("patch", desiredUpdatedObject, existingObject)

	// Same for status, using the same patch.
	if tc.HasStatus() {
		patchData, patchType := tc.GetPatchStatus(existingObject)

		ginkgo.By(fmt.Sprintf("Patching status:\n%s", patchData))
		existingObject, err = client.Patch(ctx, desiredInitialObject.GetName(), patchType, []byte(patchData), metav1.PatchOptions{FieldValidation: "Strict"}, "status")
		framework.ExpectNoError(err, "patch resource status")
		verifyObject("patch status", desiredUpdatedObjectWithStatus, existingObject)
	}

	// Use the label as selector in List and DeleteCollection calls.
	ginkgo.By(fmt.Sprintf("Listing %s collection with label selector %s", gvr, listOptions.LabelSelector))
	items, err := client.List(ctx, listOptions)
	framework.ExpectNoError(err, "list resources")
	gomega.Expect(items.Items).Should(gomega.HaveLen(1), "Should have listed exactly the test resource.")
	verifyObject("list", desiredUpdatedObject, &items.Items[0])

	if tc.IsNamespaced() {
		ginkgo.By(fmt.Sprintf("Listing %s without namespace and with label selector %s", gvr, listOptions.LabelSelector))
		items, err := resourceClient.List(ctx, listOptions)
		framework.ExpectNoError(err, "list resources all namespaces")
		gomega.Expect(items.Items).Should(gomega.HaveLen(1), "Should have listed exactly the test resource in all namespaces.")
		verifyObject("list all namespaces", desiredUpdatedObject, &items.Items[0])
	}

	ginkgo.By(fmt.Sprintf("Deleting %s collection with label selector %s", gvr, listOptions.LabelSelector))
	err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, listOptions)
	framework.ExpectNoError(err, "delete collection of resource")
	ensureNotFound(ctx, getResource)
}

// ensureNotFound ensures that the error is NotFound.
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
			return fmt.Errorf("resource not removed yet:\n%s", marshalYAML(obj, 1))
		}
	}).WithTimeout(30 * time.Second /* From prior conformance tests, e.g. https://github.com/kubernetes/kubernetes/blame/be361a18dda0f2fab1f5e25f8067a9ed43fc3b89/test/e2e/storage/storageclass.go#L152 */).
		Should(gomega.Succeed())
}

// marshalYAML produces multi-line YAML with each line indented by a certain
// number of spaces (currently three) multiplied by the requested indention
// level.
//
// Gomega could be configured to log unstructured.Unstructured as YAML,
// but right now it isn't, so we have to that ourselves instead of
// using format.Object.
func marshalYAML(obj any, indent int) string {
	ginkgo.GinkgoHelper()
	data, err := yaml.Marshal(obj)
	framework.ExpectNoError(err, "marshal as YAML")
	spaces := strings.Repeat("   ", indent)
	data = bytes.ReplaceAll(data, []byte("\n"), []byte("\n"+spaces))
	return spaces + string(data)
}

// mergeObjects adds all fields from the second object to the first one
// and returns the new object. Existing leaf fields are overwritten,
// existing maps are updated.
//
// The input objects are not modified. The return object is completely
// independent (no shared maps or slices).
func mergeObjects(base, update *unstructured.Unstructured) *unstructured.Unstructured {
	merged := base.DeepCopy()
	if merged.Object == nil {
		merged.Object = make(map[string]any)
	}
	updateMap(merged.Object, update.Object)

	// Some fields may have been copied from the update object, so slices and maps
	// are shared. We need another DeepCopy to create a completely separate object.
	return merged.DeepCopy()
}

func updateMap(to, from map[string]any) {
	for k, v := range from {
		old := to[k]
		if old, ok := old.(map[string]any); ok {
			if v, ok := v.(map[string]any); ok {
				// Merge into existing map field.
				updateMap(old, v)
				continue
			}
		}
		// Add or overwrite field.
		to[k] = v
	}
}

// verifyObject checks that all expected fields are set as expected.
// The actual object may have additional fields, their values are ignored.
func verifyObject(what string, expected, actual *unstructured.Unstructured) {
	ginkgo.GinkgoHelper()
	diff := compareObjects(expected, actual)
	if diff != "" {
		framework.Failf("%s: unexpected actual object (- expected, + actual):\n%s", what, diff)
	}
}

func compareObjects(expected, actual *unstructured.Unstructured) string {
	diff := cmp.Diff(expected.Object, actual.Object,
		// Fields which are not in the expected object can be ignored.
		// Only existing fields need to be compared.
		//
		// A maybe (?) simpler approach would be to trim the actual object,
		// then compare with go-cmp. The advantage of telling go-cmp to
		// ignore fields is that they show up as truncated ("...") in the diff,
		// which is a bit more correct.
		cmp.FilterPath(func(path cmp.Path) bool {
			return fieldIsMissing(expected.Object, path)
		}, cmp.Ignore()),
	)
	return diff
}

// fieldIsMissing returns true if the field identified by the path is not
// present in the object. It works by recursively descending along the
// path and checking the corresponding content of the object along the way.
func fieldIsMissing(obj map[string]any, path cmp.Path) bool {
	// First entry is a NOP.
	missing := fieldIsMissingStep(obj, path[1:])
	// Uncomment for debugging...
	// fmt.Printf("fieldIsMissing: %s %v\n", path.GoString(), missing)
	return missing
}

func fieldIsMissingStep(value any, path []cmp.PathStep) bool {
	if len(path) == 0 {
		// Done, full path was checked.
		return false
	}
	// We only need to descend for certain lookup steps,
	// everything else is treated as "not missing" and thus
	// gets compared.
	switch pathElement := path[0].(type) {
	case cmp.MapIndex:
		key := pathElement.Key().String()
		value, ok := value.(map[string]any)
		if !ok {
			// Type mismatch.
			return false
		}
		entry, found := value[key]
		if !found {
			return true
		}
		return fieldIsMissingStep(entry, path[1:])
	case cmp.SliceIndex:
		key := pathElement.Key()
		value, ok := value.([]any)
		if !ok {
			// Type mismatch.
			return false
		}
		if key < 0 {
			// Not sure why go-cmp uses a negative index, so let's compare it.
			return false
		}
		if key >= len(value) {
			// Slice is smaller -> missing entry.
			return true
		}
		entry := value[key]
		return fieldIsMissingStep(entry, path[1:])
	case cmp.TypeAssertion:
		// Actual value type will be checked when needed,
		// skip the assertion here.
		return fieldIsMissingStep(value, path[1:])
	default:
		return false
	}
}
