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

package deletion

import (
	"fmt"
	"sort"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

func NewNamespacedResourcesDeleter(nsClient internalversion.NamespaceInterface,
	clientPool dynamic.ClientPool, opCache OperationNotSupportedCache,
	gvrs []unversioned.GroupVersionResource, finalizerToken api.FinalizerName,
	deleteNamespaceWhenDone bool) *NamespacedResourcesDeleter {
	return &NamespacedResourcesDeleter{
		nsClient:                nsClient,
		clientPool:              clientPool,
		opCache:                 opCache,
		groupVersionResources:   gvrs,
		finalizerToken:          finalizerToken,
		deleteNamespaceWhenDone: deleteNamespaceWhenDone,
	}
}

// NamespacedResourcesDeleter is used to delete all resources in a given namespace.
type NamespacedResourcesDeleter struct {
	// Client to manipulate the namespace.
	nsClient internalversion.NamespaceInterface
	// Dynamic client to list and delete all namespaced resources.
	clientPool dynamic.ClientPool
	// Cache of what operations are not supported on each group version resource.
	opCache OperationNotSupportedCache
	// Group version resources that should be operated on.
	groupVersionResources []unversioned.GroupVersionResource
	// The finalizer token that should be removed from the namespace
	// when all resources in that namespace have been deleted.
	finalizerToken api.FinalizerName
	// Also delete the namespace when all resources in the namespace have been deleted.
	deleteNamespaceWhenDone bool
}

// Delete deletes all resources in the given namespace.
// Before deleting resources:
// * It ensures that deletion timestamp is set on the
//   namespace (does nothing if deletion timestamp is missing).
// * Verifies that the namespace is in the "terminating" phase
//   (updates the namespace phase if it is not yet marked terminating)
// After deleting the resources:
// * It removes finalizer token from the given namespace.
// * Deletes the namespace if deleteNamespaceWhenDone is true.
//
// Returns an error if any of those steps fail.
// Returns ResourcesRemainingError if it deleted some resources but needs
// to wait for them to go away.
// Caller is expected to keep calling this until it succeeds.
func (d *NamespacedResourcesDeleter) Delete(nsName string) error {
	// Multiple controllers may edit a namespace during termination
	// first get the latest state of the namespace before proceeding
	// if the namespace was deleted already, don't do anything
	namespace, err := d.nsClient.Get(nsName)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if namespace.DeletionTimestamp == nil {
		return nil
	}

	glog.V(5).Infof("namespace controller - syncNamespace - namespace: %s, finalizerToken: %s", namespace.Name, d.finalizerToken)

	// ensure that the status is up to date on the namespace
	// if we get a not found error, we assume the namespace is truly gone
	namespace, err = d.retryOnConflictError(namespace, d.updateNamespaceStatusFunc)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// the latest view of the namespace asserts that namespace is no longer deleting..
	if namespace.DeletionTimestamp.IsZero() {
		return nil
	}

	// Delete the namespace if it is already finalized.
	if d.deleteNamespaceWhenDone && finalized(namespace) {
		return d.deleteNamespace(namespace)
	}

	// there may still be content for us to remove
	estimate, err := d.deleteAllContent(namespace.Name, *namespace.DeletionTimestamp)
	if err != nil {
		return err
	}
	if estimate > 0 {
		return &ResourcesRemainingError{estimate}
	}

	// we have removed content, so mark it finalized by us
	namespace, err = d.retryOnConflictError(namespace, d.finalizeNamespace)
	if err != nil {
		// in normal practice, this should not be possible, but if a deployment is running
		// two controllers to do namespace deletion that share a common finalizer token it's
		// possible that a not found could occur since the other controller would have finished the delete.
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// Check if we can delete now.
	if d.deleteNamespaceWhenDone && finalized(namespace) {
		return d.deleteNamespace(namespace)
	}
	return nil
}

// Deletes the given namespace.
func (d *NamespacedResourcesDeleter) deleteNamespace(namespace *api.Namespace) error {
	var opts *api.DeleteOptions
	uid := namespace.UID
	if len(uid) > 0 {
		opts = &api.DeleteOptions{Preconditions: &api.Preconditions{UID: &uid}}
	}
	err := d.nsClient.Delete(namespace.Name, opts)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	return nil
}

// ResourcesRemainingError is used to inform the caller that all resources are not yet fully removed from the namespace.
type ResourcesRemainingError struct {
	Estimate int64
}

func (e *ResourcesRemainingError) Error() string {
	return fmt.Sprintf("some content remains in the namespace, estimate %d seconds before it is removed", e.Estimate)
}

// operation is used for caching if an operation is supported on a dynamic client.
type Operation string

const (
	OperationDeleteCollection Operation = "deleteCollection"
	OperationList             Operation = "list"
	// assume a default estimate for finalizers to complete when found on items pending deletion.
	finalizerEstimateSeconds int64 = int64(15)
)

// OperationKey is an entry in a cache.
type OperationKey struct {
	Operation Operation
	Gvr       unversioned.GroupVersionResource
}

// OperationNotSupportedCache is a simple cache to remember if an operation is not supported for a resource.
// if the OperationKey maps to true, it means the operation is not supported.
type OperationNotSupportedCache map[OperationKey]bool

// isSupported returns true if the operation is supported
func (o OperationNotSupportedCache) isSupported(key OperationKey) bool {
	return !o[key]
}

// updateNamespaceFunc is a function that makes an update to a namespace
type updateNamespaceFunc func(namespace *api.Namespace) (*api.Namespace, error)

// retryOnConflictError retries the specified fn if there was a conflict error
// it will return an error if the UID for an object changes across retry operations.
// TODO RetryOnConflict should be a generic concept in client code
func (d *NamespacedResourcesDeleter) retryOnConflictError(namespace *api.Namespace, fn updateNamespaceFunc) (result *api.Namespace, err error) {
	latestNamespace := namespace
	for {
		result, err = fn(latestNamespace)
		if err == nil {
			return result, nil
		}
		if !errors.IsConflict(err) {
			return nil, err
		}
		prevNamespace := latestNamespace
		latestNamespace, err = d.nsClient.Get(latestNamespace.Name)
		if err != nil {
			return nil, err
		}
		if prevNamespace.UID != latestNamespace.UID {
			return nil, fmt.Errorf("namespace uid has changed across retries")
		}
	}
}

// updateNamespaceStatusFunc will verify that the status of the namespace is correct
func (d *NamespacedResourcesDeleter) updateNamespaceStatusFunc(
	namespace *api.Namespace) (*api.Namespace, error) {
	if namespace.DeletionTimestamp.IsZero() || namespace.Status.Phase == api.NamespaceTerminating {
		return namespace, nil
	}
	newNamespace := api.Namespace{}
	newNamespace.ObjectMeta = namespace.ObjectMeta
	newNamespace.Status = namespace.Status
	newNamespace.Status.Phase = api.NamespaceTerminating
	return d.nsClient.UpdateStatus(&newNamespace)
}

// finalized returns true if the namespace.Spec.Finalizers is an empty list
func finalized(namespace *api.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalizeNamespace removes the specified finalizerToken and finalizes the namespace
func (d *NamespacedResourcesDeleter) finalizeNamespace(namespace *api.Namespace) (*api.Namespace, error) {
	namespaceFinalize := api.Namespace{}
	namespaceFinalize.ObjectMeta = namespace.ObjectMeta
	namespaceFinalize.Spec = namespace.Spec
	finalizerSet := sets.NewString()
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != d.finalizerToken {
			finalizerSet.Insert(string(namespace.Spec.Finalizers[i]))
		}
	}
	namespaceFinalize.Spec.Finalizers = make([]api.FinalizerName, 0, len(finalizerSet))
	for _, value := range finalizerSet.List() {
		namespaceFinalize.Spec.Finalizers = append(namespaceFinalize.Spec.Finalizers, api.FinalizerName(value))
	}
	namespace, err := d.nsClient.Finalize(&namespaceFinalize)
	if err != nil {
		// it was removed already, so life is good
		if errors.IsNotFound(err) {
			return namespace, nil
		}
	}
	return namespace, err
}

// deleteCollection is a helper function that will delete the collection of resources
// it returns true if the operation was supported on the server.
// it returns an error if the operation was supported on the server but was unable to complete.
func (d *NamespacedResourcesDeleter) deleteCollection(
	dynamicClient *dynamic.Client, gvr unversioned.GroupVersionResource,
	namespace string) (bool, error) {
	glog.V(5).Infof("namespace controller - deleteCollection - namespace: %s, gvr: %v", namespace, gvr)

	key := OperationKey{Operation: OperationDeleteCollection, Gvr: gvr}
	if !d.opCache.isSupported(key) {
		glog.V(5).Infof("namespace controller - deleteCollection ignored since not supported - namespace: %s, gvr: %v", namespace, gvr)
		return false, nil
	}

	apiResource := unversioned.APIResource{Name: gvr.Resource, Namespaced: true}

	// namespace controller does not want the garbage collector to insert the orphan finalizer since it calls
	// resource deletions generically.  it will ensure all resources in the namespace are purged prior to releasing
	// namespace itself.
	orphanDependents := false
	err := dynamicClient.Resource(&apiResource, namespace).DeleteCollection(&v1.DeleteOptions{OrphanDependents: &orphanDependents}, &v1.ListOptions{})

	if err == nil {
		return true, nil
	}

	// this is strange, but we need to special case for both MethodNotSupported and NotFound errors
	// TODO: https://github.com/kubernetes/kubernetes/issues/22413
	// we have a resource returned in the discovery API that supports no top-level verbs:
	//  /apis/extensions/v1beta1/namespaces/default/replicationcontrollers
	// when working with this resource type, we will get a literal not found error rather than expected method not supported
	// remember next time that this resource does not support delete collection...
	if errors.IsMethodNotSupported(err) || errors.IsNotFound(err) {
		glog.V(5).Infof("namespace controller - deleteCollection not supported - namespace: %s, gvr: %v", namespace, gvr)
		d.opCache[key] = true
		return false, nil
	}

	glog.V(5).Infof("namespace controller - deleteCollection unexpected error - namespace: %s, gvr: %v, error: %v", namespace, gvr, err)
	return true, err
}

// listCollection will list the items in the specified namespace
// it returns the following:
//  the list of items in the collection (if found)
//  a boolean if the operation is supported
//  an error if the operation is supported but could not be completed.
func (d *NamespacedResourcesDeleter) listCollection(
	dynamicClient *dynamic.Client, gvr unversioned.GroupVersionResource, namespace string) (*runtime.UnstructuredList, bool, error) {
	glog.V(5).Infof("namespace controller - listCollection - namespace: %s, gvr: %v", namespace, gvr)

	key := OperationKey{Operation: OperationList, Gvr: gvr}
	if !d.opCache.isSupported(key) {
		glog.V(5).Infof("namespace controller - listCollection ignored since not supported - namespace: %s, gvr: %v", namespace, gvr)
		return nil, false, nil
	}

	apiResource := unversioned.APIResource{Name: gvr.Resource, Namespaced: true}
	obj, err := dynamicClient.Resource(&apiResource, namespace).List(&v1.ListOptions{})
	if err == nil {
		unstructuredList, ok := obj.(*runtime.UnstructuredList)
		if !ok {
			return nil, false, fmt.Errorf("resource: %s, expected *runtime.UnstructuredList, got %#v", apiResource.Name, obj)
		}
		return unstructuredList, true, nil
	}

	// this is strange, but we need to special case for both MethodNotSupported and NotFound errors
	// TODO: https://github.com/kubernetes/kubernetes/issues/22413
	// we have a resource returned in the discovery API that supports no top-level verbs:
	//  /apis/extensions/v1beta1/namespaces/default/replicationcontrollers
	// when working with this resource type, we will get a literal not found error rather than expected method not supported
	// remember next time that this resource does not support delete collection...
	if errors.IsMethodNotSupported(err) || errors.IsNotFound(err) {
		glog.V(5).Infof("namespace controller - listCollection not supported - namespace: %s, gvr: %v", namespace, gvr)
		d.opCache[key] = true
		return nil, false, nil
	}

	return nil, true, err
}

// deleteEachItem is a helper function that will list the collection of resources and delete each item 1 by 1.
func (d *NamespacedResourcesDeleter) deleteEachItem(
	dynamicClient *dynamic.Client, gvr unversioned.GroupVersionResource, namespace string) error {
	glog.V(5).Infof("namespace controller - deleteEachItem - namespace: %s, gvr: %v", namespace, gvr)

	unstructuredList, listSupported, err := d.listCollection(dynamicClient, gvr, namespace)
	if err != nil {
		return err
	}
	if !listSupported {
		return nil
	}
	apiResource := unversioned.APIResource{Name: gvr.Resource, Namespaced: true}
	for _, item := range unstructuredList.Items {
		if err = dynamicClient.Resource(&apiResource, namespace).Delete(item.GetName(), nil); err != nil && !errors.IsNotFound(err) && !errors.IsMethodNotSupported(err) {
			return err
		}
	}
	return nil
}

// deleteAllContentForGroupVersionResource will use the dynamic client to delete each resource identified in gvr.
// It returns an estimate of the time remaining before the remaining resources are deleted.
// If estimate > 0, not all resources are guaranteed to be gone.
func (d *NamespacedResourcesDeleter) deleteAllContentForGroupVersionResource(
	gvr unversioned.GroupVersionResource, namespace string,
	namespaceDeletedAt unversioned.Time) (int64, error) {
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - namespace: %s, gvr: %v", namespace, gvr)

	// estimate how long it will take for the resource to be deleted (needed for objects that support graceful delete)
	estimate, err := d.estimateGracefulTermination(gvr, namespace, namespaceDeletedAt)
	if err != nil {
		glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - unable to estimate - namespace: %s, gvr: %v, err: %v", namespace, gvr, err)
		return estimate, err
	}
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - estimate - namespace: %s, gvr: %v, estimate: %v", namespace, gvr, estimate)

	// get a client for this group version...
	dynamicClient, err := d.clientPool.ClientForGroupVersionResource(gvr)
	if err != nil {
		glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - unable to get client - namespace: %s, gvr: %v, err: %v", namespace, gvr, err)
		return estimate, err
	}

	// first try to delete the entire collection
	deleteCollectionSupported, err := d.deleteCollection(dynamicClient, gvr, namespace)
	if err != nil {
		return estimate, err
	}

	// delete collection was not supported, so we list and delete each item...
	if !deleteCollectionSupported {
		err = d.deleteEachItem(dynamicClient, gvr, namespace)
		if err != nil {
			return estimate, err
		}
	}

	// verify there are no more remaining items
	// it is not an error condition for there to be remaining items if local estimate is non-zero
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - checking for no more items in namespace: %s, gvr: %v", namespace, gvr)
	unstructuredList, listSupported, err := d.listCollection(dynamicClient, gvr, namespace)
	if err != nil {
		glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - error verifying no items in namespace: %s, gvr: %v, err: %v", namespace, gvr, err)
		return estimate, err
	}
	if !listSupported {
		return estimate, nil
	}
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - items remaining - namespace: %s, gvr: %v, items: %v", namespace, gvr, len(unstructuredList.Items))
	if len(unstructuredList.Items) != 0 && estimate == int64(0) {
		// if any item has a finalizer, we treat that as a normal condition, and use a default estimation to allow for GC to complete.
		for _, item := range unstructuredList.Items {
			if len(item.GetFinalizers()) > 0 {
				glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - items remaining with finalizers - namespace: %s, gvr: %v, finalizers: %v", namespace, gvr, item.GetFinalizers())
				return finalizerEstimateSeconds, nil
			}
		}
		// nothing reported a finalizer, so something was unexpected as it should have been deleted.
		return estimate, fmt.Errorf("unexpected items still remain in namespace: %s for gvr: %v", namespace, gvr)
	}
	return estimate, nil
}

// deleteAllContent will use the dynamic client to delete each resource identified in groupVersionResources.
// It returns an estimate of the time remaining before the remaining resources are deleted.
// If estimate > 0, not all resources are guaranteed to be gone.
func (d *NamespacedResourcesDeleter) deleteAllContent(
	namespace string, namespaceDeletedAt unversioned.Time) (int64, error) {
	estimate := int64(0)
	glog.V(4).Infof("namespace controller - deleteAllContent - namespace: %s, gvrs: %v", namespace, d.groupVersionResources)
	// iterate over each group version, and attempt to delete all of its resources
	// we sort resources to delete in a priority order that deletes pods LAST
	sort.Sort(sortableGroupVersionResources(d.groupVersionResources))
	for _, gvr := range d.groupVersionResources {
		gvrEstimate, err := d.deleteAllContentForGroupVersionResource(gvr, namespace, namespaceDeletedAt)
		if err != nil {
			return estimate, err
		}
		if gvrEstimate > estimate {
			estimate = gvrEstimate
		}
	}
	glog.V(4).Infof("namespace controller - deleteAllContent - namespace: %s, estimate: %v", namespace, estimate)
	return estimate, nil
}

// estimateGrracefulTermination will estimate the graceful termination required for the specific entity in the namespace
func (d *NamespacedResourcesDeleter) estimateGracefulTermination(gvr unversioned.GroupVersionResource, ns string, namespaceDeletedAt unversioned.Time) (int64, error) {
	groupResource := gvr.GroupResource()
	glog.V(5).Infof("namespace controller - estimateGracefulTermination - group %s, resource: %s", groupResource.Group, groupResource.Resource)
	estimate := int64(0)
	var err error
	switch groupResource {
	case unversioned.GroupResource{Group: "", Resource: "pods"}:
		dynamicClient, err := d.clientPool.ClientForGroupVersionResource(gvr)
		if err != nil {
			return estimate, fmt.Errorf("error in creating dynamic client for resource: %s", gvr)
		}
		estimate, err = d.estimateGracefulTerminationForPods(dynamicClient, ns)
	}
	if err != nil {
		return estimate, err
	}
	// determine if the estimate is greater than the deletion timestamp
	duration := time.Since(namespaceDeletedAt.Time)
	allowedEstimate := time.Duration(estimate) * time.Second
	if duration >= allowedEstimate {
		estimate = int64(0)
	}
	return estimate, nil
}

// estimateGracefulTerminationForPods determines the graceful termination period for pods in the namespace
func (d *NamespacedResourcesDeleter) estimateGracefulTerminationForPods(dynamicClient *dynamic.Client, ns string) (int64, error) {
	glog.V(5).Infof("namespace controller - estimateGracefulTerminationForPods - namespace %s", ns)
	estimate := int64(0)
	resource := &unversioned.APIResource{
		Namespaced: true,
		Kind:       "Pod",
	}
	listResponse, err := dynamicClient.Resource(resource, ns).List(&api.ListOptions{})
	if err != nil {
		return estimate, err
	}
	items, ok := listResponse.(*runtime.UnstructuredList)
	if !ok {
		return estimate, fmt.Errorf("unexpected: expected type runtime.UnstructuredList, got: %#v", listResponse)
	}
	for i := range items.Items {
		item := items.Items[i]
		pod, err := unstructuredToPod(item)
		if err != nil {
			return estimate, fmt.Errorf("unexpected: expected type api.Pod, got: %#v", item)
		}
		// filter out terminal pods
		phase := pod.Status.Phase
		if api.PodSucceeded == phase || api.PodFailed == phase {
			continue
		}
		if pod.Spec.TerminationGracePeriodSeconds != nil {
			grace := *pod.Spec.TerminationGracePeriodSeconds
			if grace > estimate {
				estimate = grace
			}
		}
	}
	return estimate, nil
}

func unstructuredToPod(obj *runtime.Unstructured) (*api.Pod, error) {
	json, err := runtime.Encode(runtime.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	pod := new(api.Pod)
	err = runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, pod)
	return pod, err
}

// sortableGroupVersionResources sorts the input set of resources for deletion, and orders pods to always be last.
// the idea is that the namespace controller will delete all things that spawn pods first in order to reduce the time
// those controllers spend creating pods only to be told NO in admission and potentially overwhelming cluster especially if they lack rate limiting.
type sortableGroupVersionResources []unversioned.GroupVersionResource

func (list sortableGroupVersionResources) Len() int {
	return len(list)
}

func (list sortableGroupVersionResources) Swap(i, j int) {
	list[i], list[j] = list[j], list[i]
}

func (list sortableGroupVersionResources) Less(i, j int) bool {
	return list[j].Group == "" && list[j].Resource == "pods"
}
