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
	"reflect"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	v1clientset "k8s.io/client-go/kubernetes/typed/core/v1"
)

// Interface to delete a namespace with all resources in it.
type NamespacedResourcesDeleterInterface interface {
	Delete(nsName string) error
}

func NewNamespacedResourcesDeleter(nsClient v1clientset.NamespaceInterface,
	dynamicClient dynamic.DynamicInterface, podsGetter v1clientset.PodsGetter,
	discoverResourcesFn func() ([]*metav1.APIResourceList, error),
	finalizerToken v1.FinalizerName, deleteNamespaceWhenDone bool) NamespacedResourcesDeleterInterface {
	d := &namespacedResourcesDeleter{
		nsClient:      nsClient,
		dynamicClient: dynamicClient,
		podsGetter:    podsGetter,
		opCache: &operationNotSupportedCache{
			m: make(map[operationKey]bool),
		},
		discoverResourcesFn:     discoverResourcesFn,
		finalizerToken:          finalizerToken,
		deleteNamespaceWhenDone: deleteNamespaceWhenDone,
	}
	d.initOpCache()
	return d
}

var _ NamespacedResourcesDeleterInterface = &namespacedResourcesDeleter{}

// namespacedResourcesDeleter is used to delete all resources in a given namespace.
type namespacedResourcesDeleter struct {
	// Client to manipulate the namespace.
	nsClient v1clientset.NamespaceInterface
	// Dynamic client to list and delete all namespaced resources.
	dynamicClient dynamic.DynamicInterface
	// Interface to get PodInterface.
	podsGetter v1clientset.PodsGetter
	// Cache of what operations are not supported on each group version resource.
	opCache             *operationNotSupportedCache
	discoverResourcesFn func() ([]*metav1.APIResourceList, error)
	// The finalizer token that should be removed from the namespace
	// when all resources in that namespace have been deleted.
	finalizerToken v1.FinalizerName
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
func (d *namespacedResourcesDeleter) Delete(nsName string) error {
	// Multiple controllers may edit a namespace during termination
	// first get the latest state of the namespace before proceeding
	// if the namespace was deleted already, don't do anything
	namespace, err := d.nsClient.Get(nsName, metav1.GetOptions{})
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

func (d *namespacedResourcesDeleter) initOpCache() {
	// pre-fill opCache with the discovery info
	//
	// TODO(sttts): get rid of opCache and http 405 logic around it and trust discovery info
	resources, err := d.discoverResourcesFn()
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to get all supported resources from server: %v", err))
	}
	if len(resources) == 0 {
		glog.Fatalf("Unable to get any supported resources from server: %v", err)
	}
	deletableGroupVersionResources := []schema.GroupVersionResource{}
	for _, rl := range resources {
		gv, err := schema.ParseGroupVersion(rl.GroupVersion)
		if err != nil {
			glog.Errorf("Failed to parse GroupVersion %q, skipping: %v", rl.GroupVersion, err)
			continue
		}

		for _, r := range rl.APIResources {
			gvr := schema.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: r.Name}
			verbs := sets.NewString([]string(r.Verbs)...)

			if !verbs.Has("delete") {
				glog.V(6).Infof("Skipping resource %v because it cannot be deleted.", gvr)
			}

			for _, op := range []operation{operationList, operationDeleteCollection} {
				if !verbs.Has(string(op)) {
					d.opCache.setNotSupported(operationKey{operation: op, gvr: gvr})
				}
			}
			deletableGroupVersionResources = append(deletableGroupVersionResources, gvr)
		}
	}
}

// Deletes the given namespace.
func (d *namespacedResourcesDeleter) deleteNamespace(namespace *v1.Namespace) error {
	var opts *metav1.DeleteOptions
	uid := namespace.UID
	if len(uid) > 0 {
		opts = &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}}
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
type operation string

const (
	operationDeleteCollection operation = "deletecollection"
	operationList             operation = "list"
	// assume a default estimate for finalizers to complete when found on items pending deletion.
	finalizerEstimateSeconds int64 = int64(15)
)

// operationKey is an entry in a cache.
type operationKey struct {
	operation operation
	gvr       schema.GroupVersionResource
}

// operationNotSupportedCache is a simple cache to remember if an operation is not supported for a resource.
// if the operationKey maps to true, it means the operation is not supported.
type operationNotSupportedCache struct {
	lock sync.RWMutex
	m    map[operationKey]bool
}

// isSupported returns true if the operation is supported
func (o *operationNotSupportedCache) isSupported(key operationKey) bool {
	o.lock.RLock()
	defer o.lock.RUnlock()
	return !o.m[key]
}

func (o *operationNotSupportedCache) setNotSupported(key operationKey) {
	o.lock.Lock()
	defer o.lock.Unlock()
	o.m[key] = true
}

// updateNamespaceFunc is a function that makes an update to a namespace
type updateNamespaceFunc func(namespace *v1.Namespace) (*v1.Namespace, error)

// retryOnConflictError retries the specified fn if there was a conflict error
// it will return an error if the UID for an object changes across retry operations.
// TODO RetryOnConflict should be a generic concept in client code
func (d *namespacedResourcesDeleter) retryOnConflictError(namespace *v1.Namespace, fn updateNamespaceFunc) (result *v1.Namespace, err error) {
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
		latestNamespace, err = d.nsClient.Get(latestNamespace.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if prevNamespace.UID != latestNamespace.UID {
			return nil, fmt.Errorf("namespace uid has changed across retries")
		}
	}
}

// updateNamespaceStatusFunc will verify that the status of the namespace is correct
func (d *namespacedResourcesDeleter) updateNamespaceStatusFunc(namespace *v1.Namespace) (*v1.Namespace, error) {
	if namespace.DeletionTimestamp.IsZero() || namespace.Status.Phase == v1.NamespaceTerminating {
		return namespace, nil
	}
	newNamespace := v1.Namespace{}
	newNamespace.ObjectMeta = namespace.ObjectMeta
	newNamespace.Status = namespace.Status
	newNamespace.Status.Phase = v1.NamespaceTerminating
	return d.nsClient.UpdateStatus(&newNamespace)
}

// finalized returns true if the namespace.Spec.Finalizers is an empty list
func finalized(namespace *v1.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalizeNamespace removes the specified finalizerToken and finalizes the namespace
func (d *namespacedResourcesDeleter) finalizeNamespace(namespace *v1.Namespace) (*v1.Namespace, error) {
	namespaceFinalize := v1.Namespace{}
	namespaceFinalize.ObjectMeta = namespace.ObjectMeta
	namespaceFinalize.Spec = namespace.Spec
	finalizerSet := sets.NewString()
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != d.finalizerToken {
			finalizerSet.Insert(string(namespace.Spec.Finalizers[i]))
		}
	}
	namespaceFinalize.Spec.Finalizers = make([]v1.FinalizerName, 0, len(finalizerSet))
	for _, value := range finalizerSet.List() {
		namespaceFinalize.Spec.Finalizers = append(namespaceFinalize.Spec.Finalizers, v1.FinalizerName(value))
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
func (d *namespacedResourcesDeleter) deleteCollection(gvr schema.GroupVersionResource, namespace string) (bool, error) {
	glog.V(5).Infof("namespace controller - deleteCollection - namespace: %s, gvr: %v", namespace, gvr)

	key := operationKey{operation: operationDeleteCollection, gvr: gvr}
	if !d.opCache.isSupported(key) {
		glog.V(5).Infof("namespace controller - deleteCollection ignored since not supported - namespace: %s, gvr: %v", namespace, gvr)
		return false, nil
	}

	// namespace controller does not want the garbage collector to insert the orphan finalizer since it calls
	// resource deletions generically.  it will ensure all resources in the namespace are purged prior to releasing
	// namespace itself.
	background := metav1.DeletePropagationBackground
	opts := &metav1.DeleteOptions{PropagationPolicy: &background}
	err := d.dynamicClient.NamespacedResource(gvr, namespace).DeleteCollection(opts, metav1.ListOptions{})

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
		d.opCache.setNotSupported(key)
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
func (d *namespacedResourcesDeleter) listCollection(gvr schema.GroupVersionResource, namespace string) (*unstructured.UnstructuredList, bool, error) {
	glog.V(5).Infof("namespace controller - listCollection - namespace: %s, gvr: %v", namespace, gvr)

	key := operationKey{operation: operationList, gvr: gvr}
	if !d.opCache.isSupported(key) {
		glog.V(5).Infof("namespace controller - listCollection ignored since not supported - namespace: %s, gvr: %v", namespace, gvr)
		return nil, false, nil
	}

	unstructuredList, err := d.dynamicClient.NamespacedResource(gvr, namespace).List(metav1.ListOptions{IncludeUninitialized: true})
	if err == nil {
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
		d.opCache.setNotSupported(key)
		return nil, false, nil
	}

	return nil, true, err
}

// deleteEachItem is a helper function that will list the collection of resources and delete each item 1 by 1.
func (d *namespacedResourcesDeleter) deleteEachItem(gvr schema.GroupVersionResource, namespace string) error {
	glog.V(5).Infof("namespace controller - deleteEachItem - namespace: %s, gvr: %v", namespace, gvr)

	unstructuredList, listSupported, err := d.listCollection(gvr, namespace)
	if err != nil {
		return err
	}
	if !listSupported {
		return nil
	}
	for _, item := range unstructuredList.Items {
		background := metav1.DeletePropagationBackground
		opts := &metav1.DeleteOptions{PropagationPolicy: &background}
		if err = d.dynamicClient.NamespacedResource(gvr, namespace).Delete(item.GetName(), opts); err != nil && !errors.IsNotFound(err) && !errors.IsMethodNotSupported(err) {
			return err
		}
	}
	return nil
}

// deleteAllContentForGroupVersionResource will use the dynamic client to delete each resource identified in gvr.
// It returns an estimate of the time remaining before the remaining resources are deleted.
// If estimate > 0, not all resources are guaranteed to be gone.
func (d *namespacedResourcesDeleter) deleteAllContentForGroupVersionResource(
	gvr schema.GroupVersionResource, namespace string,
	namespaceDeletedAt metav1.Time) (int64, error) {
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - namespace: %s, gvr: %v", namespace, gvr)

	// estimate how long it will take for the resource to be deleted (needed for objects that support graceful delete)
	estimate, err := d.estimateGracefulTermination(gvr, namespace, namespaceDeletedAt)
	if err != nil {
		glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - unable to estimate - namespace: %s, gvr: %v, err: %v", namespace, gvr, err)
		return estimate, err
	}
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - estimate - namespace: %s, gvr: %v, estimate: %v", namespace, gvr, estimate)

	// first try to delete the entire collection
	deleteCollectionSupported, err := d.deleteCollection(gvr, namespace)
	if err != nil {
		return estimate, err
	}

	// delete collection was not supported, so we list and delete each item...
	if !deleteCollectionSupported {
		err = d.deleteEachItem(gvr, namespace)
		if err != nil {
			return estimate, err
		}
	}

	// verify there are no more remaining items
	// it is not an error condition for there to be remaining items if local estimate is non-zero
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - checking for no more items in namespace: %s, gvr: %v", namespace, gvr)
	unstructuredList, listSupported, err := d.listCollection(gvr, namespace)
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
func (d *namespacedResourcesDeleter) deleteAllContent(namespace string, namespaceDeletedAt metav1.Time) (int64, error) {
	estimate := int64(0)
	glog.V(4).Infof("namespace controller - deleteAllContent - namespace: %s", namespace)
	resources, err := d.discoverResourcesFn()
	if err != nil {
		return estimate, err
	}
	// TODO(sttts): get rid of opCache and pass the verbs (especially "deletecollection") down into the deleter
	deletableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"delete"}}, resources)
	groupVersionResources, err := discovery.GroupVersionResources(deletableResources)
	if err != nil {
		return estimate, err
	}
	var errs []error
	for gvr := range groupVersionResources {
		gvrEstimate, err := d.deleteAllContentForGroupVersionResource(gvr, namespace, namespaceDeletedAt)
		if err != nil {
			// If there is an error, hold on to it but proceed with all the remaining
			// groupVersionResources.
			errs = append(errs, err)
		}
		if gvrEstimate > estimate {
			estimate = gvrEstimate
		}
	}
	if len(errs) > 0 {
		return estimate, utilerrors.NewAggregate(errs)
	}
	glog.V(4).Infof("namespace controller - deleteAllContent - namespace: %s, estimate: %v", namespace, estimate)
	return estimate, nil
}

// estimateGrracefulTermination will estimate the graceful termination required for the specific entity in the namespace
func (d *namespacedResourcesDeleter) estimateGracefulTermination(gvr schema.GroupVersionResource, ns string, namespaceDeletedAt metav1.Time) (int64, error) {
	groupResource := gvr.GroupResource()
	glog.V(5).Infof("namespace controller - estimateGracefulTermination - group %s, resource: %s", groupResource.Group, groupResource.Resource)
	estimate := int64(0)
	var err error
	switch groupResource {
	case schema.GroupResource{Group: "", Resource: "pods"}:
		estimate, err = d.estimateGracefulTerminationForPods(ns)
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
func (d *namespacedResourcesDeleter) estimateGracefulTerminationForPods(ns string) (int64, error) {
	glog.V(5).Infof("namespace controller - estimateGracefulTerminationForPods - namespace %s", ns)
	estimate := int64(0)
	podsGetter := d.podsGetter
	if podsGetter == nil || reflect.ValueOf(podsGetter).IsNil() {
		return estimate, fmt.Errorf("unexpected: podsGetter is nil. Cannot estimate grace period seconds for pods")
	}
	items, err := podsGetter.Pods(ns).List(metav1.ListOptions{IncludeUninitialized: true})
	if err != nil {
		return estimate, err
	}
	for i := range items.Items {
		pod := items.Items[i]
		// filter out terminal pods
		phase := pod.Status.Phase
		if v1.PodSucceeded == phase || v1.PodFailed == phase {
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
