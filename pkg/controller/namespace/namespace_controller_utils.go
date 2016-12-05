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

package namespace

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

// contentRemainingError is used to inform the caller that content is not fully removed from the namespace
type contentRemainingError struct {
	Estimate int64
}

func (e *contentRemainingError) Error() string {
	return fmt.Sprintf("some content remains in the namespace, estimate %d seconds before it is removed", e.Estimate)
}

// operation is used for caching if an operation is supported on a dynamic client.
type operation string

const (
	operationDeleteCollection operation = "deleteCollection"
	operationList             operation = "list"
	// assume a default estimate for finalizers to complete when found on items pending deletion.
	finalizerEstimateSeconds int64 = int64(15)
)

// operationKey is an entry in a cache.
type operationKey struct {
	op  operation
	gvr schema.GroupVersionResource
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
type updateNamespaceFunc func(kubeClient clientset.Interface, namespace *v1.Namespace) (*v1.Namespace, error)

// retryOnConflictError retries the specified fn if there was a conflict error
// it will return an error if the UID for an object changes across retry operations.
// TODO RetryOnConflict should be a generic concept in client code
func retryOnConflictError(kubeClient clientset.Interface, namespace *v1.Namespace, fn updateNamespaceFunc) (result *v1.Namespace, err error) {
	latestNamespace := namespace
	for {
		result, err = fn(kubeClient, latestNamespace)
		if err == nil {
			return result, nil
		}
		if !errors.IsConflict(err) {
			return nil, err
		}
		prevNamespace := latestNamespace
		latestNamespace, err = kubeClient.Core().Namespaces().Get(latestNamespace.Name)
		if err != nil {
			return nil, err
		}
		if prevNamespace.UID != latestNamespace.UID {
			return nil, fmt.Errorf("namespace uid has changed across retries")
		}
	}
}

// updateNamespaceStatusFunc will verify that the status of the namespace is correct
func updateNamespaceStatusFunc(kubeClient clientset.Interface, namespace *v1.Namespace) (*v1.Namespace, error) {
	if namespace.DeletionTimestamp.IsZero() || namespace.Status.Phase == v1.NamespaceTerminating {
		return namespace, nil
	}
	newNamespace := v1.Namespace{}
	newNamespace.ObjectMeta = namespace.ObjectMeta
	newNamespace.Status = namespace.Status
	newNamespace.Status.Phase = v1.NamespaceTerminating
	return kubeClient.Core().Namespaces().UpdateStatus(&newNamespace)
}

// finalized returns true if the namespace.Spec.Finalizers is an empty list
func finalized(namespace *v1.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalizeNamespaceFunc returns a function that knows how to finalize a namespace for specified token.
func finalizeNamespaceFunc(finalizerToken v1.FinalizerName) updateNamespaceFunc {
	return func(kubeClient clientset.Interface, namespace *v1.Namespace) (*v1.Namespace, error) {
		return finalizeNamespace(kubeClient, namespace, finalizerToken)
	}
}

// finalizeNamespace removes the specified finalizerToken and finalizes the namespace
func finalizeNamespace(kubeClient clientset.Interface, namespace *v1.Namespace, finalizerToken v1.FinalizerName) (*v1.Namespace, error) {
	namespaceFinalize := v1.Namespace{}
	namespaceFinalize.ObjectMeta = namespace.ObjectMeta
	namespaceFinalize.Spec = namespace.Spec
	finalizerSet := sets.NewString()
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != finalizerToken {
			finalizerSet.Insert(string(namespace.Spec.Finalizers[i]))
		}
	}
	namespaceFinalize.Spec.Finalizers = make([]v1.FinalizerName, 0, len(finalizerSet))
	for _, value := range finalizerSet.List() {
		namespaceFinalize.Spec.Finalizers = append(namespaceFinalize.Spec.Finalizers, v1.FinalizerName(value))
	}
	namespace, err := kubeClient.Core().Namespaces().Finalize(&namespaceFinalize)
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
func deleteCollection(
	dynamicClient *dynamic.Client,
	opCache *operationNotSupportedCache,
	gvr schema.GroupVersionResource,
	namespace string,
) (bool, error) {
	glog.V(5).Infof("namespace controller - deleteCollection - namespace: %s, gvr: %v", namespace, gvr)

	key := operationKey{op: operationDeleteCollection, gvr: gvr}
	if !opCache.isSupported(key) {
		glog.V(5).Infof("namespace controller - deleteCollection ignored since not supported - namespace: %s, gvr: %v", namespace, gvr)
		return false, nil
	}

	apiResource := metav1.APIResource{Name: gvr.Resource, Namespaced: true}

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
		opCache.setNotSupported(key)
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
func listCollection(
	dynamicClient *dynamic.Client,
	opCache *operationNotSupportedCache,
	gvr schema.GroupVersionResource,
	namespace string,
) (*runtime.UnstructuredList, bool, error) {
	glog.V(5).Infof("namespace controller - listCollection - namespace: %s, gvr: %v", namespace, gvr)

	key := operationKey{op: operationList, gvr: gvr}
	if !opCache.isSupported(key) {
		glog.V(5).Infof("namespace controller - listCollection ignored since not supported - namespace: %s, gvr: %v", namespace, gvr)
		return nil, false, nil
	}

	apiResource := metav1.APIResource{Name: gvr.Resource, Namespaced: true}
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
		opCache.setNotSupported(key)
		return nil, false, nil
	}

	return nil, true, err
}

// deleteEachItem is a helper function that will list the collection of resources and delete each item 1 by 1.
func deleteEachItem(
	dynamicClient *dynamic.Client,
	opCache *operationNotSupportedCache,
	gvr schema.GroupVersionResource,
	namespace string,
) error {
	glog.V(5).Infof("namespace controller - deleteEachItem - namespace: %s, gvr: %v", namespace, gvr)

	unstructuredList, listSupported, err := listCollection(dynamicClient, opCache, gvr, namespace)
	if err != nil {
		return err
	}
	if !listSupported {
		return nil
	}
	apiResource := metav1.APIResource{Name: gvr.Resource, Namespaced: true}
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
func deleteAllContentForGroupVersionResource(
	kubeClient clientset.Interface,
	clientPool dynamic.ClientPool,
	opCache *operationNotSupportedCache,
	gvr schema.GroupVersionResource,
	namespace string,
	namespaceDeletedAt metav1.Time,
) (int64, error) {
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - namespace: %s, gvr: %v", namespace, gvr)

	// estimate how long it will take for the resource to be deleted (needed for objects that support graceful delete)
	estimate, err := estimateGracefulTermination(kubeClient, gvr, namespace, namespaceDeletedAt)
	if err != nil {
		glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - unable to estimate - namespace: %s, gvr: %v, err: %v", namespace, gvr, err)
		return estimate, err
	}
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - estimate - namespace: %s, gvr: %v, estimate: %v", namespace, gvr, estimate)

	// get a client for this group version...
	dynamicClient, err := clientPool.ClientForGroupVersionResource(gvr)
	if err != nil {
		glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - unable to get client - namespace: %s, gvr: %v, err: %v", namespace, gvr, err)
		return estimate, err
	}

	// first try to delete the entire collection
	deleteCollectionSupported, err := deleteCollection(dynamicClient, opCache, gvr, namespace)
	if err != nil {
		return estimate, err
	}

	// delete collection was not supported, so we list and delete each item...
	if !deleteCollectionSupported {
		err = deleteEachItem(dynamicClient, opCache, gvr, namespace)
		if err != nil {
			return estimate, err
		}
	}

	// verify there are no more remaining items
	// it is not an error condition for there to be remaining items if local estimate is non-zero
	glog.V(5).Infof("namespace controller - deleteAllContentForGroupVersionResource - checking for no more items in namespace: %s, gvr: %v", namespace, gvr)
	unstructuredList, listSupported, err := listCollection(dynamicClient, opCache, gvr, namespace)
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
func deleteAllContent(
	kubeClient clientset.Interface,
	clientPool dynamic.ClientPool,
	opCache *operationNotSupportedCache,
	groupVersionResources map[schema.GroupVersionResource]struct{},
	namespace string,
	namespaceDeletedAt metav1.Time,
) (int64, error) {
	estimate := int64(0)
	glog.V(4).Infof("namespace controller - deleteAllContent - namespace: %s, gvrs: %v", namespace, groupVersionResources)
	for gvr := range groupVersionResources {
		gvrEstimate, err := deleteAllContentForGroupVersionResource(kubeClient, clientPool, opCache, gvr, namespace, namespaceDeletedAt)
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

// syncNamespace orchestrates deletion of a Namespace and its associated content.
func syncNamespace(
	kubeClient clientset.Interface,
	clientPool dynamic.ClientPool,
	opCache *operationNotSupportedCache,
	discoverResourcesFn func() ([]*metav1.APIResourceList, error),
	namespace *v1.Namespace,
	finalizerToken v1.FinalizerName,
) error {
	if namespace.DeletionTimestamp == nil {
		return nil
	}

	// multiple controllers may edit a namespace during termination
	// first get the latest state of the namespace before proceeding
	// if the namespace was deleted already, don't do anything
	namespace, err := kubeClient.Core().Namespaces().Get(namespace.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	glog.V(5).Infof("namespace controller - syncNamespace - namespace: %s, finalizerToken: %s", namespace.Name, finalizerToken)

	// ensure that the status is up to date on the namespace
	// if we get a not found error, we assume the namespace is truly gone
	namespace, err = retryOnConflictError(kubeClient, namespace, updateNamespaceStatusFunc)
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

	// if the namespace is already finalized, delete it
	if finalized(namespace) {
		var opts *v1.DeleteOptions
		uid := namespace.UID
		if len(uid) > 0 {
			opts = &v1.DeleteOptions{Preconditions: &v1.Preconditions{UID: &uid}}
		}
		err = kubeClient.Core().Namespaces().Delete(namespace.Name, opts)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		return nil
	}

	// there may still be content for us to remove
	resources, err := discoverResourcesFn()
	if err != nil {
		return err
	}
	// TODO(sttts): get rid of opCache and pass the verbs (especially "deletecollection") down into the deleter
	deletableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"delete"}}, resources)
	groupVersionResources, err := discovery.GroupVersionResources(deletableResources)
	if err != nil {
		return err
	}
	estimate, err := deleteAllContent(kubeClient, clientPool, opCache, groupVersionResources, namespace.Name, *namespace.DeletionTimestamp)
	if err != nil {
		return err
	}
	if estimate > 0 {
		return &contentRemainingError{estimate}
	}

	// we have removed content, so mark it finalized by us
	result, err := retryOnConflictError(kubeClient, namespace, finalizeNamespaceFunc(finalizerToken))
	if err != nil {
		// in normal practice, this should not be possible, but if a deployment is running
		// two controllers to do namespace deletion that share a common finalizer token it's
		// possible that a not found could occur since the other controller would have finished the delete.
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// now check if all finalizers have reported that we delete now
	if finalized(result) {
		err = kubeClient.Core().Namespaces().Delete(namespace.Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}

	return nil
}

// estimateGrracefulTermination will estimate the graceful termination required for the specific entity in the namespace
func estimateGracefulTermination(kubeClient clientset.Interface, groupVersionResource schema.GroupVersionResource, ns string, namespaceDeletedAt metav1.Time) (int64, error) {
	groupResource := groupVersionResource.GroupResource()
	glog.V(5).Infof("namespace controller - estimateGracefulTermination - group %s, resource: %s", groupResource.Group, groupResource.Resource)
	estimate := int64(0)
	var err error
	switch groupResource {
	case schema.GroupResource{Group: "", Resource: "pods"}:
		estimate, err = estimateGracefulTerminationForPods(kubeClient, ns)
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
func estimateGracefulTerminationForPods(kubeClient clientset.Interface, ns string) (int64, error) {
	glog.V(5).Infof("namespace controller - estimateGracefulTerminationForPods - namespace %s", ns)
	estimate := int64(0)
	items, err := kubeClient.Core().Pods(ns).List(v1.ListOptions{})
	if err != nil {
		return estimate, err
	}
	for i := range items.Items {
		// filter out terminal pods
		phase := items.Items[i].Status.Phase
		if v1.PodSucceeded == phase || v1.PodFailed == phase {
			continue
		}
		if items.Items[i].Spec.TerminationGracePeriodSeconds != nil {
			grace := *items.Items[i].Spec.TerminationGracePeriodSeconds
			if grace > estimate {
				estimate = grace
			}
		}
	}
	return estimate, nil
}
