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
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	kcpcorev1client "github.com/kcp-dev/client-go/kubernetes/typed/core/v1"
	kcpmetadata "github.com/kcp-dev/client-go/metadata"
	"github.com/kcp-dev/logicalcluster/v3"
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/discovery"
)

// NamespacedResourcesDeleterInterface is the interface to delete a namespace with all resources in it.
type NamespacedResourcesDeleterInterface interface {
	Delete(ctx context.Context, clusterName logicalcluster.Name, nsName string) error
}

// NewNamespacedResourcesDeleter returns a new NamespacedResourcesDeleter.
func NewNamespacedResourcesDeleter(ctx context.Context, nsClient kcpcorev1client.NamespaceClusterInterface,
	metadataClient kcpmetadata.ClusterInterface, podsGetter kcpcorev1client.PodsClusterGetter,
	discoverResourcesFn func(clusterName logicalcluster.Path) ([]*metav1.APIResourceList, error),
	finalizerToken v1.FinalizerName) NamespacedResourcesDeleterInterface {
	d := &namespacedResourcesDeleter{
		nsClient:            nsClient,
		metadataClient:      metadataClient,
		podsGetter:          podsGetter,
		opCaches:            map[logicalcluster.Name]*operationNotSupportedCache{},
		discoverResourcesFn: discoverResourcesFn,
		finalizerToken:      finalizerToken,
	}
	return d
}

var _ NamespacedResourcesDeleterInterface = &namespacedResourcesDeleter{}

// namespacedResourcesDeleter is used to delete all resources in a given namespace.
type namespacedResourcesDeleter struct {
	// Client to manipulate the namespace.
	nsClient kcpcorev1client.NamespaceClusterInterface
	// Dynamic client to list and delete all namespaced resources.
	metadataClient kcpmetadata.ClusterInterface
	// Interface to get PodInterface.
	podsGetter kcpcorev1client.PodsClusterGetter
	// Cache of what operations are not supported on each group version resource.
	opCaches      map[logicalcluster.Name]*operationNotSupportedCache
	opCachesMutex sync.RWMutex

	discoverResourcesFn func(clusterName logicalcluster.Path) ([]*metav1.APIResourceList, error)
	// The finalizer token that should be removed from the namespace
	// when all resources in that namespace have been deleted.
	finalizerToken v1.FinalizerName
}

// Delete deletes all resources in the given namespace.
// Before deleting resources:
//   - It ensures that deletion timestamp is set on the
//     namespace (does nothing if deletion timestamp is missing).
//   - Verifies that the namespace is in the "terminating" phase
//     (updates the namespace phase if it is not yet marked terminating)
//
// After deleting the resources:
// * It removes finalizer token from the given namespace.
//
// Returns an error if any of those steps fail.
// Returns ResourcesRemainingError if it deleted some resources but needs
// to wait for them to go away.
// Caller is expected to keep calling this until it succeeds.
func (d *namespacedResourcesDeleter) Delete(ctx context.Context, clusterName logicalcluster.Name, nsName string) error {
	// Multiple controllers may edit a namespace during termination
	// first get the latest state of the namespace before proceeding
	// if the namespace was deleted already, don't do anything
	namespace, err := d.nsClient.Cluster(clusterName.Path()).Get(ctx, nsName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if namespace.DeletionTimestamp == nil {
		return nil
	}

	klog.FromContext(ctx).V(5).Info("Namespace controller - syncNamespace", "namespace", namespace.Name, "finalizerToken", d.finalizerToken)

	// ensure that the status is up to date on the namespace
	// if we get a not found error, we assume the namespace is truly gone
	namespace, err = d.retryOnConflictError(ctx, namespace, d.updateNamespaceStatusFunc)
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

	// return if it is already finalized.
	if finalized(namespace) {
		return nil
	}

	// there may still be content for us to remove
	estimate, err := d.deleteAllContent(ctx, namespace)
	if err != nil {
		return err
	}
	if estimate > 0 {
		return &ResourcesRemainingError{estimate}
	}

	// we have removed content, so mark it finalized by us
	_, err = d.retryOnConflictError(ctx, namespace, d.finalizeNamespace)
	if err != nil {
		// in normal practice, this should not be possible, but if a deployment is running
		// two controllers to do namespace deletion that share a common finalizer token it's
		// possible that a not found could occur since the other controller would have finished the delete.
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}
	return nil
}

func (d *namespacedResourcesDeleter) initOpCache(ctx context.Context, clusterName logicalcluster.Name) {
	// pre-fill opCache with the discovery info
	//
	// TODO(sttts): get rid of opCache and http 405 logic around it and trust discovery info
	resources, err := d.discoverResourcesFn(clusterName.Path())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to get all supported resources from server: %v", err))
	}
	logger := klog.FromContext(ctx)
	if len(resources) == 0 {
		logger.Error(err, "Unable to get any supported resources from server")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	for _, rl := range resources {
		gv, err := schema.ParseGroupVersion(rl.GroupVersion)
		if err != nil {
			logger.Error(err, "Failed to parse GroupVersion, skipping", "groupVersion", rl.GroupVersion)
			continue
		}

		for _, r := range rl.APIResources {
			gvr := schema.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: r.Name}
			verbs := sets.NewString([]string(r.Verbs)...)

			if !verbs.Has("delete") {
				logger.V(6).Info("Skipping resource because it cannot be deleted", "resource", gvr)
			}

			for _, op := range []operation{operationList, operationDeleteCollection} {
				if !verbs.Has(string(op)) {
					d.opCaches[clusterName].setNotSupported(operationKey{operation: op, gvr: gvr})
				}
			}
		}
	}
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
	finalizerEstimateSeconds = int64(15)
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

// isSupported returns true if the operation is supported
func (d *namespacedResourcesDeleter) isSupported(ctx context.Context, clusterName logicalcluster.Name, key operationKey) (bool, error) {
	// Quick read-only check to see if the cache already exists
	d.opCachesMutex.RLock()
	cache, exists := d.opCaches[clusterName]
	d.opCachesMutex.RUnlock()

	if exists {
		return cache.isSupported(key), nil
	}

	// Doesn't exist - may need to create
	d.opCachesMutex.Lock()
	defer d.opCachesMutex.Unlock()

	// Check again, with the write lock held, to see if it exists. It's possible another goroutine set it in between
	// when we checked with the read lock held, and now.
	cache, exists = d.opCaches[clusterName]
	if exists {
		return cache.isSupported(key), nil
	}

	// Definitely doesn't exist - need to create it.
	cache = &operationNotSupportedCache{
		m: make(map[operationKey]bool),
	}
	d.opCaches[clusterName] = cache

	d.initOpCache(ctx, clusterName)

	return cache.isSupported(key), nil
}

// updateNamespaceFunc is a function that makes an update to a namespace
type updateNamespaceFunc func(ctx context.Context, namespace *v1.Namespace) (*v1.Namespace, error)

// retryOnConflictError retries the specified fn if there was a conflict error
// it will return an error if the UID for an object changes across retry operations.
// TODO RetryOnConflict should be a generic concept in client code
func (d *namespacedResourcesDeleter) retryOnConflictError(ctx context.Context, namespace *v1.Namespace, fn updateNamespaceFunc) (result *v1.Namespace, err error) {
	latestNamespace := namespace
	for {
		result, err = fn(ctx, latestNamespace)
		if err == nil {
			return result, nil
		}
		if !errors.IsConflict(err) {
			return nil, err
		}
		prevNamespace := latestNamespace
		latestNamespace, err = d.nsClient.Cluster(logicalcluster.From(latestNamespace).Path()).Get(ctx, latestNamespace.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if prevNamespace.UID != latestNamespace.UID {
			return nil, fmt.Errorf("namespace uid has changed across retries")
		}
	}
}

// updateNamespaceStatusFunc will verify that the status of the namespace is correct
func (d *namespacedResourcesDeleter) updateNamespaceStatusFunc(ctx context.Context, namespace *v1.Namespace) (*v1.Namespace, error) {
	if namespace.DeletionTimestamp.IsZero() || namespace.Status.Phase == v1.NamespaceTerminating {
		return namespace, nil
	}
	newNamespace := namespace.DeepCopy()
	newNamespace.Status.Phase = v1.NamespaceTerminating
	return d.nsClient.Cluster(logicalcluster.From(namespace).Path()).UpdateStatus(ctx, newNamespace, metav1.UpdateOptions{})
}

// finalized returns true if the namespace.Spec.Finalizers is an empty list
func finalized(namespace *v1.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalizeNamespace removes the specified finalizerToken and finalizes the namespace
func (d *namespacedResourcesDeleter) finalizeNamespace(ctx context.Context, namespace *v1.Namespace) (*v1.Namespace, error) {
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
	namespace, err := d.nsClient.Cluster(logicalcluster.From(namespace).Path()).Finalize(ctx, &namespaceFinalize, metav1.UpdateOptions{})
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
func (d *namespacedResourcesDeleter) deleteCollection(ctx context.Context, clusterName logicalcluster.Name, gvr schema.GroupVersionResource, namespace string) (bool, error) {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Namespace controller - deleteCollection", "namespace", namespace, "resource", gvr)

	key := operationKey{operation: operationDeleteCollection, gvr: gvr}
	if supported, err := d.isSupported(ctx, clusterName, key); err != nil {
		return false, err
	} else if !supported {
		logger.V(5).Info("namespace controller - deleteCollection ignored since not supported - namespace", "namespace", namespace, "resource", gvr)
		return false, nil
	}

	// namespace controller does not want the garbage collector to insert the orphan finalizer since it calls
	// resource deletions generically.  it will ensure all resources in the namespace are purged prior to releasing
	// namespace itself.
	background := metav1.DeletePropagationBackground
	opts := metav1.DeleteOptions{PropagationPolicy: &background}
	err := d.metadataClient.Cluster(clusterName.Path()).Resource(gvr).Namespace(namespace).DeleteCollection(ctx, opts, metav1.ListOptions{})

	if err == nil {
		return true, nil
	}

	// this is strange, but we need to special case for both MethodNotSupported and NotFound errors
	// TODO: https://github.com/kubernetes/kubernetes/issues/22413
	// we have a resource returned in the discovery API that supports no top-level verbs:
	//  /apis/extensions/v1beta1/namespaces/default/replicationcontrollers
	// when working with this resource type, we will get a literal not found error rather than expected method not supported
	if errors.IsMethodNotSupported(err) || errors.IsNotFound(err) {
		logger.V(5).Info("Namespace controller - deleteCollection not supported", "namespace", namespace, "resource", gvr)
		return false, nil
	}

	logger.V(5).Info("Namespace controller - deleteCollection unexpected error", "namespace", namespace, "resource", gvr, "err", err)
	return true, err
}

// listCollection will list the items in the specified namespace
// it returns the following:
//
//	the list of items in the collection (if found)
//	a boolean if the operation is supported
//	an error if the operation is supported but could not be completed.
func (d *namespacedResourcesDeleter) listCollection(ctx context.Context, clusterName logicalcluster.Name, gvr schema.GroupVersionResource, namespace string) (*metav1.PartialObjectMetadataList, bool, error) {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Namespace controller - listCollection", "namespace", namespace, "resource", gvr)

	key := operationKey{operation: operationList, gvr: gvr}
	if supported, err := d.isSupported(ctx, clusterName, key); err != nil {
		logger.V(5).Info("Namespace controller - listCollection ignored since not supported", "namespace", namespace, "resource", gvr)
		return nil, false, err
	} else if !supported {
		logger.V(5).Info("namespace controller - listCollection ignored since not supported - namespace", "namespace", namespace, "resource", gvr)
		return nil, false, nil
	}

	partialList, err := d.metadataClient.Cluster(clusterName.Path()).Resource(gvr).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		return partialList, true, nil
	}

	// this is strange, but we need to special case for both MethodNotSupported and NotFound errors
	// TODO: https://github.com/kubernetes/kubernetes/issues/22413
	// we have a resource returned in the discovery API that supports no top-level verbs:
	//  /apis/extensions/v1beta1/namespaces/default/replicationcontrollers
	// when working with this resource type, we will get a literal not found error rather than expected method not supported
	if errors.IsMethodNotSupported(err) || errors.IsNotFound(err) {
		logger.V(5).Info("Namespace controller - listCollection not supported", "namespace", namespace, "resource", gvr)
		return nil, false, nil
	}

	return nil, true, err
}

// deleteEachItem is a helper function that will list the collection of resources and delete each item 1 by 1.
func (d *namespacedResourcesDeleter) deleteEachItem(ctx context.Context, clusterName logicalcluster.Name, gvr schema.GroupVersionResource, namespace string) error {
	klog.FromContext(ctx).V(5).Info("Namespace controller - deleteEachItem", "namespace", namespace, "resource", gvr)

	partialList, listSupported, err := d.listCollection(ctx, clusterName, gvr, namespace)
	if err != nil {
		return err
	}
	if !listSupported {
		return nil
	}
	for _, item := range partialList.Items {
		background := metav1.DeletePropagationBackground
		opts := metav1.DeleteOptions{PropagationPolicy: &background}
		if err = d.metadataClient.Cluster(clusterName.Path()).Resource(gvr).Namespace(namespace).Delete(ctx, item.GetName(), opts); err != nil && !errors.IsNotFound(err) && !errors.IsMethodNotSupported(err) {
			return err
		}
	}
	return nil
}

type gvrDeletionMetadata struct {
	// finalizerEstimateSeconds is an estimate of how much longer to wait.  zero means that no estimate has made and does not
	// mean that all content has been removed.
	finalizerEstimateSeconds int64
	// numRemaining is how many instances of the gvr remain
	numRemaining int
	// finalizersToNumRemaining maps finalizers to how many resources are stuck on them
	finalizersToNumRemaining map[string]int
}

// deleteAllContentForGroupVersionResource will use the dynamic client to delete each resource identified in gvr.
// It returns an estimate of the time remaining before the remaining resources are deleted.
// If estimate > 0, not all resources are guaranteed to be gone.
func (d *namespacedResourcesDeleter) deleteAllContentForGroupVersionResource(
	ctx context.Context,
	clusterName logicalcluster.Name,
	gvr schema.GroupVersionResource, namespace string,
	namespaceDeletedAt metav1.Time) (gvrDeletionMetadata, error) {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource", "namespace", namespace, "resource", gvr)

	// estimate how long it will take for the resource to be deleted (needed for objects that support graceful delete)
	estimate, err := d.estimateGracefulTermination(ctx, clusterName, gvr, namespace, namespaceDeletedAt)
	if err != nil {
		logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource - unable to estimate", "namespace", namespace, "resource", gvr, "err", err)
		return gvrDeletionMetadata{}, err
	}
	logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource - estimate", "namespace", namespace, "resource", gvr, "estimate", estimate)

	// first try to delete the entire collection
	deleteCollectionSupported, err := d.deleteCollection(ctx, clusterName, gvr, namespace)
	if err != nil {
		return gvrDeletionMetadata{finalizerEstimateSeconds: estimate}, err
	}

	// delete collection was not supported, so we list and delete each item...
	if !deleteCollectionSupported {
		err = d.deleteEachItem(ctx, clusterName, gvr, namespace)
		if err != nil {
			return gvrDeletionMetadata{finalizerEstimateSeconds: estimate}, err
		}
	}

	// verify there are no more remaining items
	// it is not an error condition for there to be remaining items if local estimate is non-zero
	logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource - checking for no more items in namespace", "namespace", namespace, "resource", gvr)
	unstructuredList, listSupported, err := d.listCollection(ctx, clusterName, gvr, namespace)
	if err != nil {
		logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource - error verifying no items in namespace", "namespace", namespace, "resource", gvr, "err", err)
		return gvrDeletionMetadata{finalizerEstimateSeconds: estimate}, err
	}
	if !listSupported {
		return gvrDeletionMetadata{finalizerEstimateSeconds: estimate}, nil
	}
	logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource - items remaining", "namespace", namespace, "resource", gvr, "items", len(unstructuredList.Items))
	if len(unstructuredList.Items) == 0 {
		// we're done
		return gvrDeletionMetadata{finalizerEstimateSeconds: 0, numRemaining: 0}, nil
	}

	// use the list to find the finalizers
	finalizersToNumRemaining := map[string]int{}
	for _, item := range unstructuredList.Items {
		for _, finalizer := range item.GetFinalizers() {
			finalizersToNumRemaining[finalizer] = finalizersToNumRemaining[finalizer] + 1
		}
	}

	if estimate != int64(0) {
		logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource - estimate is present", "namespace", namespace, "resource", gvr, "finalizers", finalizersToNumRemaining)
		return gvrDeletionMetadata{
			finalizerEstimateSeconds: estimate,
			numRemaining:             len(unstructuredList.Items),
			finalizersToNumRemaining: finalizersToNumRemaining,
		}, nil
	}

	// if any item has a finalizer, we treat that as a normal condition, and use a default estimation to allow for GC to complete.
	if len(finalizersToNumRemaining) > 0 {
		logger.V(5).Info("Namespace controller - deleteAllContentForGroupVersionResource - items remaining with finalizers", "namespace", namespace, "resource", gvr, "finalizers", finalizersToNumRemaining)
		return gvrDeletionMetadata{
			finalizerEstimateSeconds: finalizerEstimateSeconds,
			numRemaining:             len(unstructuredList.Items),
			finalizersToNumRemaining: finalizersToNumRemaining,
		}, nil
	}

	// nothing reported a finalizer, so something was unexpected as it should have been deleted.
	return gvrDeletionMetadata{
		finalizerEstimateSeconds: estimate,
		numRemaining:             len(unstructuredList.Items),
	}, fmt.Errorf("unexpected items still remain in namespace: %s for gvr: %v", namespace, gvr)
}

type allGVRDeletionMetadata struct {
	// gvrToNumRemaining is how many instances of the gvr remain
	gvrToNumRemaining map[schema.GroupVersionResource]int
	// finalizersToNumRemaining maps finalizers to how many resources are stuck on them
	finalizersToNumRemaining map[string]int
}

// deleteAllContent will use the dynamic client to delete each resource identified in groupVersionResources.
// It returns an estimate of the time remaining before the remaining resources are deleted.
// If estimate > 0, not all resources are guaranteed to be gone.
func (d *namespacedResourcesDeleter) deleteAllContent(ctx context.Context, ns *v1.Namespace) (int64, error) {
	namespace := ns.Name
	namespaceDeletedAt := *ns.DeletionTimestamp
	var errs []error
	conditionUpdater := namespaceConditionUpdater{}
	estimate := int64(0)
	logger := klog.FromContext(ctx)
	logger.V(4).Info("namespace controller - deleteAllContent", "namespace", namespace)

	resources, err := d.discoverResourcesFn(logicalcluster.From(ns).Path())
	if err != nil {
		// discovery errors are not fatal.  We often have some set of resources we can operate against even if we don't have a complete list
		errs = append(errs, err)
		conditionUpdater.ProcessDiscoverResourcesErr(err)
	}
	// TODO(sttts): get rid of opCache and pass the verbs (especially "deletecollection") down into the deleter
	deletableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"delete"}}, resources)
	groupVersionResources, err := discovery.GroupVersionResources(deletableResources)
	if err != nil {
		// discovery errors are not fatal.  We often have some set of resources we can operate against even if we don't have a complete list
		errs = append(errs, err)
		conditionUpdater.ProcessGroupVersionErr(err)
	}

	numRemainingTotals := allGVRDeletionMetadata{
		gvrToNumRemaining:        map[schema.GroupVersionResource]int{},
		finalizersToNumRemaining: map[string]int{},
	}
	for gvr := range groupVersionResources {
		gvrDeletionMetadata, err := d.deleteAllContentForGroupVersionResource(ctx, logicalcluster.From(ns), gvr, namespace, namespaceDeletedAt)
		if err != nil {
			// If there is an error, hold on to it but proceed with all the remaining
			// groupVersionResources.
			errs = append(errs, err)
			conditionUpdater.ProcessDeleteContentErr(err)
		}
		if gvrDeletionMetadata.finalizerEstimateSeconds > estimate {
			estimate = gvrDeletionMetadata.finalizerEstimateSeconds
		}
		if gvrDeletionMetadata.numRemaining > 0 {
			numRemainingTotals.gvrToNumRemaining[gvr] = gvrDeletionMetadata.numRemaining
			for finalizer, numRemaining := range gvrDeletionMetadata.finalizersToNumRemaining {
				if numRemaining == 0 {
					continue
				}
				numRemainingTotals.finalizersToNumRemaining[finalizer] = numRemainingTotals.finalizersToNumRemaining[finalizer] + numRemaining
			}
		}
	}
	conditionUpdater.ProcessContentTotals(numRemainingTotals)

	// we always want to update the conditions because if we have set a condition to "it worked" after it was previously, "it didn't work",
	// we need to reflect that information.  Recall that additional finalizers can be set on namespaces, so this finalizer may clear itself and
	// NOT remove the resource instance.
	if hasChanged := conditionUpdater.Update(ns); hasChanged {
		if _, err = d.nsClient.Cluster(logicalcluster.From(ns).Path()).UpdateStatus(ctx, ns, metav1.UpdateOptions{}); err != nil {
			utilruntime.HandleError(fmt.Errorf("couldn't update status condition for namespace %q: %v", namespace, err))
		}
	}

	// if len(errs)==0, NewAggregate returns nil.
	err = utilerrors.NewAggregate(errs)
	logger.V(4).Info("namespace controller - deleteAllContent", "namespace", namespace, "estimate", estimate, "err", err)
	return estimate, err
}

// estimateGracefulTermination will estimate the graceful termination required for the specific entity in the namespace
func (d *namespacedResourcesDeleter) estimateGracefulTermination(ctx context.Context, clusterName logicalcluster.Name, gvr schema.GroupVersionResource, ns string, namespaceDeletedAt metav1.Time) (int64, error) {
	groupResource := gvr.GroupResource()
	klog.FromContext(ctx).V(5).Info("Namespace controller - estimateGracefulTermination", "group", groupResource.Group, "resource", groupResource.Resource)
	estimate := int64(0)
	var err error
	switch groupResource {
	case schema.GroupResource{Group: "", Resource: "pods"}:
		estimate, err = d.estimateGracefulTerminationForPods(ctx, clusterName, ns)
	}
	if err != nil {
		return 0, err
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
func (d *namespacedResourcesDeleter) estimateGracefulTerminationForPods(ctx context.Context, clusterName logicalcluster.Name, ns string) (int64, error) {
	klog.FromContext(ctx).V(5).Info("Namespace controller - estimateGracefulTerminationForPods", "namespace", ns)
	estimate := int64(0)
	podsGetter := d.podsGetter
	if podsGetter == nil || reflect.ValueOf(podsGetter).IsNil() {
		return 0, fmt.Errorf("unexpected: podsGetter is nil. Cannot estimate grace period seconds for pods")
	}
	items, err := podsGetter.Pods().Cluster(clusterName.Path()).Namespace(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		return 0, err
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
