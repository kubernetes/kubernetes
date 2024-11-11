/*
Copyright 2017 The Kubernetes Authors.

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

package finalizer

import (
	"context"
	"fmt"
	"reflect"
	"time"

	"k8s.io/klog/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	client "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1"
)

// OverlappingBuiltInResources returns the set of built-in group/resources that are persisted
// in storage paths that overlap with CRD storage paths, and should not be deleted
// by this controller if an associated CRD is deleted.
func OverlappingBuiltInResources() map[schema.GroupResource]bool {
	return map[schema.GroupResource]bool{
		{Group: "apiregistration.k8s.io", Resource: "apiservices"}:             true,
		{Group: "apiextensions.k8s.io", Resource: "customresourcedefinitions"}: true,
	}
}

// CRDFinalizer is a controller that finalizes the CRD by deleting all the CRs associated with it.
type CRDFinalizer struct {
	crdClient      client.CustomResourceDefinitionsGetter
	crClientGetter CRClientGetter

	crdLister listers.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.TypedRateLimitingInterface[string]
}

// ListerCollectionDeleter combines rest.Lister and rest.CollectionDeleter.
type ListerCollectionDeleter interface {
	rest.Lister
	rest.CollectionDeleter
}

// CRClientGetter knows how to get a ListerCollectionDeleter for a given CRD UID.
type CRClientGetter interface {
	// GetCustomResourceListerCollectionDeleter gets the ListerCollectionDeleter for the given CRD
	// UID.
	GetCustomResourceListerCollectionDeleter(crd *apiextensionsv1.CustomResourceDefinition) (ListerCollectionDeleter, error)
}

// NewCRDFinalizer creates a new CRDFinalizer.
func NewCRDFinalizer(
	crdInformer informers.CustomResourceDefinitionInformer,
	crdClient client.CustomResourceDefinitionsGetter,
	crClientGetter CRClientGetter,
) *CRDFinalizer {
	c := &CRDFinalizer{
		crdClient:      crdClient,
		crdLister:      crdInformer.Lister(),
		crdSynced:      crdInformer.Informer().HasSynced,
		crClientGetter: crClientGetter,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "crd_finalizer"},
		),
	}

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
	})

	c.syncFn = c.sync

	return c
}

func (c *CRDFinalizer) sync(key string) error {
	cachedCRD, err := c.crdLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	// no work to do
	if cachedCRD.DeletionTimestamp.IsZero() || !apiextensionshelpers.CRDHasFinalizer(cachedCRD, apiextensionsv1.CustomResourceCleanupFinalizer) {
		return nil
	}

	crd := cachedCRD.DeepCopy()

	// update the status condition.  This cleanup could take a while.
	apiextensionshelpers.SetCRDCondition(crd, apiextensionsv1.CustomResourceDefinitionCondition{
		Type:    apiextensionsv1.Terminating,
		Status:  apiextensionsv1.ConditionTrue,
		Reason:  "InstanceDeletionInProgress",
		Message: "CustomResource deletion is in progress",
	})
	crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(context.TODO(), crd, metav1.UpdateOptions{})
	if apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
		// deleted or changed in the meantime, we'll get called again
		return nil
	}
	if err != nil {
		return err
	}

	// Now we can start deleting items.  We should use the REST API to ensure that all normal admission runs.
	// Since we control the endpoints, we know that delete collection works. No need to delete if not established.
	if OverlappingBuiltInResources()[schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Spec.Names.Plural}] {
		// Skip deletion, explain why, and proceed to remove the finalizer and delete the CRD
		apiextensionshelpers.SetCRDCondition(crd, apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Terminating,
			Status:  apiextensionsv1.ConditionFalse,
			Reason:  "OverlappingBuiltInResource",
			Message: "instances overlap with built-in resources in storage",
		})
	} else if apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
		cond, deleteErr := c.deleteInstances(crd)
		apiextensionshelpers.SetCRDCondition(crd, cond)
		if deleteErr != nil {
			if _, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(context.TODO(), crd, metav1.UpdateOptions{}); err != nil {
				utilruntime.HandleError(err)
			}
			return deleteErr
		}
	} else {
		apiextensionshelpers.SetCRDCondition(crd, apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Terminating,
			Status:  apiextensionsv1.ConditionFalse,
			Reason:  "NeverEstablished",
			Message: "resource was never established",
		})
	}

	apiextensionshelpers.CRDRemoveFinalizer(crd, apiextensionsv1.CustomResourceCleanupFinalizer)
	_, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(context.TODO(), crd, metav1.UpdateOptions{})
	if apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
		// deleted or changed in the meantime, we'll get called again
		return nil
	}
	return err
}

func (c *CRDFinalizer) deleteInstances(crd *apiextensionsv1.CustomResourceDefinition) (apiextensionsv1.CustomResourceDefinitionCondition, error) {
	// Now we can start deleting items. While it would be ideal to use a REST API client, doing so
	// could incorrectly delete a ThirdPartyResource with the same URL as the CustomResource, so we go
	// directly to the storage instead. Since we control the storage, we know that delete collection works.
	crClient, err := c.crClientGetter.GetCustomResourceListerCollectionDeleter(crd)
	if err != nil {
		err = fmt.Errorf("unable to find a custom resource client for %s.%s: %v", crd.Status.AcceptedNames.Plural, crd.Spec.Group, err)
		return apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Terminating,
			Status:  apiextensionsv1.ConditionTrue,
			Reason:  "InstanceDeletionFailed",
			Message: fmt.Sprintf("could not list instances: %v", err),
		}, err
	}

	ctx := genericapirequest.NewContext()
	allResources, err := crClient.List(ctx, nil)
	if err != nil {
		return apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Terminating,
			Status:  apiextensionsv1.ConditionTrue,
			Reason:  "InstanceDeletionFailed",
			Message: fmt.Sprintf("could not list instances: %v", err),
		}, err
	}

	deletedNamespaces := sets.String{}
	deleteErrors := []error{}
	for _, item := range allResources.(*unstructured.UnstructuredList).Items {
		metadata, err := meta.Accessor(&item)
		if err != nil {
			utilruntime.HandleError(err)
			continue
		}
		if deletedNamespaces.Has(metadata.GetNamespace()) {
			continue
		}
		// don't retry deleting the same namespace
		deletedNamespaces.Insert(metadata.GetNamespace())
		nsCtx := genericapirequest.WithNamespace(ctx, metadata.GetNamespace())
		if _, err := crClient.DeleteCollection(nsCtx, rest.ValidateAllObjectFunc, nil, nil); err != nil {
			deleteErrors = append(deleteErrors, err)
			continue
		}
	}
	if deleteError := utilerrors.NewAggregate(deleteErrors); deleteError != nil {
		return apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Terminating,
			Status:  apiextensionsv1.ConditionTrue,
			Reason:  "InstanceDeletionFailed",
			Message: fmt.Sprintf("could not issue all deletes: %v", deleteError),
		}, deleteError
	}

	// now we need to wait until all the resources are deleted.  Start with a simple poll before we do anything fancy.
	// TODO not all servers are synchronized on caches.  It is possible for a stale one to still be creating things.
	// Once we have a mechanism for servers to indicate their states, we should check that for concurrence.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		listObj, err := crClient.List(ctx, nil)
		if err != nil {
			return false, err
		}
		if len(listObj.(*unstructured.UnstructuredList).Items) == 0 {
			return true, nil
		}
		klog.V(2).Infof("%s.%s waiting for %d items to be removed", crd.Status.AcceptedNames.Plural, crd.Spec.Group, len(listObj.(*unstructured.UnstructuredList).Items))
		return false, nil
	})
	if err != nil {
		return apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Terminating,
			Status:  apiextensionsv1.ConditionTrue,
			Reason:  "InstanceDeletionCheck",
			Message: fmt.Sprintf("could not confirm zero CustomResources remaining: %v", err),
		}, err
	}
	return apiextensionsv1.CustomResourceDefinitionCondition{
		Type:    apiextensionsv1.Terminating,
		Status:  apiextensionsv1.ConditionFalse,
		Reason:  "InstanceDeletionCompleted",
		Message: "removed all instances",
	}, nil
}

func (c *CRDFinalizer) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Info("Starting CRDFinalizer")
	defer klog.Info("Shutting down CRDFinalizer")

	if !cache.WaitForCacheSync(stopCh, c.crdSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *CRDFinalizer) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *CRDFinalizer) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncFn(key)
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *CRDFinalizer) enqueue(obj *apiextensionsv1.CustomResourceDefinition) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}

	c.queue.Add(key)
}

func (c *CRDFinalizer) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensionsv1.CustomResourceDefinition)
	// only queue deleted things
	if !castObj.DeletionTimestamp.IsZero() && apiextensionshelpers.CRDHasFinalizer(castObj, apiextensionsv1.CustomResourceCleanupFinalizer) {
		c.enqueue(castObj)
	}
}

func (c *CRDFinalizer) updateCustomResourceDefinition(oldObj, newObj interface{}) {
	oldCRD := oldObj.(*apiextensionsv1.CustomResourceDefinition)
	newCRD := newObj.(*apiextensionsv1.CustomResourceDefinition)
	// only queue deleted things that haven't been finalized by us
	if newCRD.DeletionTimestamp.IsZero() || !apiextensionshelpers.CRDHasFinalizer(newCRD, apiextensionsv1.CustomResourceCleanupFinalizer) {
		return
	}

	// always requeue resyncs just in case
	if oldCRD.ResourceVersion == newCRD.ResourceVersion {
		c.enqueue(newCRD)
		return
	}

	// If the only difference is in the terminating condition, then there's no reason to requeue here.  This controller
	// is likely to be the originator, so requeuing would hot-loop us.  Failures are requeued by the workqueue directly.
	// This is a low traffic and scale resource, so the copy is terrible.  It's not good, so better ideas
	// are welcome.
	oldCopy := oldCRD.DeepCopy()
	newCopy := newCRD.DeepCopy()
	oldCopy.ResourceVersion = ""
	newCopy.ResourceVersion = ""
	apiextensionshelpers.RemoveCRDCondition(oldCopy, apiextensionsv1.Terminating)
	apiextensionshelpers.RemoveCRDCondition(newCopy, apiextensionsv1.Terminating)

	if !reflect.DeepEqual(oldCopy, newCopy) {
		c.enqueue(newCRD)
	}
}
