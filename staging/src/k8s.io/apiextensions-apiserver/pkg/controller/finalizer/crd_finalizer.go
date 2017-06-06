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
	"fmt"
	"reflect"
	"time"

	"github.com/golang/glog"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/conversion"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	client "k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
)

var cloner = conversion.NewCloner()

// CRDFinalizer is a controller that finalizes the CRD by deleting all the CRs associated with it.
type CRDFinalizer struct {
	crdClient      client.CustomResourceDefinitionsGetter
	crClientGetter CRClientGetter

	crdLister listers.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.RateLimitingInterface
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
	GetCustomResourceListerCollectionDeleter(crd *apiextensions.CustomResourceDefinition) ListerCollectionDeleter
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
		queue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "CustomResourceDefinition-CRDFinalizer"),
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
	if cachedCRD.DeletionTimestamp.IsZero() || !apiextensions.CRDHasFinalizer(cachedCRD, apiextensions.CustomResourceCleanupFinalizer) {
		return nil
	}

	crd := &apiextensions.CustomResourceDefinition{}
	if err := apiextensions.DeepCopy_apiextensions_CustomResourceDefinition(cachedCRD, crd, cloner); err != nil {
		return err
	}

	// update the status condition.  This cleanup could take a while.
	apiextensions.SetCRDCondition(crd, apiextensions.CustomResourceDefinitionCondition{
		Type:    apiextensions.Terminating,
		Status:  apiextensions.ConditionTrue,
		Reason:  "InstanceDeletionInProgress",
		Message: "CustomResource deletion is in progress",
	})
	crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
	if err != nil {
		return err
	}

	// Now we can start deleting items.  We should use the REST API to ensure that all normal admission runs.
	// Since we control the endpoints, we know that delete collection works. No need to delete if not established.
	if apiextensions.IsCRDConditionTrue(crd, apiextensions.Established) {
		cond, deleteErr := c.deleteInstances(crd)
		apiextensions.SetCRDCondition(crd, cond)
		if deleteErr != nil {
			crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
			if err != nil {
				utilruntime.HandleError(err)
			}
			return deleteErr
		}
	} else {
		apiextensions.SetCRDCondition(crd, apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionFalse,
			Reason:  "NeverEstablished",
			Message: "resource was never established",
		})
	}

	apiextensions.CRDRemoveFinalizer(crd, apiextensions.CustomResourceCleanupFinalizer)
	crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
	if err != nil {
		return err
	}

	// and now issue another delete, which should clean it all up if no finalizers remain or no-op if they do
	return c.crdClient.CustomResourceDefinitions().Delete(crd.Name, nil)
}

func (c *CRDFinalizer) deleteInstances(crd *apiextensions.CustomResourceDefinition) (apiextensions.CustomResourceDefinitionCondition, error) {
	// Now we can start deleting items. While it would be ideal to use a REST API client, doing so
	// could incorrectly delete a ThirdPartyResource with the same URL as the CustomResource, so we go
	// directly to the storage instead. Since we control the storage, we know that delete collection works.
	crClient := c.crClientGetter.GetCustomResourceListerCollectionDeleter(crd)
	if crClient == nil {
		err := fmt.Errorf("unable to find a custom resource client for %s.%s", crd.Status.AcceptedNames.Plural, crd.Spec.Group)
		return apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionTrue,
			Reason:  "InstanceDeletionFailed",
			Message: fmt.Sprintf("could not list instances: %v", err),
		}, err
	}

	ctx := genericapirequest.NewContext()
	allResources, err := crClient.List(ctx, nil)
	if err != nil {
		return apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionTrue,
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
		if _, err := crClient.DeleteCollection(nsCtx, nil, nil); err != nil {
			deleteErrors = append(deleteErrors, err)
			continue
		}
	}
	if deleteError := utilerrors.NewAggregate(deleteErrors); deleteError != nil {
		return apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionTrue,
			Reason:  "InstanceDeletionFailed",
			Message: fmt.Sprintf("could not issue all deletes: %v", deleteError),
		}, deleteError
	}

	// now we need to wait until all the resources are deleted.  Start with a simple poll before we do anything fancy.
	// TODO not all servers are synchronized on caches.  It is possible for a stale one to still be creating things.
	// Once we have a mechanism for servers to indicate their states, we should check that for concurrence.
	err = wait.PollImmediate(5*time.Second, 1*time.Minute, func() (bool, error) {
		listObj, err := crClient.List(ctx, nil)
		if err != nil {
			return false, err
		}
		if len(listObj.(*unstructured.UnstructuredList).Items) == 0 {
			return true, nil
		}
		glog.V(2).Infof("%s.%s waiting for %d items to be removed", crd.Status.AcceptedNames.Plural, crd.Spec.Group, len(listObj.(*unstructured.UnstructuredList).Items))
		return false, nil
	})
	if err != nil {
		return apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionTrue,
			Reason:  "InstanceDeletionCheck",
			Message: fmt.Sprintf("could not confirm zero CustomResources remaining: %v", err),
		}, err
	}
	return apiextensions.CustomResourceDefinitionCondition{
		Type:    apiextensions.Terminating,
		Status:  apiextensions.ConditionFalse,
		Reason:  "InstanceDeletionCompleted",
		Message: "removed all instances",
	}, nil
}

func (c *CRDFinalizer) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	glog.Infof("Starting CRDFinalizer")
	defer glog.Infof("Shutting down CRDFinalizer")

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

	err := c.syncFn(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *CRDFinalizer) enqueue(obj *apiextensions.CustomResourceDefinition) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", obj, err))
		return
	}

	c.queue.Add(key)
}

func (c *CRDFinalizer) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	// only queue deleted things
	if !castObj.DeletionTimestamp.IsZero() && apiextensions.CRDHasFinalizer(castObj, apiextensions.CustomResourceCleanupFinalizer) {
		c.enqueue(castObj)
	}
}

func (c *CRDFinalizer) updateCustomResourceDefinition(oldObj, newObj interface{}) {
	oldCRD := oldObj.(*apiextensions.CustomResourceDefinition)
	newCRD := newObj.(*apiextensions.CustomResourceDefinition)
	// only queue deleted things that haven't been finalized by us
	if newCRD.DeletionTimestamp.IsZero() || !apiextensions.CRDHasFinalizer(newCRD, apiextensions.CustomResourceCleanupFinalizer) {
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
	oldCopy := &apiextensions.CustomResourceDefinition{}
	if err := apiextensions.DeepCopy_apiextensions_CustomResourceDefinition(oldCRD, oldCopy, cloner); err != nil {
		utilruntime.HandleError(err)
		c.enqueue(newCRD)
		return
	}
	newCopy := &apiextensions.CustomResourceDefinition{}
	if err := apiextensions.DeepCopy_apiextensions_CustomResourceDefinition(newCRD, newCopy, cloner); err != nil {
		utilruntime.HandleError(err)
		c.enqueue(newCRD)
		return
	}
	oldCopy.ResourceVersion = ""
	newCopy.ResourceVersion = ""
	apiextensions.RemoveCRDCondition(oldCopy, apiextensions.Terminating)
	apiextensions.RemoveCRDCondition(newCopy, apiextensions.Terminating)

	if !reflect.DeepEqual(oldCopy, newCopy) {
		c.enqueue(newCRD)
	}
}
