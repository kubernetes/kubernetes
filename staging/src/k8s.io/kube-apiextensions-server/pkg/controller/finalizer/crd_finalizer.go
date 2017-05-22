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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
	client "k8s.io/kube-apiextensions-server/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion"
	informers "k8s.io/kube-apiextensions-server/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/kube-apiextensions-server/pkg/client/listers/apiextensions/internalversion"
)

var cloner = conversion.NewCloner()

// This controller finalizes the CRD by deleting all the CRs associated with it.
type CRDFinalizer struct {
	crdClient client.CustomResourceDefinitionsGetter
	// clientPool is a dynamic client used to delete the individual instances
	clientPool dynamic.ClientPool

	crdLister listers.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.RateLimitingInterface
}

func NewCRDFinalizer(
	crdInformer informers.CustomResourceDefinitionInformer,
	crdClient client.CustomResourceDefinitionsGetter,
	clientPool dynamic.ClientPool,
) *CRDFinalizer {
	c := &CRDFinalizer{
		crdClient:  crdClient,
		clientPool: clientPool,
		crdLister:  crdInformer.Lister(),
		crdSynced:  crdInformer.Informer().HasSynced,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "CustomResourceDefinition-CRDFinalizer"),
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

	// Its possible for a naming conflict to have removed this resource from the API after instances were created.
	// For now we will cowardly stop finalizing.  If we don't go through the REST API, weird things may happen:
	// no audit trail, no admission checks or side effects, finalization would probably still work but defaulting
	// would be missed.  It would be a mess.
	// This requires human intervention to solve, update status so they have a reason.
	// TODO split coreNamesAccepted from extendedNamesAccepted.  If coreNames were accepted, then we have something to cleanup
	// and the endpoint is serviceable.  if they aren't, then there's nothing to cleanup.
	if !apiextensions.IsCRDConditionFalse(crd, apiextensions.NameConflict) {
		apiextensions.SetCRDCondition(crd, apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionTrue,
			Reason:  "InstanceDeletionStuck",
			Message: fmt.Sprintf("cannot proceed with deletion because of %v condition", apiextensions.NameConflict),
		})
		crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
		if err != nil {
			return err
		}
		return fmt.Errorf("cannot proceed with deletion because of %v condition", apiextensions.NameConflict)
	}

	// Now we can start deleting items.  We should use the REST API to ensure that all normal admission runs.
	// Since we control the endpoints, we know that delete collection works.
	crClient, err := c.clientPool.ClientForGroupVersionResource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Version, Resource: crd.Status.AcceptedNames.Plural})
	if err != nil {
		return err
	}
	crAPIResource := &metav1.APIResource{
		Name:         crd.Status.AcceptedNames.Plural,
		SingularName: crd.Status.AcceptedNames.Singular,
		Namespaced:   crd.Spec.Scope == apiextensions.NamespaceScoped,
		Kind:         crd.Status.AcceptedNames.Kind,
		Verbs:        metav1.Verbs([]string{"deletecollection", "list"}),
		ShortNames:   crd.Status.AcceptedNames.ShortNames,
	}
	crResourceClient := crClient.Resource(crAPIResource, "" /* namespace all */)
	allResources, err := crResourceClient.List(metav1.ListOptions{})
	if err != nil {
		return err
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
		if err := crClient.Resource(crAPIResource, metadata.GetNamespace()).DeleteCollection(nil, metav1.ListOptions{}); err != nil {
			deleteErrors = append(deleteErrors, err)
			continue
		}
	}
	if deleteError := utilerrors.NewAggregate(deleteErrors); deleteError != nil {
		apiextensions.SetCRDCondition(crd, apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionTrue,
			Reason:  "InstanceDeletionFailed",
			Message: fmt.Sprintf("could not issue all deletes: %v", deleteError),
		})
		crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
		if err != nil {
			utilruntime.HandleError(err)
		}
		return deleteError
	}

	// now we need to wait until all the resources are deleted.  Start with a simple poll before we do anything fancy.
	// TODO not all servers are synchronized on caches.  It is possible for a stale one to still be creating things.
	// Once we have a mechanism for servers to indicate their states, we should check that for concurrence.
	listErr := wait.PollImmediate(5*time.Second, 1*time.Minute, func() (bool, error) {
		listObj, err := crResourceClient.List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(listObj.(*unstructured.UnstructuredList).Items) == 0 {
			return true, nil
		}
		glog.V(2).Infof("%s.%s waiting for %d items to be removed", crd.Status.AcceptedNames.Plural, crd.Spec.Group, len(listObj.(*unstructured.UnstructuredList).Items))
		return false, nil
	})
	if listErr != nil {
		apiextensions.SetCRDCondition(crd, apiextensions.CustomResourceDefinitionCondition{
			Type:    apiextensions.Terminating,
			Status:  apiextensions.ConditionTrue,
			Reason:  "InstanceDeletionCheck",
			Message: fmt.Sprintf("could not confirm zero CustomResources remaining: %v", listErr),
		})
		crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
		if err != nil {
			utilruntime.HandleError(err)
		}
		return listErr
	}

	apiextensions.SetCRDCondition(crd, apiextensions.CustomResourceDefinitionCondition{
		Type:    apiextensions.Terminating,
		Status:  apiextensions.ConditionFalse,
		Reason:  "InstanceDeletionCompleted",
		Message: "removed all instances",
	})
	apiextensions.CRDRemoveFinalizer(crd, apiextensions.CustomResourceCleanupFinalizer)
	crd, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
	if err != nil {
		return err
	}

	// and now issue another delete, which should clean it all up if no finalizers remain or no-op if they do
	return c.crdClient.CustomResourceDefinitions().Delete(crd.Name, nil)
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
