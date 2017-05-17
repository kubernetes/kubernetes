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

package status

import (
	"fmt"
	"reflect"
	"time"

	"github.com/golang/glog"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
	client "k8s.io/kube-apiextensions-server/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion"
	informers "k8s.io/kube-apiextensions-server/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/kube-apiextensions-server/pkg/client/listers/apiextensions/internalversion"
)

var cloner = conversion.NewCloner()

// This controller is reserving names. To avoid conflicts, be sure to run only one instance of the worker at a time.
// This could eventually be lifted, but starting simple.
type NamingConditionController struct {
	crdClient client.CustomResourceDefinitionsGetter

	crdLister listers.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced
	// crdMutationCache backs our lister and keeps track of committed updates to avoid racy
	// write/lookup cycles.  It's got 100 slots by default, so it unlikely to overrun
	// TODO to revisit this if naming conflicts are found to occur in the wild
	crdMutationCache cache.MutationCache

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.RateLimitingInterface
}

func NewNamingConditionController(
	crdInformer informers.CustomResourceDefinitionInformer,
	crdClient client.CustomResourceDefinitionsGetter,
) *NamingConditionController {
	c := &NamingConditionController{
		crdClient: crdClient,
		crdLister: crdInformer.Lister(),
		crdSynced: crdInformer.Informer().HasSynced,
		queue:     workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "CustomResourceDefinition-NamingConditionController"),
	}

	informerIndexer := crdInformer.Informer().GetIndexer()
	c.crdMutationCache = cache.NewIntegerResourceVersionMutationCache(informerIndexer, informerIndexer)

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
		DeleteFunc: c.deleteCustomResourceDefinition,
	})

	c.syncFn = c.sync

	return c
}

func (c *NamingConditionController) getAcceptedNamesForGroup(group string) (allResources sets.String, allKinds sets.String) {
	allResources = sets.String{}
	allKinds = sets.String{}

	list, err := c.crdLister.List(labels.Everything())
	if err != nil {
		panic(err)
	}

	for _, curr := range list {
		if curr.Spec.Group != group {
			continue
		}

		// for each item here, see if we have a mutation cache entry that is more recent
		// this makes sure that if we tight loop on update and run, our mutation cache will show
		// us the version of the objects we just updated to.
		item := curr
		obj, exists, err := c.crdMutationCache.GetByKey(curr.Name)
		if exists && err == nil {
			item = obj.(*apiextensions.CustomResourceDefinition)
		}

		allResources.Insert(item.Status.AcceptedNames.Plural)
		allResources.Insert(item.Status.AcceptedNames.Singular)
		allResources.Insert(item.Status.AcceptedNames.ShortNames...)

		allKinds.Insert(item.Status.AcceptedNames.Kind)
		allKinds.Insert(item.Status.AcceptedNames.ListKind)
	}

	return allResources, allKinds
}

func (c *NamingConditionController) calculateNames(in *apiextensions.CustomResourceDefinition) (apiextensions.CustomResourceDefinitionNames, apiextensions.CustomResourceDefinitionCondition) {
	// Get the names that have already been claimed
	allResources, allKinds := c.getAcceptedNamesForGroup(in.Spec.Group)

	condition := apiextensions.CustomResourceDefinitionCondition{
		Type:   apiextensions.NameConflict,
		Status: apiextensions.ConditionUnknown,
	}

	requestedNames := in.Spec.Names
	acceptedNames := in.Status.AcceptedNames
	newNames := in.Status.AcceptedNames

	// Check each name for mismatches.  If there's a mismatch between spec and status, then try to deconflict.
	// Continue on errors so that the status is the best match possible
	if err := equalToAcceptedOrFresh(requestedNames.Plural, acceptedNames.Plural, allResources); err != nil {
		condition.Status = apiextensions.ConditionTrue
		condition.Reason = "Plural"
		condition.Message = err.Error()
	} else {
		newNames.Plural = requestedNames.Plural
	}
	if err := equalToAcceptedOrFresh(requestedNames.Singular, acceptedNames.Singular, allResources); err != nil {
		condition.Status = apiextensions.ConditionTrue
		condition.Reason = "Singular"
		condition.Message = err.Error()
	} else {
		newNames.Singular = requestedNames.Singular
	}
	if !reflect.DeepEqual(requestedNames.ShortNames, acceptedNames.ShortNames) {
		errs := []error{}
		existingShortNames := sets.NewString(acceptedNames.ShortNames...)
		for _, shortName := range requestedNames.ShortNames {
			// if the shortname is already ours, then we're fine
			if existingShortNames.Has(shortName) {
				continue
			}
			if err := equalToAcceptedOrFresh(shortName, "", allResources); err != nil {
				errs = append(errs, err)
			}

		}
		if err := utilerrors.NewAggregate(errs); err != nil {
			condition.Status = apiextensions.ConditionTrue
			condition.Reason = "ShortNames"
			condition.Message = err.Error()
		} else {
			newNames.ShortNames = requestedNames.ShortNames
		}
	}

	if err := equalToAcceptedOrFresh(requestedNames.Kind, acceptedNames.Kind, allKinds); err != nil {
		condition.Status = apiextensions.ConditionTrue
		condition.Reason = "Kind"
		condition.Message = err.Error()
	} else {
		newNames.Kind = requestedNames.Kind
	}
	if err := equalToAcceptedOrFresh(requestedNames.ListKind, acceptedNames.ListKind, allKinds); err != nil {
		condition.Status = apiextensions.ConditionTrue
		condition.Reason = "ListKind"
		condition.Message = err.Error()
	} else {
		newNames.ListKind = requestedNames.ListKind
	}

	// if we haven't changed the condition, then our names must be good.
	if condition.Status == apiextensions.ConditionUnknown {
		condition.Status = apiextensions.ConditionFalse
		condition.Reason = "NoConflicts"
		condition.Message = "no conflicts found"
	}

	return newNames, condition
}

func equalToAcceptedOrFresh(requestedName, acceptedName string, usedNames sets.String) error {
	if requestedName == acceptedName {
		return nil
	}
	if !usedNames.Has(requestedName) {
		return nil
	}

	return fmt.Errorf("%q is already in use", requestedName)
}

func (c *NamingConditionController) sync(key string) error {
	inCustomResourceDefinition, err := c.crdLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	acceptedNames, namingCondition := c.calculateNames(inCustomResourceDefinition)
	// nothing to do if accepted names and NameConflict condition didn't change
	if reflect.DeepEqual(inCustomResourceDefinition.Status.AcceptedNames, acceptedNames) &&
		apiextensions.IsCRDConditionEquivalent(
			&namingCondition,
			apiextensions.FindCRDCondition(inCustomResourceDefinition, apiextensions.NameConflict)) {
		return nil
	}

	crd := &apiextensions.CustomResourceDefinition{}
	if err := apiextensions.DeepCopy_apiextensions_CustomResourceDefinition(inCustomResourceDefinition, crd, cloner); err != nil {
		return err
	}

	crd.Status.AcceptedNames = acceptedNames
	apiextensions.SetCRDCondition(crd, namingCondition)

	updatedObj, err := c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
	if err != nil {
		return err
	}

	// if the update was successful, go ahead and add the entry to the mutation cache
	c.crdMutationCache.Mutation(updatedObj)

	// we updated our status, so we may be releasing a name.  When this happens, we need to rekick everything in our group
	// if we fail to rekick, just return as normal.  We'll get everything on a resync
	list, err := c.crdLister.List(labels.Everything())
	if err != nil {
		return nil
	}
	for _, curr := range list {
		if curr.Spec.Group == crd.Spec.Group {
			c.queue.Add(curr.Name)
		}
	}

	return nil
}

func (c *NamingConditionController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	glog.Infof("Starting NamingConditionController")
	defer glog.Infof("Shutting down NamingConditionController")

	if !cache.WaitForCacheSync(stopCh, c.crdSynced) {
		return
	}

	// only start one worker thread since its a slow moving API and the naming conflict resolution bits aren't thread-safe
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *NamingConditionController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *NamingConditionController) processNextWorkItem() bool {
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

func (c *NamingConditionController) enqueue(obj *apiextensions.CustomResourceDefinition) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", obj, err))
		return
	}

	c.queue.Add(key)
}

func (c *NamingConditionController) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	glog.V(4).Infof("Adding %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *NamingConditionController) updateCustomResourceDefinition(obj, _ interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	glog.V(4).Infof("Updating %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *NamingConditionController) deleteCustomResourceDefinition(obj interface{}) {
	castObj, ok := obj.(*apiextensions.CustomResourceDefinition)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiextensions.CustomResourceDefinition)
		if !ok {
			glog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	glog.V(4).Infof("Deleting %q", castObj.Name)
	c.enqueue(castObj)
}
