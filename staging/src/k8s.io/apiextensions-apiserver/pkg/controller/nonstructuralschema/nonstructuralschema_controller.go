/*
Copyright 2019 The Kubernetes Authors.

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

package nonstructuralschema

import (
	"fmt"
	"sync"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	client "k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
)

// ConditionController is maintaining the NonStructuralSchema condition.
type ConditionController struct {
	crdClient client.CustomResourceDefinitionsGetter

	crdLister listers.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.RateLimitingInterface

	// last generation this controller updated the condition per CRD name (to avoid two
	// different version of the apiextensions-apiservers in HA to fight for the right message)
	lastSeenGenerationLock sync.Mutex
	lastSeenGeneration     map[string]int64
}

// NewConditionController constructs a non-structural schema condition controller.
func NewConditionController(
	crdInformer informers.CustomResourceDefinitionInformer,
	crdClient client.CustomResourceDefinitionsGetter,
) *ConditionController {
	c := &ConditionController{
		crdClient:          crdClient,
		crdLister:          crdInformer.Lister(),
		crdSynced:          crdInformer.Informer().HasSynced,
		queue:              workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "non_structural_schema_condition_controller"),
		lastSeenGeneration: map[string]int64{},
	}

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
		DeleteFunc: c.deleteCustomResourceDefinition,
	})

	c.syncFn = c.sync

	return c
}

func calculateCondition(in *apiextensions.CustomResourceDefinition) *apiextensions.CustomResourceDefinitionCondition {
	cond := &apiextensions.CustomResourceDefinitionCondition{
		Type:   apiextensions.NonStructuralSchema,
		Status: apiextensions.ConditionUnknown,
	}

	allErrs := field.ErrorList{}

	if in.Spec.Validation != nil && in.Spec.Validation.OpenAPIV3Schema != nil {
		s, err := schema.NewStructural(in.Spec.Validation.OpenAPIV3Schema)
		if err != nil {
			cond.Reason = "StructuralError"
			cond.Message = fmt.Sprintf("failed to check global validation schema: %v", err)
			return cond
		}

		pth := field.NewPath("spec", "validation", "openAPIV3Schema")

		allErrs = append(allErrs, schema.ValidateStructural(s, pth)...)
	}

	for _, v := range in.Spec.Versions {
		if v.Schema == nil || v.Schema.OpenAPIV3Schema == nil {
			continue
		}

		s, err := schema.NewStructural(v.Schema.OpenAPIV3Schema)
		if err != nil {
			cond.Reason = "StructuralError"
			cond.Message = fmt.Sprintf("failed to check validation schema for version %s: %v", v.Name, err)
			return cond
		}

		pth := field.NewPath("spec", "version").Key(v.Name).Child("schema", "openAPIV3Schema")

		allErrs = append(allErrs, schema.ValidateStructural(s, pth)...)
	}

	if len(allErrs) == 0 {
		return nil
	}

	cond.Status = apiextensions.ConditionTrue
	cond.Reason = "Violations"
	cond.Message = allErrs.ToAggregate().Error()

	return cond
}

func (c *ConditionController) sync(key string) error {
	inCustomResourceDefinition, err := c.crdLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	// avoid repeated calculation for the same generation
	c.lastSeenGenerationLock.Lock()
	lastSeen, seenBefore := c.lastSeenGeneration[inCustomResourceDefinition.Name]
	c.lastSeenGenerationLock.Unlock()
	if seenBefore && inCustomResourceDefinition.Generation <= lastSeen {
		return nil
	}

	// check old condition
	cond := calculateCondition(inCustomResourceDefinition)
	old := apiextensions.FindCRDCondition(inCustomResourceDefinition, apiextensions.NonStructuralSchema)

	if cond == nil && old == nil {
		return nil
	}
	if cond != nil && old != nil && old.Status == cond.Status && old.Reason == cond.Reason && old.Message == cond.Message {
		return nil
	}

	// update condition
	crd := inCustomResourceDefinition.DeepCopy()
	if cond == nil {
		apiextensions.RemoveCRDCondition(crd, apiextensions.NonStructuralSchema)
	} else {
		cond.LastTransitionTime = metav1.NewTime(time.Now())
		apiextensions.SetCRDCondition(crd, *cond)
	}

	_, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(crd)
	if apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
		// deleted or changed in the meantime, we'll get called again
		return nil
	}
	if err != nil {
		return err
	}

	// store generation in order to avoid repeated updates for the same generation (and potential
	// fights of API server in HA environments).
	c.lastSeenGenerationLock.Lock()
	defer c.lastSeenGenerationLock.Unlock()
	c.lastSeenGeneration[crd.Name] = crd.Generation

	return nil
}

// Run starts the controller.
func (c *ConditionController) Run(threadiness int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting NonStructuralSchemaConditionController")
	defer klog.Infof("Shutting down NonStructuralSchemaConditionController")

	if !cache.WaitForCacheSync(stopCh, c.crdSynced) {
		return
	}

	for i := 0; i < threadiness; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *ConditionController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *ConditionController) processNextWorkItem() bool {
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

func (c *ConditionController) enqueue(obj *apiextensions.CustomResourceDefinition) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}

	c.queue.Add(key)
}

func (c *ConditionController) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	klog.V(4).Infof("Adding %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *ConditionController) updateCustomResourceDefinition(obj, _ interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	klog.V(4).Infof("Updating %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *ConditionController) deleteCustomResourceDefinition(obj interface{}) {
	castObj, ok := obj.(*apiextensions.CustomResourceDefinition)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiextensions.CustomResourceDefinition)
		if !ok {
			klog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}

	c.lastSeenGenerationLock.Lock()
	defer c.lastSeenGenerationLock.Unlock()
	delete(c.lastSeenGeneration, castObj.Name)
}
