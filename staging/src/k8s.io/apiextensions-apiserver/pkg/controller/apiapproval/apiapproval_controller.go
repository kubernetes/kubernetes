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

package apiapproval

import (
	"context"
	"fmt"
	"sync"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	client "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

// KubernetesAPIApprovalPolicyConformantConditionController is maintaining the KubernetesAPIApprovalPolicyConformant condition.
type KubernetesAPIApprovalPolicyConformantConditionController struct {
	crdClient client.CustomResourceDefinitionsGetter

	crdLister listers.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(ctx context.Context, key string) error

	queue workqueue.TypedRateLimitingInterface[string]

	// last protectedAnnotation value this controller updated the condition per CRD name (to avoid two
	// different version of the apiextensions-apiservers in HA to fight for the right message)
	lastSeenProtectedAnnotationLock sync.Mutex
	lastSeenProtectedAnnotation     map[string]string
}

// NewKubernetesAPIApprovalPolicyConformantConditionController constructs a KubernetesAPIApprovalPolicyConformant schema condition controller.
func NewKubernetesAPIApprovalPolicyConformantConditionController(
	crdInformer informers.CustomResourceDefinitionInformer,
	crdClient client.CustomResourceDefinitionsGetter,
) *KubernetesAPIApprovalPolicyConformantConditionController {
	c := &KubernetesAPIApprovalPolicyConformantConditionController{
		crdClient: crdClient,
		crdLister: crdInformer.Lister(),
		crdSynced: crdInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "kubernetes_api_approval_conformant_condition_controller"},
		),
		lastSeenProtectedAnnotation: map[string]string{},
	}

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
		DeleteFunc: c.deleteCustomResourceDefinition,
	})

	c.syncFn = c.sync

	return c
}

// calculateCondition determines the new KubernetesAPIApprovalPolicyConformant condition
func calculateCondition(crd *apiextensionsv1.CustomResourceDefinition) *apiextensionsv1.CustomResourceDefinitionCondition {
	if !apihelpers.IsProtectedCommunityGroup(crd.Spec.Group) {
		return nil
	}

	approvalState, reason := apihelpers.GetAPIApprovalState(crd.Annotations)
	switch approvalState {
	case apihelpers.APIApprovalInvalid:
		return &apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.KubernetesAPIApprovalPolicyConformant,
			Status:  apiextensionsv1.ConditionFalse,
			Reason:  "InvalidAnnotation",
			Message: reason,
		}
	case apihelpers.APIApprovalMissing:
		return &apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.KubernetesAPIApprovalPolicyConformant,
			Status:  apiextensionsv1.ConditionFalse,
			Reason:  "MissingAnnotation",
			Message: reason,
		}
	case apihelpers.APIApproved:
		return &apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.KubernetesAPIApprovalPolicyConformant,
			Status:  apiextensionsv1.ConditionTrue,
			Reason:  "ApprovedAnnotation",
			Message: reason,
		}
	case apihelpers.APIApprovalBypassed:
		return &apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.KubernetesAPIApprovalPolicyConformant,
			Status:  apiextensionsv1.ConditionFalse,
			Reason:  "UnapprovedAnnotation",
			Message: reason,
		}
	default:
		return &apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.KubernetesAPIApprovalPolicyConformant,
			Status:  apiextensionsv1.ConditionUnknown,
			Reason:  "UnknownAnnotation",
			Message: reason,
		}
	}
}

func (c *KubernetesAPIApprovalPolicyConformantConditionController) sync(ctx context.Context, key string) error {
	inCustomResourceDefinition, err := c.crdLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	// avoid repeated calculation for the same annotation
	protectionAnnotationValue := inCustomResourceDefinition.Annotations[apiextensionsv1.KubeAPIApprovedAnnotation]
	c.lastSeenProtectedAnnotationLock.Lock()
	lastSeen, seenBefore := c.lastSeenProtectedAnnotation[inCustomResourceDefinition.Name]
	c.lastSeenProtectedAnnotationLock.Unlock()
	if seenBefore && protectionAnnotationValue == lastSeen {
		return nil
	}

	// check old condition
	cond := calculateCondition(inCustomResourceDefinition)
	if cond == nil {
		// because group is immutable, if we have no condition now, we have no need to remove a condition.
		return nil
	}
	old := apihelpers.FindCRDCondition(inCustomResourceDefinition, apiextensionsv1.KubernetesAPIApprovalPolicyConformant)

	// don't attempt a write if all the condition details are the same
	if old != nil && old.Status == cond.Status && old.Reason == cond.Reason && old.Message == cond.Message {
		// no need to update annotation because we took no action.
		return nil
	}

	// update condition
	crd := inCustomResourceDefinition.DeepCopy()
	apihelpers.SetCRDCondition(crd, *cond)

	_, err = c.crdClient.CustomResourceDefinitions().UpdateStatus(ctx, crd, metav1.UpdateOptions{})
	if apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
		// deleted or changed in the meantime, we'll get called again
		return nil
	}
	if err != nil {
		return err
	}

	// store annotation in order to avoid repeated updates for the same annotation (and potential
	// fights of API server in HA environments).
	c.lastSeenProtectedAnnotationLock.Lock()
	defer c.lastSeenProtectedAnnotationLock.Unlock()
	c.lastSeenProtectedAnnotation[crd.Name] = protectionAnnotationValue

	return nil
}

// Run starts the controller.
func (c *KubernetesAPIApprovalPolicyConformantConditionController) Run(workers int, stopCh <-chan struct{}) {
	c.RunWithContext(workers, wait.ContextForChannel(stopCh))
}

// RunWithContext starts the controller with a context.
//
//logcheck:context // RunWithContext should be used instead of Run in code which supports contextual logging.
func (c *KubernetesAPIApprovalPolicyConformantConditionController) RunWithContext(workers int, ctx context.Context) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting KubernetesAPIApprovalPolicyConformantConditionController")
	defer klog.Infof("Shutting down KubernetesAPIApprovalPolicyConformantConditionController")

	if !cache.WaitForCacheSync(ctx.Done(), c.crdSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

func (c *KubernetesAPIApprovalPolicyConformantConditionController) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *KubernetesAPIApprovalPolicyConformantConditionController) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncFn(ctx, key)
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *KubernetesAPIApprovalPolicyConformantConditionController) enqueue(obj *apiextensionsv1.CustomResourceDefinition) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", obj, err))
		return
	}

	c.queue.Add(key)
}

func (c *KubernetesAPIApprovalPolicyConformantConditionController) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensionsv1.CustomResourceDefinition)
	klog.V(4).Infof("Adding %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *KubernetesAPIApprovalPolicyConformantConditionController) updateCustomResourceDefinition(obj, _ interface{}) {
	castObj := obj.(*apiextensionsv1.CustomResourceDefinition)
	klog.V(4).Infof("Updating %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *KubernetesAPIApprovalPolicyConformantConditionController) deleteCustomResourceDefinition(obj interface{}) {
	castObj, ok := obj.(*apiextensionsv1.CustomResourceDefinition)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiextensionsv1.CustomResourceDefinition)
		if !ok {
			klog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}

	c.lastSeenProtectedAnnotationLock.Lock()
	defer c.lastSeenProtectedAnnotationLock.Unlock()
	delete(c.lastSeenProtectedAnnotation, castObj.Name)
}
