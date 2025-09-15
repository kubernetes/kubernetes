/*
Copyright 2018 The Kubernetes Authors.

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

package establish

import (
	"context"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	client "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1"
)

// EstablishingController controls how and when CRD is established.
type EstablishingController struct {
	crdClient client.CustomResourceDefinitionsGetter
	crdLister listers.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.TypedRateLimitingInterface[string]
}

// NewEstablishingController creates new EstablishingController.
func NewEstablishingController(crdInformer informers.CustomResourceDefinitionInformer,
	crdClient client.CustomResourceDefinitionsGetter) *EstablishingController {
	ec := &EstablishingController{
		crdClient: crdClient,
		crdLister: crdInformer.Lister(),
		crdSynced: crdInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "crdEstablishing"},
		),
	}

	ec.syncFn = ec.sync

	return ec
}

// QueueCRD adds CRD into the establishing queue.
func (ec *EstablishingController) QueueCRD(key string, timeout time.Duration) {
	ec.queue.AddAfter(key, timeout)
}

// Run starts the EstablishingController.
func (ec *EstablishingController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer ec.queue.ShutDown()

	klog.Info("Starting EstablishingController")
	defer klog.Info("Shutting down EstablishingController")

	if !cache.WaitForCacheSync(stopCh, ec.crdSynced) {
		return
	}

	// only start one worker thread since its a slow moving API
	go wait.Until(ec.runWorker, time.Second, stopCh)

	<-stopCh
}

func (ec *EstablishingController) runWorker() {
	for ec.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.
// It returns false when it's time to quit.
func (ec *EstablishingController) processNextWorkItem() bool {
	key, quit := ec.queue.Get()
	if quit {
		return false
	}
	defer ec.queue.Done(key)

	err := ec.syncFn(key)
	if err == nil {
		ec.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	ec.queue.AddRateLimited(key)

	return true
}

// sync is used to turn CRDs into the Established state.
func (ec *EstablishingController) sync(key string) error {
	cachedCRD, err := ec.crdLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	if !apiextensionshelpers.IsCRDConditionTrue(cachedCRD, apiextensionsv1.NamesAccepted) ||
		apiextensionshelpers.IsCRDConditionTrue(cachedCRD, apiextensionsv1.Established) {
		return nil
	}

	crd := cachedCRD.DeepCopy()

	// If the conversion webhook CABundle is invalid, set Established
	// condition to false and provide a reason
	if cachedCRD.Spec.Conversion != nil &&
		cachedCRD.Spec.Conversion.Webhook != nil &&
		cachedCRD.Spec.Conversion.Webhook.ClientConfig != nil &&
		len(webhook.ValidateCABundle(field.NewPath(""), cachedCRD.Spec.Conversion.Webhook.ClientConfig.CABundle)) > 0 {
		errorCondition := apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Established,
			Status:  apiextensionsv1.ConditionFalse,
			Reason:  "InvalidCABundle",
			Message: "The conversion webhook CABundle is invalid",
		}
		apiextensionshelpers.SetCRDCondition(crd, errorCondition)
	} else {
		establishedCondition := apiextensionsv1.CustomResourceDefinitionCondition{
			Type:    apiextensionsv1.Established,
			Status:  apiextensionsv1.ConditionTrue,
			Reason:  "InitialNamesAccepted",
			Message: "the initial names have been accepted",
		}
		apiextensionshelpers.SetCRDCondition(crd, establishedCondition)
	}

	// Update server with new CRD condition.
	_, err = ec.crdClient.CustomResourceDefinitions().UpdateStatus(context.TODO(), crd, metav1.UpdateOptions{})
	if apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
		// deleted or changed in the meantime, we'll get called again
		return nil
	}
	if err != nil {
		return err
	}

	return nil
}
