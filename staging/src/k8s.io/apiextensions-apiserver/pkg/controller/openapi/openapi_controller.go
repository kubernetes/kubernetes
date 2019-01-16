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

package openapi

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	apiextensionsopenapi "k8s.io/apiextensions-apiserver/pkg/openapi"
)

// OpenAPIController watches CRDs, and registers validation schema for CRDs to
// OpenAPI aggregation manager
type OpenAPIController struct {
	crdLister  listers.CustomResourceDefinitionLister
	crdsSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(version schema.GroupVersionKind) error

	queue workqueue.RateLimitingInterface

	openAPIAggregationManager apiextensionsopenapi.AggregationManager
}

func NewOpenAPIController(crdInformer informers.CustomResourceDefinitionInformer) *OpenAPIController {
	c := &OpenAPIController{
		crdLister:  crdInformer.Lister(),
		crdsSynced: crdInformer.Informer().HasSynced,

		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "crd_openapi_controller"),
	}

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
		DeleteFunc: c.deleteCustomResourceDefinition,
	})

	c.syncFn = c.sync

	return c
}

func (c *OpenAPIController) sync(version schema.GroupVersionKind) error {
	if c.openAPIAggregationManager == nil || !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceValidation) {
		return nil
	}
	crds, err := c.crdLister.List(labels.Everything())
	if err != nil {
		return err
	}
	apiServiceName := version.String()
	found := false
	for _, crd := range crds {
		if !apiextensions.IsCRDConditionTrue(crd, apiextensions.Established) {
			continue
		}
		if crd.Spec.Group != version.Group || crd.Spec.Names.Kind != version.Kind {
			continue
		}
		if !hasVersionServed(crd, version.Version) {
			continue
		}

		found = true
		validationSchema, err := getSchemaForVersion(crd, version.Version)
		if err != nil {
			return err
		}
		// We aggregate the schema even if it's nil as it maybe a removal of the schema for this CRD,
		// and the aggreated OpenAPI spec should reflect this change.
		crdspec, etag, err := apiextensionsopenapi.CustomResourceDefinitionOpenAPISpec(&crd.Spec, version.Version, validationSchema)
		if err != nil {
			return err
		}

		// Add/update the local API service's spec for the CRD in apiExtensionsServer's
		// openAPIAggregationManager
		if err := c.openAPIAggregationManager.AddUpdateLocalAPIServiceSpec(apiServiceName, crdspec, etag); err != nil {
			return err
		}
	}

	if !found {
		// Remove the local API service for the CRD in apiExtensionsServer's
		// openAPIAggregationManager.
		// Note that we don't check if apiServiceName exists in openAPIAggregationManager
		// because RemoveAPIServiceSpec properly handles non-existing API service by
		// returning no error.
		return c.openAPIAggregationManager.RemoveAPIServiceSpec(apiServiceName)
	}
	return nil
}

func (c *OpenAPIController) Run(stopCh <-chan struct{}, crdOpenAPIAggregationManager apiextensionsopenapi.AggregationManager) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer klog.Infof("Shutting down OpenAPIController")

	klog.Infof("Starting OpenAPIController")

	c.openAPIAggregationManager = crdOpenAPIAggregationManager

	if !cache.WaitForCacheSync(stopCh, c.crdsSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	// only start one worker thread since its a slow moving API
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *OpenAPIController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue. It returns false when it's time to quit.
func (c *OpenAPIController) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncFn(key.(schema.GroupVersionKind))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *OpenAPIController) enqueue(obj *apiextensions.CustomResourceDefinition) {
	for _, v := range obj.Spec.Versions {
		c.queue.Add(schema.GroupVersionKind{Group: obj.Spec.Group, Version: v.Name, Kind: obj.Spec.Names.Kind})
	}
}

func (c *OpenAPIController) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	klog.V(4).Infof("Adding customresourcedefinition %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *OpenAPIController) updateCustomResourceDefinition(oldObj, newObj interface{}) {
	castNewObj := newObj.(*apiextensions.CustomResourceDefinition)
	castOldObj := oldObj.(*apiextensions.CustomResourceDefinition)
	klog.V(4).Infof("Updating customresourcedefinition %s", castOldObj.Name)
	// Enqueue both old and new object to make sure we remove and add appropriate Versions.
	// The working queue will resolve any duplicates and only changes will stay in the queue.
	c.enqueue(castNewObj)
	c.enqueue(castOldObj)
}

func (c *OpenAPIController) deleteCustomResourceDefinition(obj interface{}) {
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
	klog.V(4).Infof("Deleting customresourcedefinition %q", castObj.Name)
	c.enqueue(castObj)
}
