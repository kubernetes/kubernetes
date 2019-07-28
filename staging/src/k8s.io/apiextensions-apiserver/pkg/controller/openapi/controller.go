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

package openapi

import (
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/go-openapi/spec"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	"k8s.io/kube-openapi/pkg/handler"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
)

// Controller watches CustomResourceDefinitions and publishes validation schema
type Controller struct {
	crdLister  listers.CustomResourceDefinitionLister
	crdsSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(string) error

	queue workqueue.RateLimitingInterface

	staticSpec     *spec.Swagger
	openAPIService *handler.OpenAPIService

	// specs per version and per CRD name
	lock     sync.Mutex
	crdSpecs map[string]map[string]*spec.Swagger
}

// NewController creates a new Controller with input CustomResourceDefinition informer
func NewController(crdInformer informers.CustomResourceDefinitionInformer) *Controller {
	c := &Controller{
		crdLister:  crdInformer.Lister(),
		crdsSynced: crdInformer.Informer().HasSynced,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "crd_openapi_controller"),
		crdSpecs:   map[string]map[string]*spec.Swagger{},
	}

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
		DeleteFunc: c.deleteCustomResourceDefinition,
	})

	c.syncFn = c.sync
	return c
}

// Run sets openAPIAggregationManager and starts workers
func (c *Controller) Run(staticSpec *spec.Swagger, openAPIService *handler.OpenAPIService, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer klog.Infof("Shutting down OpenAPI controller")

	klog.Infof("Starting OpenAPI controller")

	c.staticSpec = staticSpec
	c.openAPIService = openAPIService

	if !cache.WaitForCacheSync(stopCh, c.crdsSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	// create initial spec to avoid merging once per CRD on startup
	crds, err := c.crdLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to initially list all CRDs: %v", err))
		return
	}
	for _, crd := range crds {
		if !apiextensions.IsCRDConditionTrue(crd, apiextensions.Established) {
			continue
		}
		newSpecs, changed, err := buildVersionSpecs(crd, nil)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to build OpenAPI spec of CRD %s: %v", crd.Name, err))
		} else if !changed {
			continue
		}
		c.crdSpecs[crd.Name] = newSpecs
	}
	if err := c.updateSpecLocked(); err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to initially create OpenAPI spec for CRDs: %v", err))
		return
	}

	// only start one worker thread since its a slow moving API
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *Controller) runWorker() {
	for c.processNextWorkItem() {
	}
}

func (c *Controller) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	// log slow aggregations
	start := time.Now()
	defer func() {
		elapsed := time.Since(start)
		if elapsed > time.Second {
			klog.Warningf("slow openapi aggregation of %q: %s", key.(string), elapsed)
		}
	}()

	err := c.syncFn(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)
	return true
}

func (c *Controller) sync(name string) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	crd, err := c.crdLister.Get(name)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}

	// do we have to remove all specs of this CRD?
	if errors.IsNotFound(err) || !apiextensions.IsCRDConditionTrue(crd, apiextensions.Established) {
		if _, found := c.crdSpecs[name]; !found {
			return nil
		}
		delete(c.crdSpecs, name)
		return c.updateSpecLocked()
	}

	// compute CRD spec and see whether it changed
	oldSpecs := c.crdSpecs[crd.Name]
	newSpecs, changed, err := buildVersionSpecs(crd, oldSpecs)
	if err != nil {
		return err
	}
	if !changed {
		return nil
	}

	// update specs of this CRD
	c.crdSpecs[crd.Name] = newSpecs
	return c.updateSpecLocked()
}

func buildVersionSpecs(crd *apiextensions.CustomResourceDefinition, oldSpecs map[string]*spec.Swagger) (map[string]*spec.Swagger, bool, error) {
	newSpecs := map[string]*spec.Swagger{}
	anyChanged := false
	for _, v := range crd.Spec.Versions {
		if !v.Served {
			continue
		}
		spec, err := BuildSwagger(crd, v.Name)
		if err != nil {
			return nil, false, err
		}
		newSpecs[v.Name] = spec
		if oldSpecs[v.Name] == nil || !reflect.DeepEqual(oldSpecs[v.Name], spec) {
			anyChanged = true
		}
	}
	if !anyChanged && len(oldSpecs) == len(newSpecs) {
		return newSpecs, false, nil
	}

	return newSpecs, true, nil
}

// updateSpecLocked aggregates all OpenAPI specs and updates openAPIService.
// It is not thread-safe. The caller is responsible to hold proper lock (Controller.lock).
func (c *Controller) updateSpecLocked() error {
	crdSpecs := []*spec.Swagger{}
	for _, versionSpecs := range c.crdSpecs {
		for _, s := range versionSpecs {
			crdSpecs = append(crdSpecs, s)
		}
	}
	return c.openAPIService.UpdateSpec(mergeSpecs(c.staticSpec, crdSpecs...))
}

func (c *Controller) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	klog.V(4).Infof("Adding customresourcedefinition %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *Controller) updateCustomResourceDefinition(oldObj, newObj interface{}) {
	castNewObj := newObj.(*apiextensions.CustomResourceDefinition)
	klog.V(4).Infof("Updating customresourcedefinition %s", castNewObj.Name)
	c.enqueue(castNewObj)
}

func (c *Controller) deleteCustomResourceDefinition(obj interface{}) {
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

func (c *Controller) enqueue(obj *apiextensions.CustomResourceDefinition) {
	c.queue.Add(obj.Name)
}
