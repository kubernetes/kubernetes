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
	"sync"
	"time"

	"github.com/google/uuid"

	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/controller/openapi/builder"
)

// Controller watches CustomResourceDefinitions and publishes validation schema
type Controller struct {
	crdLister  listers.CustomResourceDefinitionLister
	crdsSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(string) error

	queue workqueue.TypedRateLimitingInterface[string]

	staticSpec *spec.Swagger

	openAPIService *handler.OpenAPIService

	// specs by name. The specs are lazily constructed on request.
	// The lock is for the map only.
	lock        sync.Mutex
	specsByName map[string]*specCache
}

// specCache holds the merged version spec for a CRD as well as the CRD object.
// The spec is created lazily from the CRD object on request.
// The mergedVersionSpec is only created on instantiation and is never
// changed. crdCache is a cached.Replaceable and updates are thread
// safe. Thus, no lock is needed to protect this struct.
type specCache struct {
	crdCache          cached.LastSuccess[*apiextensionsv1.CustomResourceDefinition]
	mergedVersionSpec cached.Value[*spec.Swagger]
}

func (s *specCache) update(crd *apiextensionsv1.CustomResourceDefinition) {
	s.crdCache.Store(cached.Static(crd, generateCRDHash(crd)))
}

func createSpecCache(crd *apiextensionsv1.CustomResourceDefinition) *specCache {
	s := specCache{}
	s.update(crd)

	s.mergedVersionSpec = cached.Transform[*apiextensionsv1.CustomResourceDefinition](func(crd *apiextensionsv1.CustomResourceDefinition, etag string, err error) (*spec.Swagger, string, error) {
		if err != nil {
			// This should never happen, but return the err if it does.
			return nil, "", err
		}
		mergeSpec := &spec.Swagger{}
		for _, v := range crd.Spec.Versions {
			if !v.Served {
				continue
			}
			s, err := builder.BuildOpenAPIV2(crd, v.Name, builder.Options{
				V2:                      true,
				IncludeSelectableFields: utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceFieldSelectors),
			})
			// Defaults must be pruned here for CRDs to cleanly merge with the static
			// spec that already has defaults pruned
			if err != nil {
				return nil, "", err
			}
			s.Definitions = handler.PruneDefaults(s.Definitions)
			mergeSpec, err = builder.MergeSpecs(mergeSpec, s)
			if err != nil {
				return nil, "", err
			}
		}
		return mergeSpec, generateCRDHash(crd), nil
	}, &s.crdCache)
	return &s
}

// NewController creates a new Controller with input CustomResourceDefinition informer
func NewController(crdInformer informers.CustomResourceDefinitionInformer) *Controller {
	c := &Controller{
		crdLister:  crdInformer.Lister(),
		crdsSynced: crdInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "crd_openapi_controller"},
		),
		specsByName: map[string]*specCache{},
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
		if !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
			continue
		}
		c.specsByName[crd.Name] = createSpecCache(crd)
	}
	c.updateSpecLocked()

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
			klog.Warningf("slow openapi aggregation of %q: %s", key, elapsed)
		}
	}()

	err := c.syncFn(key)
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
	if errors.IsNotFound(err) || !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
		if _, found := c.specsByName[name]; !found {
			return nil
		}
		delete(c.specsByName, name)
		klog.V(2).Infof("Updating CRD OpenAPI spec because %s was removed", name)
		regenerationCounter.With(map[string]string{"crd": name, "reason": "remove"})
		c.updateSpecLocked()
		return nil
	}

	// If CRD spec already exists, update the CRD.
	// specCache.update() includes the ETag so an update on a spec
	// resulting in the same ETag will be a noop.
	s, exists := c.specsByName[crd.Name]
	if exists {
		s.update(crd)
		klog.V(2).Infof("Updating CRD OpenAPI spec because %s changed", name)
		regenerationCounter.With(map[string]string{"crd": name, "reason": "update"})
		return nil
	}

	c.specsByName[crd.Name] = createSpecCache(crd)
	klog.V(2).Infof("Updating CRD OpenAPI spec because %s changed", name)
	regenerationCounter.With(map[string]string{"crd": name, "reason": "add"})
	c.updateSpecLocked()
	return nil
}

// updateSpecLocked updates the cached spec graph.
func (c *Controller) updateSpecLocked() {
	specList := make([]cached.Value[*spec.Swagger], 0, len(c.specsByName))
	for crd := range c.specsByName {
		specList = append(specList, c.specsByName[crd].mergedVersionSpec)
	}

	cache := cached.MergeList(func(results []cached.Result[*spec.Swagger]) (*spec.Swagger, string, error) {
		localCRDSpec := make([]*spec.Swagger, 0, len(results))
		for k := range results {
			if results[k].Err == nil {
				localCRDSpec = append(localCRDSpec, results[k].Value)
			}
		}
		mergedSpec, err := builder.MergeSpecs(c.staticSpec, localCRDSpec...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to merge specs: %v", err)
		}
		// A UUID is returned for the etag because we will only
		// create a new merger when a CRD has changed. A hash based
		// etag is more expensive because the CRDs are not
		// premarshalled.
		return mergedSpec, uuid.New().String(), nil
	}, specList)
	c.openAPIService.UpdateSpecLazy(cache)
}

func (c *Controller) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensionsv1.CustomResourceDefinition)
	klog.V(4).Infof("Adding customresourcedefinition %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *Controller) updateCustomResourceDefinition(oldObj, newObj interface{}) {
	castNewObj := newObj.(*apiextensionsv1.CustomResourceDefinition)
	klog.V(4).Infof("Updating customresourcedefinition %s", castNewObj.Name)
	c.enqueue(castNewObj)
}

func (c *Controller) deleteCustomResourceDefinition(obj interface{}) {
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
	klog.V(4).Infof("Deleting customresourcedefinition %q", castObj.Name)
	c.enqueue(castObj)
}

func (c *Controller) enqueue(obj *apiextensionsv1.CustomResourceDefinition) {
	c.queue.Add(obj.Name)
}

func generateCRDHash(crd *apiextensionsv1.CustomResourceDefinition) string {
	return fmt.Sprintf("%s,%d", crd.UID, crd.Generation)
}
