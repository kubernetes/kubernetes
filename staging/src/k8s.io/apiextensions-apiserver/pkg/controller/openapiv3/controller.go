/*
Copyright 2021 The Kubernetes Authors.

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

package openapiv3

import (
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"golang.org/x/exp/slices"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/controller/openapi/builder"
)

// Controller watches CustomResourceDefinitions and publishes OpenAPI v3
type Controller struct {
	crdLister  listers.CustomResourceDefinitionLister
	crdsSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(string) error

	queue workqueue.RateLimitingInterface

	openAPIV3Service *handler3.OpenAPIService

	// specs per version and per CRD name
	lock sync.Mutex

	// cacheByNameandGV is a graph of cached.Value items that build the OpenAPI V3
	cacheByNameandGV map[string]map[schema.GroupVersion]*crdCache
}

type crdCache struct {
	crd        cached.LastSuccess[*apiextensionsv1.CustomResourceDefinition]
	crdOpenAPI cached.Value[*spec3.OpenAPI]
}

// NewController creates a new Controller with input CustomResourceDefinition informer
func NewController(crdInformer informers.CustomResourceDefinitionInformer) *Controller {
	c := &Controller{
		crdLister:        crdInformer.Lister(),
		crdsSynced:       crdInformer.Informer().HasSynced,
		queue:            workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "crd_openapi_v3_controller"),
		cacheByNameandGV: map[string]map[schema.GroupVersion]*crdCache{},
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
func (c *Controller) Run(openAPIV3Service *handler3.OpenAPIService, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer klog.Infof("Shutting down OpenAPI V3 controller")

	klog.Infof("Starting OpenAPI V3 controller")

	c.openAPIV3Service = openAPIV3Service

	if !cache.WaitForCacheSync(stopCh, c.crdsSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	crds, err := c.crdLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to initially list all CRDs: %v", err))
		return
	}
	for _, crd := range crds {
		if !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
			continue
		}
		versions := []string{}
		for _, v := range crd.Spec.Versions {
			if !v.Served {
				continue
			}
			versions = append(versions, v.Name)
		}
		c.addUpdateCRD(crd, crd.Name, versions)
	}

	// only start one worker thread since its a slow moving API
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *Controller) sync(name string) error {
	// controller single threaded so lock is technically unnecessary.
	c.lock.Lock()
	defer c.lock.Unlock()

	crd, err := c.crdLister.Get(name)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}

	if errors.IsNotFound(err) || !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
		c.cleanCRD(name, []string{})
		return nil
	}

	versions := []string{}
	for _, v := range crd.Spec.Versions {
		if !v.Served {
			continue
		}
		versions = append(versions, v.Name)
	}
	c.addUpdateCRD(crd, crd.Name, versions)

	return nil
}

func (c *Controller) cleanCRD(name string, versionsRemaining []string) {
	crdSpecs, ok := c.cacheByNameandGV[name]
	if !ok {
		return
	}
	for gv := range crdSpecs {
		if !slices.Contains(versionsRemaining, gv.Version) {
			delete(crdSpecs, gv)
			c.updateGroupVersion(gv)
		}
	}
	if len(crdSpecs) == 0 {
		delete(c.cacheByNameandGV, name)
	}
}
func (c *Controller) addUpdateCRD(crd *apiextensionsv1.CustomResourceDefinition, crdName string, versions []string) {
	crdSpecs, ok := c.cacheByNameandGV[crdName]
	if !ok {
		crdSpecs = make(map[schema.GroupVersion]*crdCache)
	}
	for _, version := range versions {
		gv := schema.GroupVersion{Group: crd.Spec.Group, Version: version}
		if _, ok := crdSpecs[gv]; ok {
			crdSpecs[gv].crd.Store(cached.Static(crd, generateCRDHash(crd)))
			continue
		}
		crdSpecs[gv] = c.buildCRDCache(crd, crdName, version)
		c.updateGroupVersion(gv)
	}
	c.cleanCRD(crdName, versions)
}

func (c *Controller) updateGroupVersion(gv schema.GroupVersion) error {
	// delete if not needed?
	specList := []cached.Value[*spec3.OpenAPI]{}
	for crd := range c.cacheByNameandGV {
		if _, ok := c.cacheByNameandGV[crd][gv]; ok {
			specList = append(specList, c.cacheByNameandGV[crd][gv].crdOpenAPI)
		}
	}
	cache := cached.MergeList(func(results []cached.Result[*spec3.OpenAPI]) (*spec3.OpenAPI, string, error) {
		localCRDSpec := make([]*spec3.OpenAPI, 0, len(results))
		for k := range results {
			if results[k].Err == nil {
				localCRDSpec = append(localCRDSpec, results[k].Value)
			}
		}
		mergedSpec, err := builder.MergeSpecsV3(localCRDSpec...)
		if err != nil {
			return nil, "", fmt.Errorf("failed to merge specs: %v", err)
		}
		return mergedSpec, uuid.New().String(), nil
	}, specList)
	c.openAPIV3Service.UpdateGroupVersionLazy(groupVersionToOpenAPIV3Path(gv), cache)
	return nil
}

func (c *Controller) buildCRDCache(crd *apiextensionsv1.CustomResourceDefinition, name, versionName string) *crdCache {
	crdcache := &crdCache{}
	crdcache.crd.Store(cached.Static(crd, generateCRDHash(crd)))
	crdcache.crdOpenAPI = cached.Transform[*apiextensionsv1.CustomResourceDefinition](func(crd *apiextensionsv1.CustomResourceDefinition, etag string, err error) (*spec3.OpenAPI, string, error) {
		v3, err := builder.BuildOpenAPIV3(crd, versionName, builder.Options{V2: false})
		return v3, generateCRDHash(crd), err
	}, &crdcache.crd)
	return crdcache
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

	err := c.syncFn(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)
	return true
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
