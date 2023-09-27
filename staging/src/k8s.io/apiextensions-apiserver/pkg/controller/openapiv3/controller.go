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
	"reflect"
	"sync"
	"time"

	kcpcache "github.com/kcp-dev/apimachinery/v2/pkg/cache"
	kcpapiextensionsv1informers "github.com/kcp-dev/client-go/apiextensions/informers/apiextensions/v1"
	kcpapiextensionsv1listers "github.com/kcp-dev/client-go/apiextensions/listers/apiextensions/v1"

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/controller/openapi/builder"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"
)

// Controller watches CustomResourceDefinitions and publishes OpenAPI v3
type Controller struct {
	crdLister  kcpapiextensionsv1listers.CustomResourceDefinitionClusterLister
	crdsSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(string) error

	queue workqueue.TypedRateLimitingInterface[string]

	openAPIV3Service *handler3.OpenAPIService

	// specs per version and per CRD name
	lock             sync.Mutex
	specsByGVandName map[schema.GroupVersion]map[string]*spec3.OpenAPI
}

// NewController creates a new Controller with input CustomResourceDefinition informer
func NewController(crdInformer kcpapiextensionsv1informers.CustomResourceDefinitionClusterInformer) *Controller {
	c := &Controller{
		crdLister:  crdInformer.Lister(),
		crdsSynced: crdInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "crd_openapi_v3_controller"},
		),
		specsByGVandName: map[schema.GroupVersion]map[string]*spec3.OpenAPI{},
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
		for _, v := range crd.Spec.Versions {
			if !v.Served {
				continue
			}
			c.buildV3Spec(crd, crd.Name, v.Name)
		}
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

func (c *Controller) sync(key string) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	clusterName, _, name, err := kcpcache.SplitMetaClusterNamespaceKey(key)
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	}
	crd, err := c.crdLister.Cluster(clusterName).Get(name)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}

	if errors.IsNotFound(err) || !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
		c.deleteCRD(name)
		return nil
	}

	for _, v := range crd.Spec.Versions {
		if !v.Served {
			continue
		}
		c.buildV3Spec(crd, name, v.Name)
	}

	return nil
}

func (c *Controller) deleteCRD(name string) {
	for gv, crdListForGV := range c.specsByGVandName {
		_, needOpenAPIUpdate := crdListForGV[name]
		if needOpenAPIUpdate {
			delete(crdListForGV, name)
			if len(crdListForGV) == 0 {
				delete(c.specsByGVandName, gv)
			}
			regenerationCounter.With(map[string]string{"group": gv.Group, "version": gv.Version, "crd": name, "reason": "remove"})
			c.updateGroupVersion(gv)
		}
	}
}

func (c *Controller) updateGroupVersion(gv schema.GroupVersion) error {
	if _, ok := c.specsByGVandName[gv]; !ok {
		c.openAPIV3Service.DeleteGroupVersion(groupVersionToOpenAPIV3Path(gv))
		return nil
	}

	var specs []*spec3.OpenAPI
	for _, spec := range c.specsByGVandName[gv] {
		specs = append(specs, spec)
	}

	mergedSpec, err := builder.MergeSpecsV3(specs...)
	if err != nil {
		return fmt.Errorf("failed to merge specs: %v", err)
	}

	c.openAPIV3Service.UpdateGroupVersion(groupVersionToOpenAPIV3Path(gv), mergedSpec)
	return nil
}

func (c *Controller) updateCRDSpec(crd *apiextensionsv1.CustomResourceDefinition, name, versionName string, v3 *spec3.OpenAPI) error {
	gv := schema.GroupVersion{
		Group:   crd.Spec.Group,
		Version: versionName,
	}

	_, ok := c.specsByGVandName[gv]
	reason := "update"
	if !ok {
		reason = "add"
		c.specsByGVandName[gv] = map[string]*spec3.OpenAPI{}
	}

	oldSpec, ok := c.specsByGVandName[gv][name]
	if ok {
		if reflect.DeepEqual(oldSpec, v3) {
			// no changes to CRD
			return nil
		}
	}
	c.specsByGVandName[gv][name] = v3
	regenerationCounter.With(map[string]string{"crd": name, "group": gv.Group, "version": gv.Version, "reason": reason})
	return c.updateGroupVersion(gv)
}

func (c *Controller) buildV3Spec(crd *apiextensionsv1.CustomResourceDefinition, name, versionName string) error {
	v3, err := builder.BuildOpenAPIV3(crd, versionName, builder.Options{
		V2:                      false,
		IncludeSelectableFields: utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceFieldSelectors),
	})

	if err != nil {
		return err
	}

	c.updateCRDSpec(crd, name, versionName, v3)
	return nil
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
	key, err := kcpcache.DeletionHandlingMetaClusterNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}
	c.queue.Add(key)
}
