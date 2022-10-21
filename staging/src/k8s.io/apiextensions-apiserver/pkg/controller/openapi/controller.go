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

	kcpcache "github.com/kcp-dev/apimachinery/pkg/cache"
	"github.com/kcp-dev/logicalcluster/v2"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/routes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
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

	queue workqueue.RateLimitingInterface

	staticSpec             *spec.Swagger
	openAPIServiceProvider routes.OpenAPIServiceProvider

	// specs per cluster and per version and per CRD name
	lock     sync.Mutex
	crdSpecs map[logicalcluster.Name]map[string]map[string]*spec.Swagger
}

// NewController creates a new Controller with input CustomResourceDefinition informer
func NewController(crdInformer informers.CustomResourceDefinitionInformer) *Controller {
	c := &Controller{
		crdLister:  crdInformer.Lister(),
		crdsSynced: crdInformer.Informer().HasSynced,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "crd_openapi_controller"),
		crdSpecs:   map[logicalcluster.Name]map[string]map[string]*spec.Swagger{},
	}

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
		DeleteFunc: c.deleteCustomResourceDefinition,
	})

	c.syncFn = c.sync
	return c
}

// HACK:
//  Everything regarding OpenAPI and resource discovery is managed through controllers currently
// (a number of controllers highly coupled with the corresponding http handlers).
// The following code is an attempt at provising CRD tenancy while accommodating the current design without being too much invasive,
// because doing differently would have meant too much refactoring..
// But in the long run the "do this dynamically, not as part of a controller" is probably going to be important.
// openapi/crd generation is expensive, so doing on a controller means that CPU and memory scale O(crds),
// when we really want them to scale O(active_clusters).

func (c *Controller) setClusterCrdSpecs(clusterName logicalcluster.Name, crdName string, newSpecs map[string]*spec.Swagger) {
	_, found := c.crdSpecs[clusterName]
	if !found {
		c.crdSpecs[clusterName] = map[string]map[string]*spec.Swagger{}
	}
	c.crdSpecs[clusterName][crdName] = newSpecs
	c.openAPIServiceProvider.AddCuster(clusterName)
}

func (c *Controller) removeClusterCrdSpecs(clusterName logicalcluster.Name, crdName string) bool {
	_, crdsForClusterFound := c.crdSpecs[clusterName]
	if !crdsForClusterFound {
		return false
	}
	if _, found := c.crdSpecs[clusterName][crdName]; !found {
		return false
	}
	delete(c.crdSpecs[clusterName], crdName)
	if len(c.crdSpecs[clusterName]) == 0 {
		delete(c.crdSpecs, clusterName)
		c.openAPIServiceProvider.RemoveCuster(clusterName)
	}
	return true
}

func (c *Controller) getClusterCrdSpecs(clusterName logicalcluster.Name, crdName string) (map[string]*spec.Swagger, bool) {
	_, specsFoundForCluster := c.crdSpecs[clusterName]
	if !specsFoundForCluster {
		return map[string]*spec.Swagger{}, false
	}
	crdSpecs, found := c.crdSpecs[clusterName][crdName]
	return crdSpecs, found
}

// Run sets openAPIAggregationManager and starts workers
func (c *Controller) Run(staticSpec *spec.Swagger, openAPIServiceProvider routes.OpenAPIServiceProvider, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer klog.Infof("Shutting down OpenAPI controller")

	klog.Infof("Starting OpenAPI controller")

	c.staticSpec = staticSpec
	c.openAPIServiceProvider = openAPIServiceProvider

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
		newSpecs, changed, err := buildVersionSpecs(crd, nil)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to build OpenAPI spec of CRD %s: %v", crd.Name, err))
		} else if !changed {
			continue
		}
		c.setClusterCrdSpecs(logicalcluster.From(crd), crd.Name, newSpecs)
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

func (c *Controller) sync(key string) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	clusterName, _, crdName, err := kcpcache.SplitMetaClusterNamespaceKey(key)
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	}

	crd, err := c.crdLister.Get(clusterName.String() + "|" + crdName)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}

	// do we have to remove all specs of this CRD?
	if errors.IsNotFound(err) || !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
		if !c.removeClusterCrdSpecs(clusterName, crdName) {
			return nil
		}
		klog.V(2).Infof("Updating CRD OpenAPI spec because %s was removed", crdName)
		regenerationCounter.With(map[string]string{"crd": crdName, "reason": "remove"})
		return c.updateSpecLocked()
	}

	// compute CRD spec and see whether it changed
	oldSpecs, updated := c.getClusterCrdSpecs(logicalcluster.From(crd), crd.Name)
	newSpecs, changed, err := buildVersionSpecs(crd, oldSpecs)
	if err != nil {
		return err
	}
	if !changed {
		return nil
	}

	// update specs of this CRD
	c.setClusterCrdSpecs(logicalcluster.From(crd), crd.Name, newSpecs)
	klog.V(2).Infof("Updating CRD OpenAPI spec because %s changed", crd.Name)
	reason := "add"
	if updated {
		reason = "update"
	}
	regenerationCounter.With(map[string]string{"crd": crd.Name, "reason": reason})
	return c.updateSpecLocked()
}

func buildVersionSpecs(crd *apiextensionsv1.CustomResourceDefinition, oldSpecs map[string]*spec.Swagger) (map[string]*spec.Swagger, bool, error) {
	newSpecs := map[string]*spec.Swagger{}
	anyChanged := false
	for _, v := range crd.Spec.Versions {
		if !v.Served {
			continue
		}
		// Defaults are not pruned here, but before being served.
		spec, err := builder.BuildOpenAPIV2(crd, v.Name, builder.Options{V2: true})
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
	var errs []error
	for clusterName, clusterCrdSpecs := range c.crdSpecs {
		crdSpecs := []*spec.Swagger{}
		for _, versionSpecs := range clusterCrdSpecs {
			for _, s := range versionSpecs {
				crdSpecs = append(crdSpecs, s)
			}
		}
		mergedSpec, err := builder.MergeSpecs(c.staticSpec, crdSpecs...)
		if err != nil {
			return fmt.Errorf("failed to merge specs: %v", err)
		}
		if err := c.openAPIServiceProvider.ForCluster(clusterName).UpdateSpec(mergedSpec); err != nil {
			errs = append(errs, err)
		}
	}

	return utilerrors.NewAggregate(errs)
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
	key, _ := kcpcache.MetaClusterNamespaceKeyFunc(obj)
	c.queue.Add(key)
}
