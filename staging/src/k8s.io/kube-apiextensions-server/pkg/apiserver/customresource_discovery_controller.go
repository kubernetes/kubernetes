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

package apiserver

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
	informers "k8s.io/kube-apiextensions-server/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/kube-apiextensions-server/pkg/client/listers/apiextensions/internalversion"
)

type CustomResourceDiscoveryController struct {
	versionHandler *customResourceVersionDiscoveryHandler
	groupHandler   *customResourceGroupDiscoveryHandler

	customResourceLister  listers.CustomResourceLister
	customResourcesSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(version schema.GroupVersion) error

	queue workqueue.RateLimitingInterface
}

func NewCustomResourceDiscoveryController(customResourceInformer informers.CustomResourceInformer, versionHandler *customResourceVersionDiscoveryHandler, groupHandler *customResourceGroupDiscoveryHandler) *CustomResourceDiscoveryController {
	c := &CustomResourceDiscoveryController{
		versionHandler:        versionHandler,
		groupHandler:          groupHandler,
		customResourceLister:  customResourceInformer.Lister(),
		customResourcesSynced: customResourceInformer.Informer().HasSynced,

		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "CustomResourceDiscoveryController"),
	}

	customResourceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResource,
		UpdateFunc: c.updateCustomResource,
		DeleteFunc: c.deleteCustomResource,
	})

	c.syncFn = c.sync

	return c
}

func (c *CustomResourceDiscoveryController) sync(version schema.GroupVersion) error {

	foundVersion := false
	foundGroup := false

	apiVersionsForDiscovery := []metav1.GroupVersionForDiscovery{}
	apiResourcesForDiscovery := []metav1.APIResource{}

	customResources, err := c.customResourceLister.List(labels.Everything())
	if err != nil {
		return err
	}
	for _, customResource := range customResources {
		if customResource.Spec.Group != version.Group {
			continue
		}
		foundGroup = true
		apiVersionsForDiscovery = append(apiVersionsForDiscovery, metav1.GroupVersionForDiscovery{
			GroupVersion: customResource.Spec.Group + "/" + customResource.Spec.Version,
			Version:      customResource.Spec.Version,
		})

		if customResource.Spec.Version != version.Version {
			continue
		}
		foundVersion = true

		apiResourcesForDiscovery = append(apiResourcesForDiscovery, metav1.APIResource{
			Name:         customResource.Spec.Names.Plural,
			SingularName: customResource.Spec.Names.Singular,
			Namespaced:   customResource.Spec.Scope == apiextensions.NamespaceScoped,
			Kind:         customResource.Spec.Names.Kind,
			Verbs:        metav1.Verbs([]string{"delete", "deletecollection", "get", "list", "patch", "create", "update", "watch"}),
			ShortNames:   customResource.Spec.Names.ShortNames,
		})
	}

	if !foundGroup {
		c.groupHandler.unsetDiscovery(version.Group)
		c.versionHandler.unsetDiscovery(version)
		return nil
	}

	apiGroup := metav1.APIGroup{
		Name:             version.Group,
		Versions:         apiVersionsForDiscovery,
		PreferredVersion: apiVersionsForDiscovery[0],
	}
	c.groupHandler.setDiscovery(version.Group, discovery.NewAPIGroupHandler(Codecs, apiGroup))

	if !foundVersion {
		c.versionHandler.unsetDiscovery(version)
		return nil
	}
	c.versionHandler.setDiscovery(version, discovery.NewAPIVersionHandler(Codecs, version, discovery.APIResourceListerFunc(func() []metav1.APIResource {
		return apiResourcesForDiscovery
	})))

	return nil
}

func (c *CustomResourceDiscoveryController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer glog.Infof("Shutting down CustomResourceDiscoveryController")

	glog.Infof("Starting CustomResourceDiscoveryController")

	if !cache.WaitForCacheSync(stopCh, c.customResourcesSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	// only start one worker thread since its a slow moving API
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *CustomResourceDiscoveryController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *CustomResourceDiscoveryController) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncFn(key.(schema.GroupVersion))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *CustomResourceDiscoveryController) enqueue(obj *apiextensions.CustomResource) {
	c.queue.Add(schema.GroupVersion{Group: obj.Spec.Group, Version: obj.Spec.Version})
}

func (c *CustomResourceDiscoveryController) addCustomResource(obj interface{}) {
	castObj := obj.(*apiextensions.CustomResource)
	glog.V(4).Infof("Adding %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *CustomResourceDiscoveryController) updateCustomResource(obj, _ interface{}) {
	castObj := obj.(*apiextensions.CustomResource)
	glog.V(4).Infof("Updating %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *CustomResourceDiscoveryController) deleteCustomResource(obj interface{}) {
	castObj, ok := obj.(*apiextensions.CustomResource)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiextensions.CustomResource)
		if !ok {
			glog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	glog.V(4).Infof("Deleting %q", castObj.Name)
	c.enqueue(castObj)
}
