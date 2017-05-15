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

type DiscoveryController struct {
	versionHandler *versionDiscoveryHandler
	groupHandler   *groupDiscoveryHandler

	customResourceDefinitionLister  listers.CustomResourceDefinitionLister
	customResourceDefinitionsSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(version schema.GroupVersion) error

	queue workqueue.RateLimitingInterface
}

func NewDiscoveryController(customResourceDefinitionInformer informers.CustomResourceDefinitionInformer, versionHandler *versionDiscoveryHandler, groupHandler *groupDiscoveryHandler) *DiscoveryController {
	c := &DiscoveryController{
		versionHandler:                  versionHandler,
		groupHandler:                    groupHandler,
		customResourceDefinitionLister:  customResourceDefinitionInformer.Lister(),
		customResourceDefinitionsSynced: customResourceDefinitionInformer.Informer().HasSynced,

		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "DiscoveryController"),
	}

	customResourceDefinitionInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addCustomResourceDefinition,
		UpdateFunc: c.updateCustomResourceDefinition,
		DeleteFunc: c.deleteCustomResourceDefinition,
	})

	c.syncFn = c.sync

	return c
}

func (c *DiscoveryController) sync(version schema.GroupVersion) error {

	apiVersionsForDiscovery := []metav1.GroupVersionForDiscovery{}
	apiResourcesForDiscovery := []metav1.APIResource{}

	customResourceDefinitions, err := c.customResourceDefinitionLister.List(labels.Everything())
	if err != nil {
		return err
	}
	foundVersion := false
	foundGroup := false
	for _, customResourceDefinition := range customResourceDefinitions {
		// TODO add status checking

		if customResourceDefinition.Spec.Group != version.Group {
			continue
		}
		foundGroup = true
		apiVersionsForDiscovery = append(apiVersionsForDiscovery, metav1.GroupVersionForDiscovery{
			GroupVersion: customResourceDefinition.Spec.Group + "/" + customResourceDefinition.Spec.Version,
			Version:      customResourceDefinition.Spec.Version,
		})

		if customResourceDefinition.Spec.Version != version.Version {
			continue
		}
		foundVersion = true

		apiResourcesForDiscovery = append(apiResourcesForDiscovery, metav1.APIResource{
			Name:         customResourceDefinition.Spec.Names.Plural,
			SingularName: customResourceDefinition.Spec.Names.Singular,
			Namespaced:   customResourceDefinition.Spec.Scope == apiextensions.NamespaceScoped,
			Kind:         customResourceDefinition.Spec.Names.Kind,
			Verbs:        metav1.Verbs([]string{"delete", "deletecollection", "get", "list", "patch", "create", "update", "watch"}),
			ShortNames:   customResourceDefinition.Spec.Names.ShortNames,
		})
	}

	if !foundGroup {
		c.groupHandler.unsetDiscovery(version.Group)
		c.versionHandler.unsetDiscovery(version)
		return nil
	}

	apiGroup := metav1.APIGroup{
		Name:     version.Group,
		Versions: apiVersionsForDiscovery,
		// the preferred versions for a group is arbitrary since there cannot be duplicate resources
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

func (c *DiscoveryController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer glog.Infof("Shutting down DiscoveryController")

	glog.Infof("Starting DiscoveryController")

	if !cache.WaitForCacheSync(stopCh, c.customResourceDefinitionsSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	// only start one worker thread since its a slow moving API
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *DiscoveryController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *DiscoveryController) processNextWorkItem() bool {
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

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *DiscoveryController) enqueue(obj *apiextensions.CustomResourceDefinition) {
	c.queue.Add(schema.GroupVersion{Group: obj.Spec.Group, Version: obj.Spec.Version})
}

func (c *DiscoveryController) addCustomResourceDefinition(obj interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	glog.V(4).Infof("Adding customresourcedefinition %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *DiscoveryController) updateCustomResourceDefinition(obj, _ interface{}) {
	castObj := obj.(*apiextensions.CustomResourceDefinition)
	glog.V(4).Infof("Updating customresourcedefinition %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *DiscoveryController) deleteCustomResourceDefinition(obj interface{}) {
	castObj, ok := obj.(*apiextensions.CustomResourceDefinition)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiextensions.CustomResourceDefinition)
		if !ok {
			glog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	glog.V(4).Infof("Deleting customresourcedefinition %q", castObj.Name)
	c.enqueue(castObj)
}
