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

package thirdparty

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	crdinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	crdlisters "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kubernetes/pkg/apis/extensions"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/extensions/internalversion"
	listers "k8s.io/kubernetes/pkg/client/listers/extensions/internalversion"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
)

// AutoAPIServiceRegistration is an interface which callers can re-declare locally and properly cast to for
// adding and removing APIServices
type AutoAPIServiceRegistration interface {
	// AddAPIServiceToSync adds an API service to auto-register.
	AddAPIServiceToSync(in *apiregistration.APIService)
	// RemoveAPIServiceToSync removes an API service to auto-register.
	RemoveAPIServiceToSync(name string)
}

type tprRegistrationController struct {
	tprLister listers.ThirdPartyResourceLister
	tprSynced cache.InformerSynced
	crdLister crdlisters.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	apiServiceRegistration AutoAPIServiceRegistration

	syncHandler func(groupVersion schema.GroupVersion) error

	// queue is where incoming work is placed to de-dup and to allow "easy" rate limited requeues on errors
	// this is actually keyed by a groupVersion
	queue workqueue.RateLimitingInterface
}

// NewAutoRegistrationController returns a controller which will register TPR GroupVersions with the auto APIService registration
// controller so they automatically stay in sync.
// In order to stay sane with both TPR and CRD present, we have a single controller that manages both.  When choosing whether to have an
// APIService, we simply iterate through both.
func NewAutoRegistrationController(tprInformer informers.ThirdPartyResourceInformer, crdinformer crdinformers.CustomResourceDefinitionInformer, apiServiceRegistration AutoAPIServiceRegistration) *tprRegistrationController {
	c := &tprRegistrationController{
		tprLister:              tprInformer.Lister(),
		tprSynced:              tprInformer.Informer().HasSynced,
		crdLister:              crdinformer.Lister(),
		crdSynced:              crdinformer.Informer().HasSynced,
		apiServiceRegistration: apiServiceRegistration,
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "tpr-autoregister"),
	}
	c.syncHandler = c.handleVersionUpdate

	tprInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			cast := obj.(*extensions.ThirdPartyResource)
			c.enqueueTPR(cast)
		},
		UpdateFunc: func(_, obj interface{}) {
			cast := obj.(*extensions.ThirdPartyResource)
			c.enqueueTPR(cast)
		},
		DeleteFunc: func(obj interface{}) {
			cast, ok := obj.(*extensions.ThirdPartyResource)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					glog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
					return
				}
				cast, ok = tombstone.Obj.(*extensions.ThirdPartyResource)
				if !ok {
					glog.V(2).Infof("Tombstone contained unexpected object: %#v", obj)
					return
				}
			}
			c.enqueueTPR(cast)
		},
	})

	crdinformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			cast := obj.(*apiextensions.CustomResourceDefinition)
			c.enqueueCRD(cast)
		},
		UpdateFunc: func(_, obj interface{}) {
			cast := obj.(*apiextensions.CustomResourceDefinition)
			c.enqueueCRD(cast)
		},
		DeleteFunc: func(obj interface{}) {
			cast, ok := obj.(*apiextensions.CustomResourceDefinition)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					glog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
					return
				}
				cast, ok = tombstone.Obj.(*apiextensions.CustomResourceDefinition)
				if !ok {
					glog.V(2).Infof("Tombstone contained unexpected object: %#v", obj)
					return
				}
			}
			c.enqueueCRD(cast)
		},
	})

	return c
}

func (c *tprRegistrationController) Run(threadiness int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	// make sure the work queue is shutdown which will trigger workers to end
	defer c.queue.ShutDown()

	glog.Infof("Starting tpr-autoregister controller")
	defer glog.Infof("Shutting down tpr-autoregister controller")

	// wait for your secondary caches to fill before starting your work
	if !controller.WaitForCacheSync("tpr-autoregister", stopCh, c.tprSynced) {
		return
	}

	// start up your worker threads based on threadiness.  Some controllers have multiple kinds of workers
	for i := 0; i < threadiness; i++ {
		// runWorker will loop until "something bad" happens.  The .Until will then rekick the worker
		// after one second
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	// wait until we're told to stop
	<-stopCh
}

func (c *tprRegistrationController) runWorker() {
	// hot loop until we're told to stop.  processNextWorkItem will automatically wait until there's work
	// available, so we don't worry about secondary waits
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *tprRegistrationController) processNextWorkItem() bool {
	// pull the next work item from queue.  It should be a key we use to lookup something in a cache
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	// you always have to indicate to the queue that you've completed a piece of work
	defer c.queue.Done(key)

	// do your work on the key.  This method will contains your "do stuff" logic
	err := c.syncHandler(key.(schema.GroupVersion))
	if err == nil {
		// if you had no error, tell the queue to stop tracking history for your key.  This will
		// reset things like failure counts for per-item rate limiting
		c.queue.Forget(key)
		return true
	}

	// there was a failure so be sure to report it.  This method allows for pluggable error handling
	// which can be used for things like cluster-monitoring
	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))
	// since we failed, we should requeue the item to work on later.  This method will add a backoff
	// to avoid hotlooping on particular items (they're probably still not going to work right away)
	// and overall controller protection (everything I've done is broken, this controller needs to
	// calm down or it can starve other useful work) cases.
	c.queue.AddRateLimited(key)

	return true
}

func (c *tprRegistrationController) enqueueTPR(tpr *extensions.ThirdPartyResource) {
	_, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(tpr)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	for _, version := range tpr.Versions {
		c.queue.Add(schema.GroupVersion{Group: group, Version: version.Name})
	}
}

func (c *tprRegistrationController) enqueueCRD(crd *apiextensions.CustomResourceDefinition) {
	c.queue.Add(schema.GroupVersion{Group: crd.Spec.Group, Version: crd.Spec.Version})
}

func (c *tprRegistrationController) handleVersionUpdate(groupVersion schema.GroupVersion) error {
	found := false
	apiServiceName := groupVersion.Version + "." + groupVersion.Group

	// check all TPRs.  There shouldn't that many, but if we have problems later we can index them
	tprs, err := c.tprLister.List(labels.Everything())
	if err != nil {
		return err
	}
	for _, tpr := range tprs {
		_, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(tpr)
		if err != nil {
			return err
		}
		for _, version := range tpr.Versions {
			if version.Name == groupVersion.Version && group == groupVersion.Group {
				found = true
				break
			}
		}
	}

	// check all CRDs.  There shouldn't that many, but if we have problems later we can index them
	crds, err := c.crdLister.List(labels.Everything())
	if err != nil {
		return err
	}
	for _, crd := range crds {
		if crd.Spec.Version == groupVersion.Version && crd.Spec.Group == groupVersion.Group {
			found = true
			break
		}
	}

	if !found {
		c.apiServiceRegistration.RemoveAPIServiceToSync(apiServiceName)
		return nil
	}

	c.apiServiceRegistration.AddAPIServiceToSync(&apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: apiServiceName},
		Spec: apiregistration.APIServiceSpec{
			Group:                groupVersion.Group,
			Version:              groupVersion.Version,
			GroupPriorityMinimum: 1000, // TPRs should have relatively low priority
			VersionPriority:      100,  // TPRs should have relatively low priority
		},
	})

	return nil
}
