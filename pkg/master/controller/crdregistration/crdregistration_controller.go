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

package crdRegistrationController

import (
	"github.com/golang/glog"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	crdinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	crdlisters "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/staging/src/k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/staging/src/k8s.io/apimachinery/pkg/util/wait"
	"time"
)

// AutoAPIServiceRegistration is an interface which callers can re-declare locally and properly cast to for
// adding and removing APIServices
type AutoAPIServiceRegistration interface {
	// AddAPIServiceToSync adds an API service to auto-register.
	AddAPIServiceToSync(in *apiregistration.APIService)
	// RemoveAPIServiceToSync removes an API service to auto-register.
	RemoveAPIServiceToSync(name string)
}

type crdRegistrationController struct {
	crdLister crdlisters.CustomResourceDefinitionLister
	crdSynced cache.InformerSynced

	apiServiceRegistration AutoAPIServiceRegistration

	syncedInitialSet chan struct{}
}

// NewAutoRegistrationController returns a controller which will register CRD GroupVersions with the auto APIService registration
// controller so they automatically stay in sync.
func NewAutoRegistrationController(crdinformer crdinformers.CustomResourceDefinitionInformer, apiServiceRegistration AutoAPIServiceRegistration) *crdRegistrationController {
	c := &crdRegistrationController{
		crdLister:              crdinformer.Lister(),
		crdSynced:              crdinformer.Informer().HasSynced,
		apiServiceRegistration: apiServiceRegistration,
		syncedInitialSet:       make(chan struct{}),
	}

	crdinformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.WaitForInitialSync()
			cast := obj.(*apiextensions.CustomResourceDefinition)
			c.addCRDAPIService(cast)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.WaitForInitialSync()
			c.updateCRDAPIService(oldObj.(*apiextensions.CustomResourceDefinition), newObj.(*apiextensions.CustomResourceDefinition))
		},
		DeleteFunc: func(obj interface{}) {
			c.WaitForInitialSync()
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
			c.removeCRDAPIService(cast)
		},
	})
	return c
}

func (c *crdRegistrationController) Run(stopCh <-chan struct{}) {
	// wait for your secondary caches to fill before starting your work
	if !controller.WaitForCacheSync("crd-autoregister", stopCh, c.crdSynced) {
		return
	}

	// process each item in the list once
	if crds, err := c.crdLister.List(labels.Everything()); err != nil {
		utilruntime.HandleError(err)
	} else {
		for _, crd := range crds {
			c.addCRDAPIService(crd)
		}
	}

	close(c.syncedInitialSet)
}

// WaitForInitialSync blocks until the initial set of CRD resources has been processed
func (c *crdRegistrationController) WaitForInitialSync() {
	<-c.syncedInitialSet
}

// TODO(mehdy): pass only group/version instead of whole CRD object? This approach is more versions friendly though.
func (c *crdRegistrationController) addCRDAPIService(crd *apiextensions.CustomResourceDefinition) {
	apiServiceName := crd.Spec.Version + "." + crd.Spec.Group
	c.apiServiceRegistration.AddAPIServiceToSync(&apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: apiServiceName},
		Spec: apiregistration.APIServiceSpec{
			Group:                crd.Spec.Group,
			Version:              crd.Spec.Version,
			GroupPriorityMinimum: 1000, // CRDs should have relatively low priority
			VersionPriority:      100,  // CRDs should have relatively low priority
		},
	})
}

func (c *crdRegistrationController) removeCRDAPIService(crd *apiextensions.CustomResourceDefinition) {
	c.apiServiceRegistration.RemoveAPIServiceToSync(crd.Spec.Version + "." + crd.Spec.Group)
}

func (c *crdRegistrationController) updateCRDAPIService(oldCrd, newCrd *apiextensions.CustomResourceDefinition) {
	// No-op
	// CRD Group and Version cannot be changed.
}
