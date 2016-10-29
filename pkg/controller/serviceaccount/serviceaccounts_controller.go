/*
Copyright 2014 The Kubernetes Authors.

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

package serviceaccount

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/informers"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
)

// nameIndexFunc is an index function that indexes based on an object's name
func nameIndexFunc(obj interface{}) ([]string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return []string{""}, fmt.Errorf("object has no meta: %v", err)
	}
	return []string{meta.GetName()}, nil
}

// ServiceAccountsControllerOptions contains options for running a ServiceAccountsController
type ServiceAccountsControllerOptions struct {
	// ServiceAccounts is the list of service accounts to ensure exist in every namespace
	ServiceAccounts []api.ServiceAccount

	// ServiceAccountResync is the interval between full resyncs of ServiceAccounts.
	// If non-zero, all service accounts will be re-listed this often.
	// Otherwise, re-list will be delayed as long as possible (until the watch is closed or times out).
	ServiceAccountResync time.Duration

	// NamespaceResync is the interval between full resyncs of Namespaces.
	// If non-zero, all namespaces will be re-listed this often.
	// Otherwise, re-list will be delayed as long as possible (until the watch is closed or times out).
	NamespaceResync time.Duration
}

func DefaultServiceAccountsControllerOptions() ServiceAccountsControllerOptions {
	return ServiceAccountsControllerOptions{
		ServiceAccounts: []api.ServiceAccount{
			{ObjectMeta: api.ObjectMeta{Name: "default"}},
		},
	}
}

// NewServiceAccountsController returns a new *ServiceAccountsController.
func NewServiceAccountsController(saInformer informers.ServiceAccountInformer, nsInformer informers.NamespaceInformer, cl clientset.Interface, options ServiceAccountsControllerOptions) *ServiceAccountsController {
	e := &ServiceAccountsController{
		client:                  cl,
		serviceAccountsToEnsure: options.ServiceAccounts,
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "serviceaccount"),
	}
	if cl != nil && cl.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("serviceaccount_controller", cl.Core().RESTClient().GetRateLimiter())
	}

	saInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		DeleteFunc: e.serviceAccountDeleted,
	})
	nsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    e.namespaceAdded,
		UpdateFunc: e.namespaceUpdated,
	})

	e.saSynced = saInformer.Informer().HasSynced
	e.saLister = saInformer.Lister()
	e.nsSynced = nsInformer.Informer().HasSynced
	e.nsLister = nsInformer.Lister()

	e.syncHandler = e.syncNamespace

	return e
}

// ServiceAccountsController manages ServiceAccount objects inside Namespaces
type ServiceAccountsController struct {
	client                  clientset.Interface
	serviceAccountsToEnsure []api.ServiceAccount

	// To allow injection for testing.
	syncHandler func(key string) error

	saLister *cache.StoreToServiceAccountLister
	nsLister *cache.IndexerToNamespaceLister

	saSynced cache.InformerSynced
	nsSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface
}

func (c *ServiceAccountsController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	glog.Infof("Starting ServiceAccount controller")

	if !cache.WaitForCacheSync(stopCh, c.saSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down ServiceAccount controller")
}

// serviceAccountDeleted reacts to a ServiceAccount deletion by recreating a default ServiceAccount in the namespace if needed
func (c *ServiceAccountsController) serviceAccountDeleted(obj interface{}) {
	sa, ok := obj.(*api.ServiceAccount)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		sa, ok = tombstone.Obj.(*api.ServiceAccount)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a ServiceAccount %#v", obj))
			return
		}
	}
	c.queue.Add(sa.Namespace)
}

// namespaceAdded reacts to a Namespace creation by creating a default ServiceAccount object
func (c *ServiceAccountsController) namespaceAdded(obj interface{}) {
	namespace := obj.(*api.Namespace)
	c.queue.Add(namespace.Name)
}

// namespaceUpdated reacts to a Namespace update (or re-list) by creating a default ServiceAccount in the namespace if needed
func (c *ServiceAccountsController) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*api.Namespace)
	c.queue.Add(newNamespace.Name)
}

func (c *ServiceAccountsController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *ServiceAccountsController) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncHandler(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}
func (c *ServiceAccountsController) syncNamespace(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing namespace %q (%v)", key, time.Now().Sub(startTime))
	}()

	ns, err := c.nsLister.Get(key)
	if apierrs.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if ns.Status.Phase != api.NamespaceActive {
		// If namespace is not active, we shouldn't try to create anything
		return nil
	}

	createFailures := []error{}
	for i := range c.serviceAccountsToEnsure {
		sa := c.serviceAccountsToEnsure[i]
		switch _, err := c.saLister.ServiceAccounts(ns.Name).Get(sa.Name); {
		case err == nil:
			continue
		case apierrs.IsNotFound(err):
		case err != nil:
			return err
		}
		// this is only safe because we never read it and we always write it
		// TODO eliminate this once the fake client can handle creation without NS
		sa.Namespace = ns.Name

		if _, err := c.client.Core().ServiceAccounts(ns.Name).Create(&sa); err != nil && !apierrs.IsAlreadyExists(err) {
			createFailures = append(createFailures, err)
		}
	}

	return utilerrors.Flatten(utilerrors.NewAggregate(createFailures))
}
