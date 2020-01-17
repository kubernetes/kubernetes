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

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog"
)

// ServiceAccountsControllerOptions contains options for running a ServiceAccountsController
type ServiceAccountsControllerOptions struct {
	// ServiceAccounts is the list of service accounts to ensure exist in every namespace
	ServiceAccounts []v1.ServiceAccount

	// ServiceAccountResync is the interval between full resyncs of ServiceAccounts.
	// If non-zero, all service accounts will be re-listed this often.
	// Otherwise, re-list will be delayed as long as possible (until the watch is closed or times out).
	ServiceAccountResync time.Duration

	// NamespaceResync is the interval between full resyncs of Namespaces.
	// If non-zero, all namespaces will be re-listed this often.
	// Otherwise, re-list will be delayed as long as possible (until the watch is closed or times out).
	NamespaceResync time.Duration
}

// DefaultServiceAccountsControllerOptions returns the default options for creating a ServiceAccountsController.
func DefaultServiceAccountsControllerOptions() ServiceAccountsControllerOptions {
	return ServiceAccountsControllerOptions{
		ServiceAccounts: []v1.ServiceAccount{
			{ObjectMeta: metav1.ObjectMeta{Name: "default"}},
		},
	}
}

// NewServiceAccountsController returns a new *ServiceAccountsController.
func NewServiceAccountsController(saInformer coreinformers.ServiceAccountInformer, nsInformer coreinformers.NamespaceInformer, cl clientset.Interface, options ServiceAccountsControllerOptions) (*ServiceAccountsController, error) {
	e := &ServiceAccountsController{
		client:                  cl,
		serviceAccountsToEnsure: options.ServiceAccounts,
		queue:                   workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "serviceaccount"),
	}
	if cl != nil && cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := ratelimiter.RegisterMetricAndTrackRateLimiterUsage("serviceaccount_controller", cl.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	saInformer.Informer().AddEventHandlerWithResyncPeriod(cache.ResourceEventHandlerFuncs{
		DeleteFunc: e.serviceAccountDeleted,
	}, options.ServiceAccountResync)
	e.saLister = saInformer.Lister()
	e.saListerSynced = saInformer.Informer().HasSynced

	nsInformer.Informer().AddEventHandlerWithResyncPeriod(cache.ResourceEventHandlerFuncs{
		AddFunc:    e.namespaceAdded,
		UpdateFunc: e.namespaceUpdated,
	}, options.NamespaceResync)
	e.nsLister = nsInformer.Lister()
	e.nsListerSynced = nsInformer.Informer().HasSynced

	e.syncHandler = e.syncNamespace

	return e, nil
}

// ServiceAccountsController manages ServiceAccount objects inside Namespaces
type ServiceAccountsController struct {
	client                  clientset.Interface
	serviceAccountsToEnsure []v1.ServiceAccount

	// To allow injection for testing.
	syncHandler func(key string) error

	saLister       corelisters.ServiceAccountLister
	saListerSynced cache.InformerSynced

	nsLister       corelisters.NamespaceLister
	nsListerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface
}

// Run runs the ServiceAccountsController blocks until receiving signal from stopCh.
func (c *ServiceAccountsController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting service account controller")
	defer klog.Infof("Shutting down service account controller")

	if !cache.WaitForNamedCacheSync("service account", stopCh, c.saListerSynced, c.nsListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

// serviceAccountDeleted reacts to a ServiceAccount deletion by recreating a default ServiceAccount in the namespace if needed
func (c *ServiceAccountsController) serviceAccountDeleted(obj interface{}) {
	sa, ok := obj.(*v1.ServiceAccount)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		sa, ok = tombstone.Obj.(*v1.ServiceAccount)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a ServiceAccount %#v", obj))
			return
		}
	}
	c.queue.Add(sa.Namespace)
}

// namespaceAdded reacts to a Namespace creation by creating a default ServiceAccount object
func (c *ServiceAccountsController) namespaceAdded(obj interface{}) {
	namespace := obj.(*v1.Namespace)
	c.queue.Add(namespace.Name)
}

// namespaceUpdated reacts to a Namespace update (or re-list) by creating a default ServiceAccount in the namespace if needed
func (c *ServiceAccountsController) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*v1.Namespace)
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
		klog.V(4).Infof("Finished syncing namespace %q (%v)", key, time.Since(startTime))
	}()

	ns, err := c.nsLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if ns.Status.Phase != v1.NamespaceActive {
		// If namespace is not active, we shouldn't try to create anything
		return nil
	}

	createFailures := []error{}
	for _, sa := range c.serviceAccountsToEnsure {
		switch _, err := c.saLister.ServiceAccounts(ns.Name).Get(sa.Name); {
		case err == nil:
			continue
		case apierrors.IsNotFound(err):
		case err != nil:
			return err
		}
		// this is only safe because we never read it and we always write it
		// TODO eliminate this once the fake client can handle creation without NS
		sa.Namespace = ns.Name

		if _, err := c.client.CoreV1().ServiceAccounts(ns.Name).Create(&sa); err != nil && !apierrors.IsAlreadyExists(err) {
			// we can safely ignore terminating namespace errors
			if !apierrors.HasStatusCause(err, v1.NamespaceTerminatingCause) {
				createFailures = append(createFailures, err)
			}
		}
	}

	return utilerrors.Flatten(utilerrors.NewAggregate(createFailures))
}
