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
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
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
	"k8s.io/klog/v2"
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
func NewServiceAccountsController(ctx context.Context, saInformer coreinformers.ServiceAccountInformer, nsInformer coreinformers.NamespaceInformer, cl clientset.Interface, options ServiceAccountsControllerOptions) (*ServiceAccountsController, error) {
	e := &ServiceAccountsController{
		client:                  cl,
		serviceAccountsToEnsure: options.ServiceAccounts,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "serviceaccount"},
		),
	}

	saHandler, _ := saInformer.Informer().AddEventHandlerWithConfig(ctx, cache.ResourceEventHandlerFuncs{
		DeleteFunc: e.serviceAccountDeleted,
	}, cache.HandlerConfig{ResyncPeriod: options.ServiceAccountResync})
	e.saLister = saInformer.Lister()
	e.saListerSynced = saHandler.HasSynced

	nsHandler, _ := nsInformer.Informer().AddEventHandlerWithConfig(ctx, cache.ResourceEventHandlerFuncs{
		AddFunc:    e.namespaceAdded,
		UpdateFunc: e.namespaceUpdated,
	}, cache.HandlerConfig{ResyncPeriod: options.NamespaceResync})
	e.nsLister = nsInformer.Lister()
	e.nsListerSynced = nsHandler.HasSynced

	e.syncHandler = e.syncNamespace

	return e, nil
}

// ServiceAccountsController manages ServiceAccount objects inside Namespaces
type ServiceAccountsController struct {
	client                  clientset.Interface
	serviceAccountsToEnsure []v1.ServiceAccount

	// To allow injection for testing.
	syncHandler func(ctx context.Context, key string) error

	saLister       corelisters.ServiceAccountLister
	saListerSynced cache.InformerSynced

	nsLister       corelisters.NamespaceLister
	nsListerSynced cache.InformerSynced

	queue workqueue.TypedRateLimitingInterface[string]
}

// Run runs the ServiceAccountsController blocks until receiving signal from stopCh.
func (c *ServiceAccountsController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer c.queue.ShutDown()

	klog.FromContext(ctx).Info("Starting service account controller")
	defer klog.FromContext(ctx).Info("Shutting down service account controller")

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.saListerSynced, c.nsListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

// serviceAccountDeleted reacts to a ServiceAccount deletion by recreating a default ServiceAccount in the namespace if needed
func (c *ServiceAccountsController) serviceAccountDeleted(obj interface{}) {
	sa, ok := obj.(*v1.ServiceAccount)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj)) //nolint:logcheck // Not reached, shouldn't have unknown objects.
			return
		}
		sa, ok = tombstone.Obj.(*v1.ServiceAccount)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a ServiceAccount %#v", obj)) //nolint:logcheck // Not reached, shouldn't have unknown objects.
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

func (c *ServiceAccountsController) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *ServiceAccountsController) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncHandler(ctx, key)
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Syncing failed", "key", key)
	c.queue.AddRateLimited(key)

	return true
}
func (c *ServiceAccountsController) syncNamespace(ctx context.Context, key string) error {
	startTime := time.Now()
	defer func() {
		klog.FromContext(ctx).V(4).Info("Finished syncing namespace", "namespace", key, "duration", time.Since(startTime))
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

		if _, err := c.client.CoreV1().ServiceAccounts(ns.Name).Create(ctx, &sa, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			// we can safely ignore terminating namespace errors
			if !apierrors.HasStatusCause(err, v1.NamespaceTerminatingCause) {
				createFailures = append(createFailures, err)
			}
		}
	}

	return utilerrors.Flatten(utilerrors.NewAggregate(createFailures))
}
