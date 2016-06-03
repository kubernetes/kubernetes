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
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/pkg/watch"
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
func NewServiceAccountsController(cl clientset.Interface, options ServiceAccountsControllerOptions) *ServiceAccountsController {
	e := &ServiceAccountsController{
		client:                  cl,
		serviceAccountsToEnsure: options.ServiceAccounts,
	}
	if cl != nil && cl.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("serviceaccount_controller", cl.Core().GetRESTClient().GetRateLimiter())
	}
	accountSelector := fields.Everything()
	if len(options.ServiceAccounts) == 1 {
		// If we're maintaining a single account, we can scope the accounts we watch to just that name
		accountSelector = fields.SelectorFromSet(map[string]string{api.ObjectNameField: options.ServiceAccounts[0].Name})
	}
	e.serviceAccounts, e.serviceAccountController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				options.FieldSelector = accountSelector
				return e.client.Core().ServiceAccounts(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				options.FieldSelector = accountSelector
				return e.client.Core().ServiceAccounts(api.NamespaceAll).Watch(options)
			},
		},
		&api.ServiceAccount{},
		options.ServiceAccountResync,
		framework.ResourceEventHandlerFuncs{
			DeleteFunc: e.serviceAccountDeleted,
		},
		cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc},
	)

	e.namespaces, e.namespaceController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return e.client.Core().Namespaces().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return e.client.Core().Namespaces().Watch(options)
			},
		},
		&api.Namespace{},
		options.NamespaceResync,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    e.namespaceAdded,
			UpdateFunc: e.namespaceUpdated,
		},
		cache.Indexers{"name": nameIndexFunc},
	)

	return e
}

// ServiceAccountsController manages ServiceAccount objects inside Namespaces
type ServiceAccountsController struct {
	stopChan chan struct{}

	client                  clientset.Interface
	serviceAccountsToEnsure []api.ServiceAccount

	serviceAccounts cache.Indexer
	namespaces      cache.Indexer

	// Since we join two objects, we'll watch both of them with controllers.
	serviceAccountController *framework.Controller
	namespaceController      *framework.Controller
}

// Runs controller loops and returns immediately
func (e *ServiceAccountsController) Run() {
	if e.stopChan == nil {
		e.stopChan = make(chan struct{})
		go e.serviceAccountController.Run(e.stopChan)
		go e.namespaceController.Run(e.stopChan)
	}
}

// Stop gracefully shuts down this controller
func (e *ServiceAccountsController) Stop() {
	if e.stopChan != nil {
		close(e.stopChan)
		e.stopChan = nil
	}
}

// serviceAccountDeleted reacts to a ServiceAccount deletion by recreating a default ServiceAccount in the namespace if needed
func (e *ServiceAccountsController) serviceAccountDeleted(obj interface{}) {
	serviceAccount, ok := obj.(*api.ServiceAccount)
	if !ok {
		// Unknown type. If we missed a ServiceAccount deletion, the
		// corresponding secrets will be cleaned up during the Secret re-list
		return
	}
	// If the deleted service account is one we're maintaining, recreate it
	for _, sa := range e.serviceAccountsToEnsure {
		if sa.Name == serviceAccount.Name {
			e.createServiceAccountIfNeeded(sa, serviceAccount.Namespace)
		}
	}
}

// namespaceAdded reacts to a Namespace creation by creating a default ServiceAccount object
func (e *ServiceAccountsController) namespaceAdded(obj interface{}) {
	namespace := obj.(*api.Namespace)
	for _, sa := range e.serviceAccountsToEnsure {
		e.createServiceAccountIfNeeded(sa, namespace.Name)
	}
}

// namespaceUpdated reacts to a Namespace update (or re-list) by creating a default ServiceAccount in the namespace if needed
func (e *ServiceAccountsController) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*api.Namespace)
	for _, sa := range e.serviceAccountsToEnsure {
		e.createServiceAccountIfNeeded(sa, newNamespace.Name)
	}
}

// createServiceAccountIfNeeded creates a ServiceAccount with the given name in the given namespace if:
// * the named ServiceAccount does not already exist
// * the specified namespace exists
// * the specified namespace is in the ACTIVE phase
func (e *ServiceAccountsController) createServiceAccountIfNeeded(sa api.ServiceAccount, namespace string) {
	existingServiceAccount, err := e.getServiceAccount(sa.Name, namespace)
	if err != nil {
		glog.Error(err)
		return
	}
	if existingServiceAccount != nil {
		// If service account already exists, it doesn't need to be created
		return
	}

	namespaceObj, err := e.getNamespace(namespace)
	if err != nil {
		glog.Error(err)
		return
	}
	if namespaceObj == nil {
		// If namespace does not exist, no service account is needed
		return
	}
	if namespaceObj.Status.Phase != api.NamespaceActive {
		// If namespace is not active, we shouldn't try to create anything
		return
	}

	e.createServiceAccount(sa, namespace)
}

// createDefaultServiceAccount creates a default ServiceAccount in the specified namespace
func (e *ServiceAccountsController) createServiceAccount(sa api.ServiceAccount, namespace string) {
	sa.Namespace = namespace
	if _, err := e.client.Core().ServiceAccounts(namespace).Create(&sa); err != nil && !apierrs.IsAlreadyExists(err) {
		glog.Error(err)
	}
}

// getServiceAccount returns the ServiceAccount with the given name for the given namespace
func (e *ServiceAccountsController) getServiceAccount(name, namespace string) (*api.ServiceAccount, error) {
	key := &api.ServiceAccount{ObjectMeta: api.ObjectMeta{Namespace: namespace}}
	accounts, err := e.serviceAccounts.Index("namespace", key)
	if err != nil {
		return nil, err
	}

	for _, obj := range accounts {
		serviceAccount := obj.(*api.ServiceAccount)
		if name == serviceAccount.Name {
			return serviceAccount, nil
		}
	}
	return nil, nil
}

// getNamespace returns the Namespace with the given name
func (e *ServiceAccountsController) getNamespace(name string) (*api.Namespace, error) {
	key := &api.Namespace{ObjectMeta: api.ObjectMeta{Name: name}}
	namespaces, err := e.namespaces.Index("name", key)
	if err != nil {
		return nil, err
	}

	if len(namespaces) == 0 {
		return nil, nil
	}
	if len(namespaces) == 1 {
		return namespaces[0].(*api.Namespace), nil
	}
	return nil, fmt.Errorf("%d namespaces with the name %s indexed", len(namespaces), name)
}
