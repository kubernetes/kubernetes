/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// nameIndexFunc is an index function that indexes based on an object's name
func nameIndexFunc(obj interface{}) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("object has no meta: %v", err)
	}
	return meta.Name(), nil
}

type ServiceAccountControllerOptions struct {
	Name                 string
	ServiceAccountResync time.Duration
	NamespaceResync      time.Duration
}

func DefaultServiceAccountControllerOptions() ServiceAccountControllerOptions {
	return ServiceAccountControllerOptions{Name: "default"}
}

// NewServiceAccountsController returns a new *ServiceAccountsController.
func NewServiceAccountsController(cl *client.Client, options ServiceAccountControllerOptions) *ServiceAccountsController {
	e := &ServiceAccountsController{
		client: cl,
		name:   options.Name,
	}

	accountSelector := fields.SelectorFromSet(map[string]string{client.ObjectNameField: options.Name})
	e.serviceAccounts, e.serviceAccountController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return e.client.ServiceAccounts(api.NamespaceAll).List(labels.Everything(), accountSelector)
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return e.client.ServiceAccounts(api.NamespaceAll).Watch(labels.Everything(), accountSelector, rv)
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
			ListFunc: func() (runtime.Object, error) {
				return e.client.Namespaces().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return e.client.Namespaces().Watch(labels.Everything(), fields.Everything(), rv)
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

	client *client.Client
	name   string

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
	e.createDefaultServiceAccountIfNeeded(serviceAccount.Namespace)
}

// namespaceAdded reacts to a Namespace creation by creating a default ServiceAccount object
func (e *ServiceAccountsController) namespaceAdded(obj interface{}) {
	namespace := obj.(*api.Namespace)
	e.createDefaultServiceAccountIfNeeded(namespace.Name)
}

// namespaceUpdated reacts to a Namespace update (or re-list) by creating a default ServiceAccount in the namespace if needed
func (e *ServiceAccountsController) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*api.Namespace)
	e.createDefaultServiceAccountIfNeeded(newNamespace.Name)
}

// createDefaultServiceAccountIfNeeded creates a default ServiceAccount in the given namespace if:
// * it default ServiceAccount does not already exist
// * the specified namespace exists
// * the specified namespace is in the ACTIVE phase
func (e *ServiceAccountsController) createDefaultServiceAccountIfNeeded(namespace string) {
	serviceAccount, err := e.getDefaultServiceAccount(namespace)
	if err != nil {
		glog.Error(err)
		return
	}
	if serviceAccount != nil {
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

	e.createDefaultServiceAccount(namespace)
}

// createDefaultServiceAccount creates a default ServiceAccount in the specified namespace
func (e *ServiceAccountsController) createDefaultServiceAccount(namespace string) {
	serviceAccount := &api.ServiceAccount{}
	serviceAccount.Name = e.name
	serviceAccount.Namespace = namespace
	if _, err := e.client.ServiceAccounts(namespace).Create(serviceAccount); err != nil {
		glog.Error(err)
	}
}

// getDefaultServiceAccount returns the default ServiceAccount for the given namespace
func (e *ServiceAccountsController) getDefaultServiceAccount(namespace string) (*api.ServiceAccount, error) {

	key := &api.ServiceAccount{ObjectMeta: api.ObjectMeta{Namespace: namespace}}
	accounts, err := e.serviceAccounts.Index("namespace", key)
	if err != nil {
		return nil, err
	}

	for _, obj := range accounts {
		serviceAccount := obj.(*api.ServiceAccount)
		if e.name == serviceAccount.Name {
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
