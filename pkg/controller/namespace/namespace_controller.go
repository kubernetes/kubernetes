/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package namespacecontroller

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// NamespaceController is responsible for performing actions dependent upon a namespace phase
type NamespaceController struct {
	controller     *framework.Controller
	StopEverything chan struct{}
}

// NewNamespaceController creates a new NamespaceController
func NewNamespaceController(kubeClient client.Interface, resyncPeriod time.Duration) *NamespaceController {
	_, controller := framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.Namespaces().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.Namespaces().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.Namespace{},
		resyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				namespace := obj.(*api.Namespace)
				err := syncNamespace(kubeClient, *namespace)
				if err != nil {
					glog.Error(err)
				}
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				namespace := newObj.(*api.Namespace)
				err := syncNamespace(kubeClient, *namespace)
				if err != nil {
					glog.Error(err)
				}
			},
		},
	)

	return &NamespaceController{
		controller: controller,
	}
}

// Run begins observing the system.  It starts a goroutine and returns immediately.
func (nm *NamespaceController) Run() {
	if nm.StopEverything == nil {
		nm.StopEverything = make(chan struct{})
		go nm.controller.Run(nm.StopEverything)
	}
}

// Stop gracefully shutsdown this controller
func (nm *NamespaceController) Stop() {
	if nm.StopEverything != nil {
		close(nm.StopEverything)
		nm.StopEverything = nil
	}
}

// finalized returns true if the spec.finalizers is empty list
func finalized(namespace api.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalize will finalize the namespace for kubernetes
func finalize(kubeClient client.Interface, namespace api.Namespace) (*api.Namespace, error) {
	namespaceFinalize := api.Namespace{}
	namespaceFinalize.ObjectMeta = namespace.ObjectMeta
	namespaceFinalize.Spec = namespace.Spec
	finalizerSet := util.NewStringSet()
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != api.FinalizerKubernetes {
			finalizerSet.Insert(string(namespace.Spec.Finalizers[i]))
		}
	}
	namespaceFinalize.Spec.Finalizers = make([]api.FinalizerName, 0, len(finalizerSet))
	for _, value := range finalizerSet.List() {
		namespaceFinalize.Spec.Finalizers = append(namespaceFinalize.Spec.Finalizers, api.FinalizerName(value))
	}
	return kubeClient.Namespaces().Finalize(&namespaceFinalize)
}

// deleteAllContent will delete all content known to the system in a namespace
func deleteAllContent(kubeClient client.Interface, namespace string) (err error) {
	err = deleteServiceAccounts(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deleteServices(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deleteReplicationControllers(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deletePods(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deleteSecrets(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deletePersistentVolumeClaims(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deleteLimitRanges(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deleteResourceQuotas(kubeClient, namespace)
	if err != nil {
		return err
	}
	err = deleteEvents(kubeClient, namespace)
	if err != nil {
		return err
	}
	return nil
}

// syncNamespace makes namespace life-cycle decisions
func syncNamespace(kubeClient client.Interface, namespace api.Namespace) (err error) {
	if namespace.DeletionTimestamp == nil {
		return nil
	}

	// if there is a deletion timestamp, and the status is not terminating, then update status
	if !namespace.DeletionTimestamp.IsZero() && namespace.Status.Phase != api.NamespaceTerminating {
		newNamespace := api.Namespace{}
		newNamespace.ObjectMeta = namespace.ObjectMeta
		newNamespace.Status = namespace.Status
		newNamespace.Status.Phase = api.NamespaceTerminating
		result, err := kubeClient.Namespaces().Status(&newNamespace)
		if err != nil {
			return err
		}
		// work with the latest copy so we can proceed to clean up right away without another interval
		namespace = *result
	}

	// if the namespace is already finalized, delete it
	if finalized(namespace) {
		err = kubeClient.Namespaces().Delete(namespace.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		return nil
	}

	// there may still be content for us to remove
	err = deleteAllContent(kubeClient, namespace.Name)
	if err != nil {
		return err
	}

	// we have removed content, so mark it finalized by us
	result, err := finalize(kubeClient, namespace)
	if err != nil {
		return err
	}

	// now check if all finalizers have reported that we delete now
	if finalized(*result) {
		err = kubeClient.Namespaces().Delete(namespace.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}

	return nil
}

func deleteLimitRanges(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.LimitRanges(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.LimitRanges(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteResourceQuotas(kubeClient client.Interface, ns string) error {
	resourceQuotas, err := kubeClient.ResourceQuotas(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range resourceQuotas.Items {
		err := kubeClient.ResourceQuotas(ns).Delete(resourceQuotas.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteServiceAccounts(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.ServiceAccounts(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.ServiceAccounts(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteServices(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Services(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Services(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteReplicationControllers(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.ReplicationControllers(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.ReplicationControllers(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deletePods(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Pods(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Pods(ns).Delete(items.Items[i].Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteEvents(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Events(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Events(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteSecrets(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Secrets(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Secrets(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deletePersistentVolumeClaims(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.PersistentVolumeClaims(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.PersistentVolumeClaims(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}
