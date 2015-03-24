/*
Copyright 2015 Google Inc. All rights reserved.

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

package namespace

import (
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// NamespaceManager is responsible for performing actions dependent upon a namespace phase
type NamespaceManager struct {
	kubeClient client.Interface
	store      cache.Store
	syncTime   <-chan time.Time

	// To allow injection for testing.
	syncHandler func(namespace api.Namespace) error
}

// NewNamespaceManager creates a new NamespaceManager
func NewNamespaceManager(kubeClient client.Interface) *NamespaceManager {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.Namespaces().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.Namespaces().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.Namespace{},
		store,
		0,
	)
	reflector.Run()
	nm := &NamespaceManager{
		kubeClient: kubeClient,
		store:      store,
	}
	// set the synchronization handler
	nm.syncHandler = nm.syncNamespace
	return nm
}

// Run begins syncing at the specified period interval
func (nm *NamespaceManager) Run(period time.Duration) {
	nm.syncTime = time.Tick(period)
	go util.Forever(func() { nm.synchronize() }, period)
}

// Iterate over the each namespace that is in terminating phase and perform necessary clean-up
func (nm *NamespaceManager) synchronize() {
	namespaceObjs := nm.store.List()
	wg := sync.WaitGroup{}
	wg.Add(len(namespaceObjs))
	for ix := range namespaceObjs {
		go func(ix int) {
			defer wg.Done()
			namespace := namespaceObjs[ix].(*api.Namespace)
			glog.V(4).Infof("periodic sync of namespace: %v", namespace.Name)
			err := nm.syncHandler(*namespace)
			if err != nil {
				glog.Errorf("Error synchronizing: %v", err)
			}
		}(ix)
	}
	wg.Wait()
}

// finalized returns true if the spec.finalizers is empty list
func finalized(namespace api.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalize will finalize the namespace for kubernetes
func finalize(kubeClient client.Interface, namespace api.Namespace) (*api.Namespace, error) {
	namespaceFinalize := api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            namespace.Name,
			ResourceVersion: namespace.ResourceVersion,
		},
		Spec: api.NamespaceSpec{},
	}
	finalizerSet := util.NewStringSet()
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != api.FinalizerKubernetes {
			finalizerSet.Insert(string(namespace.Spec.Finalizers[i]))
		}
	}
	namespaceFinalize.Spec.Finalizers = make([]api.FinalizerName, len(finalizerSet), len(finalizerSet))
	for _, value := range finalizerSet.List() {
		namespaceFinalize.Spec.Finalizers = append(namespaceFinalize.Spec.Finalizers, api.FinalizerName(value))
	}
	return kubeClient.Namespaces().Finalize(&namespaceFinalize)
}

// deleteAllContent will delete all content known to the system in a namespace
func deleteAllContent(kubeClient client.Interface, namespace string) (err error) {
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
func (nm *NamespaceManager) syncNamespace(namespace api.Namespace) (err error) {
	if namespace.DeletionTimestamp == nil {
		return nil
	}

	// if there is a deletion timestamp, and the status is not terminating, then update status
	if !namespace.DeletionTimestamp.IsZero() && namespace.Status.Phase != api.NamespaceTerminating {
		newNamespace := api.Namespace{}
		newNamespace.ObjectMeta = namespace.ObjectMeta
		newNamespace.Status = namespace.Status
		newNamespace.Status.Phase = api.NamespaceTerminating
		result, err := nm.kubeClient.Namespaces().Status(&newNamespace)
		if err != nil {
			return err
		}
		// work with the latest copy so we can proceed to clean up right away without another interval
		namespace = *result
	}

	// if the namespace is already finalized, delete it
	if finalized(namespace) {
		err = nm.kubeClient.Namespaces().Delete(namespace.Name)
		return err
	}

	// there may still be content for us to remove
	err = deleteAllContent(nm.kubeClient, namespace.Name)
	if err != nil {
		return err
	}

	// we have removed content, so mark it finalized by us
	result, err := finalize(nm.kubeClient, namespace)
	if err != nil {
		return err
	}

	// now check if all finalizers have reported that we delete now
	if finalized(*result) {
		err = nm.kubeClient.Namespaces().Delete(namespace.Name)
		return err
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
		if err != nil {
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
		if err != nil {
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
		if err != nil {
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
		if err != nil {
			return err
		}
	}
	return nil
}

func deletePods(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Pods(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Pods(ns).Delete(items.Items[i].Name)
		if err != nil {
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
		if err != nil {
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
		if err != nil {
			return err
		}
	}
	return nil
}
