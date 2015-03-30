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

	"github.com/GoogleCloudPlatform/lmktfy/pkg/api"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/client"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/client/cache"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/fields"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/labels"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/runtime"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/util"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/watch"
	"github.com/golang/glog"
)

// NamespaceManager is responsible for performing actions dependent upon a namespace phase
type NamespaceManager struct {
	lmktfyClient client.Interface
	store      cache.Store
	syncTime   <-chan time.Time

	// To allow injection for testing.
	syncHandler func(namespace api.Namespace) error
}

// NewNamespaceManager creates a new NamespaceManager
func NewNamespaceManager(lmktfyClient client.Interface) *NamespaceManager {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return lmktfyClient.Namespaces().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return lmktfyClient.Namespaces().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.Namespace{},
		store,
		0,
	)
	reflector.Run()
	nm := &NamespaceManager{
		lmktfyClient: lmktfyClient,
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

// finalize will finalize the namespace for lmktfy
func finalize(lmktfyClient client.Interface, namespace api.Namespace) (*api.Namespace, error) {
	namespaceFinalize := api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            namespace.Name,
			ResourceVersion: namespace.ResourceVersion,
		},
		Spec: api.NamespaceSpec{},
	}
	finalizerSet := util.NewStringSet()
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != api.FinalizerLMKTFY {
			finalizerSet.Insert(string(namespace.Spec.Finalizers[i]))
		}
	}
	namespaceFinalize.Spec.Finalizers = make([]api.FinalizerName, len(finalizerSet), len(finalizerSet))
	for _, value := range finalizerSet.List() {
		namespaceFinalize.Spec.Finalizers = append(namespaceFinalize.Spec.Finalizers, api.FinalizerName(value))
	}
	return lmktfyClient.Namespaces().Finalize(&namespaceFinalize)
}

// deleteAllContent will delete all content known to the system in a namespace
func deleteAllContent(lmktfyClient client.Interface, namespace string) (err error) {
	err = deleteServices(lmktfyClient, namespace)
	if err != nil {
		return err
	}
	err = deleteReplicationControllers(lmktfyClient, namespace)
	if err != nil {
		return err
	}
	err = deletePods(lmktfyClient, namespace)
	if err != nil {
		return err
	}
	err = deleteSecrets(lmktfyClient, namespace)
	if err != nil {
		return err
	}
	err = deleteLimitRanges(lmktfyClient, namespace)
	if err != nil {
		return err
	}
	err = deleteResourceQuotas(lmktfyClient, namespace)
	if err != nil {
		return err
	}
	err = deleteEvents(lmktfyClient, namespace)
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
		result, err := nm.lmktfyClient.Namespaces().Status(&newNamespace)
		if err != nil {
			return err
		}
		// work with the latest copy so we can proceed to clean up right away without another interval
		namespace = *result
	}

	// if the namespace is already finalized, delete it
	if finalized(namespace) {
		err = nm.lmktfyClient.Namespaces().Delete(namespace.Name)
		return err
	}

	// there may still be content for us to remove
	err = deleteAllContent(nm.lmktfyClient, namespace.Name)
	if err != nil {
		return err
	}

	// we have removed content, so mark it finalized by us
	result, err := finalize(nm.lmktfyClient, namespace)
	if err != nil {
		return err
	}

	// now check if all finalizers have reported that we delete now
	if finalized(*result) {
		err = nm.lmktfyClient.Namespaces().Delete(namespace.Name)
		return err
	}

	return nil
}

func deleteLimitRanges(lmktfyClient client.Interface, ns string) error {
	items, err := lmktfyClient.LimitRanges(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := lmktfyClient.LimitRanges(ns).Delete(items.Items[i].Name)
		if err != nil {
			return err
		}
	}
	return nil
}

func deleteResourceQuotas(lmktfyClient client.Interface, ns string) error {
	resourceQuotas, err := lmktfyClient.ResourceQuotas(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range resourceQuotas.Items {
		err := lmktfyClient.ResourceQuotas(ns).Delete(resourceQuotas.Items[i].Name)
		if err != nil {
			return err
		}
	}
	return nil
}

func deleteServices(lmktfyClient client.Interface, ns string) error {
	items, err := lmktfyClient.Services(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := lmktfyClient.Services(ns).Delete(items.Items[i].Name)
		if err != nil {
			return err
		}
	}
	return nil
}

func deleteReplicationControllers(lmktfyClient client.Interface, ns string) error {
	items, err := lmktfyClient.ReplicationControllers(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := lmktfyClient.ReplicationControllers(ns).Delete(items.Items[i].Name)
		if err != nil {
			return err
		}
	}
	return nil
}

func deletePods(lmktfyClient client.Interface, ns string) error {
	items, err := lmktfyClient.Pods(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := lmktfyClient.Pods(ns).Delete(items.Items[i].Name)
		if err != nil {
			return err
		}
	}
	return nil
}

func deleteEvents(lmktfyClient client.Interface, ns string) error {
	items, err := lmktfyClient.Events(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := lmktfyClient.Events(ns).Delete(items.Items[i].Name)
		if err != nil {
			return err
		}
	}
	return nil
}

func deleteSecrets(lmktfyClient client.Interface, ns string) error {
	items, err := lmktfyClient.Secrets(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := lmktfyClient.Secrets(ns).Delete(items.Items[i].Name)
		if err != nil {
			return err
		}
	}
	return nil
}
