/*
Copyright 2018 The Kubernetes Authors.

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

package runtimeclass

import (
	"errors"
	"fmt"

	apinodev1 "k8s.io/api/node/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	nodev1 "k8s.io/client-go/listers/node/v1"
	"k8s.io/client-go/tools/cache"
)

// ErrNotSynced reports that the RuntimeClass cache has not completed its
// initial list, so a missing RuntimeClass may still exist.
var ErrNotSynced = errors.New("RuntimeClass cache has not synced")

// Manager caches RuntimeClass API objects, and provides accessors to the Kubelet.
type Manager struct {
	informerFactory informers.SharedInformerFactory
	lister          nodev1.RuntimeClassLister
	hasSynced       cache.InformerSynced
}

// NewManager returns a new RuntimeClass Manager. Run must be called before the manager can be used.
func NewManager(client clientset.Interface) *Manager {
	const resyncPeriod = 0

	factory := informers.NewSharedInformerFactory(client, resyncPeriod)
	informer := factory.Node().V1().RuntimeClasses()

	return &Manager{
		informerFactory: factory,
		lister:          informer.Lister(),
		hasSynced:       informer.Informer().HasSynced,
	}
}

// Start starts syncing the RuntimeClass cache with the apiserver.
func (m *Manager) Start(stopCh <-chan struct{}) {
	m.informerFactory.Start(stopCh)
}

// WaitForCacheSync exposes the WaitForCacheSync method on the informer factory for testing
// purposes.
func (m *Manager) WaitForCacheSync(stopCh <-chan struct{}) {
	m.informerFactory.WaitForCacheSync(stopCh)
}

// LookupRuntimeHandler returns the RuntimeHandler string associated with the given RuntimeClass
// name (or the default of "" for nil). If the RuntimeClass is not found, it returns an
// errors.NotFound error.
func (m *Manager) LookupRuntimeHandler(runtimeClassName *string) (string, error) {
	if runtimeClassName == nil || *runtimeClassName == "" {
		// The default RuntimeClass always resolves to the empty runtime handler.
		return "", nil
	}

	name := *runtimeClassName

	rc, err := m.lister.Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return "", err
		}
		return "", fmt.Errorf("failed to lookup RuntimeClass %s: %v", name, err)
	}

	return rc.Handler, nil
}

// GetRuntimeClass returns the cached RuntimeClass with the given name. Callers
// must not modify the returned object. It returns ErrNotSynced before the
// initial list completes, so callers can retry instead of treating a
// RuntimeClass that the cache has not received yet as deleted.
func (m *Manager) GetRuntimeClass(name string) (*apinodev1.RuntimeClass, error) {
	if !m.hasSynced() {
		return nil, ErrNotSynced
	}
	return m.lister.Get(name)
}
