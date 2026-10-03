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
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	nodev1 "k8s.io/client-go/listers/node/v1"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
)

// Manager caches RuntimeClass API objects, and provides accessors to the Kubelet.
type Manager struct {
	client          clientset.Interface
	informerFactory informers.SharedInformerFactory
	lister          nodev1.RuntimeClassLister
}

// NewManager returns a new RuntimeClass Manager. Run must be called before the manager can be used.
func NewManager(client clientset.Interface) *Manager {
	const resyncPeriod = 0

	factory := informers.NewSharedInformerFactory(client, resyncPeriod)
	lister := factory.Node().V1().RuntimeClasses().Lister()

	return &Manager{
		client:          client,
		informerFactory: factory,
		lister:          lister,
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
		if errors.IsNotFound(err) {
			return "", err
		}
		return "", fmt.Errorf("failed to lookup RuntimeClass %s: %v", name, err)
	}

	return rc.Handler, nil
}

// LookupRuntimeHandlerForRestore resolves the handler and option policy from
// one live RuntimeClass, so a recreated class cannot apply its allowlist to a
// handler still cached from the old object. Empty options use the normal lookup.
func (m *Manager) LookupRuntimeHandlerForRestore(ctx context.Context, runtimeClassName *string, options map[string]string) (string, error) {
	if len(options) == 0 {
		return m.LookupRuntimeHandler(runtimeClassName)
	}
	if runtimeClassName == nil || *runtimeClassName == "" {
		return "", fmt.Errorf("spec.restoreFrom.options requires spec.runtimeClassName and a RuntimeClass restore option allowlist")
	}
	name := *runtimeClassName
	rc, err := m.client.NodeV1().RuntimeClasses().Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to read RuntimeClass %q for restore options: %w", name, err)
	}
	var allowed []string
	if rc.PodCheckpoint != nil {
		allowed = rc.PodCheckpoint.AllowedRestoreOptions
	}
	if err := checkpointutil.ValidateRuntimeOptions(options, allowed); err != nil {
		return "", fmt.Errorf("spec.restoreFrom.options for RuntimeClass %q: %w", name, err)
	}
	return rc.Handler, nil
}
