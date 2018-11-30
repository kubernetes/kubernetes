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
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
)

var (
	runtimeClassGVR = schema.GroupVersionResource{
		Group:    "node.k8s.io",
		Version:  "v1alpha1",
		Resource: "runtimeclasses",
	}
)

// Manager caches RuntimeClass API objects, and provides accessors to the Kubelet.
type Manager struct {
	informer cache.SharedInformer
}

// NewManager returns a new RuntimeClass Manager. Run must be called before the manager can be used.
func NewManager(client dynamic.Interface) *Manager {
	rc := client.Resource(runtimeClassGVR)
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return rc.List(options)
		},
		WatchFunc: rc.Watch,
	}
	informer := cache.NewSharedInformer(lw, &unstructured.Unstructured{}, 0)

	return &Manager{
		informer: informer,
	}
}

// Run starts syncing the RuntimeClass cache with the apiserver.
func (m *Manager) Run(stopCh <-chan struct{}) {
	m.informer.Run(stopCh)
}

// LookupRuntimeHandler returns the RuntimeHandler string associated with the given RuntimeClass
// name (or the default of "" for nil). If the RuntimeClass is not found, it returns an
// apierrors.NotFound error.
func (m *Manager) LookupRuntimeHandler(runtimeClassName *string) (string, error) {
	if runtimeClassName == nil || *runtimeClassName == "" {
		// The default RuntimeClass always resolves to the empty runtime handler.
		return "", nil
	}

	name := *runtimeClassName
	item, exists, err := m.informer.GetStore().GetByKey(name)
	if err != nil {
		return "", fmt.Errorf("Failed to lookup RuntimeClass %s: %v", name, err)
	}
	if !exists {
		return "", errors.NewNotFound(schema.GroupResource{
			Group:    runtimeClassGVR.Group,
			Resource: runtimeClassGVR.Resource,
		}, name)
	}

	rc, ok := item.(*unstructured.Unstructured)
	if !ok {
		return "", fmt.Errorf("unexpected RuntimeClass type %T", item)
	}

	handler, _, err := unstructured.NestedString(rc.Object, "spec", "runtimeHandler")
	if err != nil {
		return "", fmt.Errorf("Invalid RuntimeClass object: %v", err)
	}

	return handler, nil
}
