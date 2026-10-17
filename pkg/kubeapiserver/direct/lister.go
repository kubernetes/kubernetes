/*
Copyright The Kubernetes Authors.

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

package direct

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
)

// Lister is a generic lister backed by a Client (mirroring client-go's listers.ResourceIndexer[T]).
// It reads directly from the watch cache (RV=0) instead of from an informer's local store.
type Lister[T runtime.Object] struct {
	client    Client
	namespace string
}

var _ cache.GenericLister = Lister[runtime.Object]{}
var _ cache.GenericNamespaceLister = Lister[runtime.Object]{}

// NewLister returns a new Lister for type T backed by client.
func NewLister[T runtime.Object](client Client) Lister[T] {
	return Lister[T]{client: client}
}

// Namespaced returns a Lister scoped to the specified namespace.
func (l Lister[T]) Namespaced(namespace string) Lister[T] {
	return Lister[T]{client: l.client, namespace: namespace}
}

// ByNamespace implements cache.GenericLister.
func (l Lister[T]) ByNamespace(namespace string) cache.GenericNamespaceLister {
	return Lister[runtime.Object]{client: l.client, namespace: namespace}
}

// List lists all resources matching selector from the watch cache (RV=0).
func (l Lister[T]) List(selector labels.Selector) ([]T, error) {
	return l.ListWithContext(context.Background(), selector)
}

// ListWithContext lists all resources matching selector from the watch cache (RV=0) using ctx.
func (l Lister[T]) ListWithContext(ctx context.Context, selector labels.Selector) ([]T, error) {
	listOptions := metav1.ListOptions{
		ResourceVersion: "0",
	}
	if selector != nil && !selector.Empty() {
		listOptions.LabelSelector = selector.String()
	}
	obj, err := l.client.List(ctx, l.namespace, listOptions)
	if err != nil {
		return nil, err
	}
	var ret []T
	err = meta.EachListItem(obj, func(item runtime.Object) error {
		typed, ok := item.(T)
		if !ok {
			return fmt.Errorf("expected %T in list from direct client, got %T", *new(T), item)
		}
		ret = append(ret, typed)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// Get retrieves the resource with the given name from the watch cache (RV=0).
func (l Lister[T]) Get(name string) (T, error) {
	obj, err := l.client.Get(context.Background(), l.namespace, name, metav1.GetOptions{ResourceVersion: "0"})
	if err != nil {
		return *new(T), err
	}
	typed, ok := obj.(T)
	if !ok {
		return *new(T), fmt.Errorf("expected %T from direct client, got %T", *new(T), obj)
	}
	return typed, nil
}

type podLister struct {
	Lister[*corev1.Pod]
}

var _ corev1listers.PodLister = &podLister{}
var _ podsecurityadmission.PodLister = &podLister{}

// NewPodLister returns a PodLister backed by a direct Client (mirroring corev1listers.NewPodLister).
func NewPodLister(client Client) corev1listers.PodLister {
	return &podLister{
		Lister: NewLister[*corev1.Pod](client),
	}
}

// Pods returns a PodNamespaceLister for the given namespace.
func (l *podLister) Pods(namespace string) corev1listers.PodNamespaceLister {
	return l.Namespaced(namespace)
}

// ListPods implements podsecurityadmission.PodLister.
func (l *podLister) ListPods(ctx context.Context, namespace string) ([]*corev1.Pod, error) {
	return l.Namespaced(namespace).ListWithContext(ctx, labels.Everything())
}
