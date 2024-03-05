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

package v1

import (
	"context"
	"errors"
	"fmt"
	time "time"

	corev1 "k8s.io/api/core/v1"
	internalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/apiserver/pkg/registry/rest"
	v1 "k8s.io/client-go/listers/core/v1"
	cache "k8s.io/client-go/tools/cache"
	core "k8s.io/kubernetes/pkg/apis/core"
	apiscorev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	internalinterfaces "k8s.io/kubernetes/pkg/controlplane/storageinformers/internalinterfaces"
)

// PodInformer provides access to a shared informer and lister for
// Pods.
type PodInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() v1.PodLister
}

type podInformer struct {
	factory          internalinterfaces.SharedInformerFactory
	tweakListOptions internalinterfaces.TweakListOptionsFunc
	namespace        string
}

// NewPodInformer constructs a new informer for Pod type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewPodInformer(gvrStorage map[schema.GroupVersionResource]rest.Storage, namespace string, resyncPeriod time.Duration, indexers cache.Indexers) cache.SharedIndexInformer {
	return NewFilteredPodInformer(gvrStorage, namespace, resyncPeriod, indexers, nil)
}

// NewFilteredPodInformer constructs a new informer for Pod type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewFilteredPodInformer(gvrStorage map[schema.GroupVersionResource]rest.Storage, namespace string, resyncPeriod time.Duration, indexers cache.Indexers, tweakListOptions internalinterfaces.TweakListOptionsFunc) cache.SharedIndexInformer {
	return cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}

				gvr := schema.GroupVersionResource{"", "v1", "pods"}
				s, ok := gvrStorage[gvr]
				if !ok {
					return nil, errors.New(fmt.Sprintf("GVR %+v is not provided in storage", gvr))
				}
				pred := &internalversion.ListOptions{}
				if err := internalversion.Convert_v1_ListOptions_To_internalversion_ListOptions(&options, pred, nil); err != nil {
					return nil, err
				}
				lister := s.(rest.Lister)
				l, err := lister.List(context.TODO(), pred)
				if err != nil {
					return nil, err
				}

				return l, nil

			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}

				gvr := schema.GroupVersionResource{"", "v1", "pods"}
				s, ok := gvrStorage[gvr]
				if !ok {
					return nil, errors.New(fmt.Sprintf("GVR %+v is not provided in storage", gvr))
				}
				pred := &internalversion.ListOptions{}
				if err := internalversion.Convert_v1_ListOptions_To_internalversion_ListOptions(&options, pred, nil); err != nil {
					return nil, err
				}
				watcher := s.(rest.Watcher)
				w, err := watcher.Watch(context.TODO(), pred)
				if err != nil {
					return nil, err
				}
				return w, nil

			},
		},
		&corev1.Pod{},
		resyncPeriod,
		indexers,
	)
}

func (f *podInformer) defaultInformer(gvrStorage map[schema.GroupVersionResource]rest.Storage, resyncPeriod time.Duration) cache.SharedIndexInformer {

	i := NewFilteredPodInformer(gvrStorage, f.namespace, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, f.tweakListOptions)
	i.SetTransform(func(obj interface{}) (interface{}, error) {
		t := &corev1.Pod{}
		if err := apiscorev1.Convert_core_Pod_To_v1_Pod(obj.(*core.Pod), t, nil); err != nil {
			return nil, err
		}
		return t, nil
	})
	return i

}

func (f *podInformer) Informer() cache.SharedIndexInformer {
	return f.factory.InformerFor(&corev1.Pod{}, f.defaultInformer)
}

func (f *podInformer) Lister() v1.PodLister {
	return v1.NewPodLister(f.Informer().GetIndexer())
}
