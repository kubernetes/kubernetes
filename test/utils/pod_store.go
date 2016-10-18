/*
Copyright 2016 The Kubernetes Authors.

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

package utils

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// Convenient wrapper around cache.Store that returns list of api.Pod instead of interface{}.
type PodStore struct {
	cache.Store
	stopCh    chan struct{}
	Reflector *cache.Reflector
}

func NewPodStore(c clientset.Interface, namespace string, label labels.Selector, field fields.Selector) *PodStore {
	lw := &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			options.LabelSelector = label
			options.FieldSelector = field
			obj, err := c.Core().Pods(namespace).List(options)
			return runtime.Object(obj), err
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			options.LabelSelector = label
			options.FieldSelector = field
			return c.Core().Pods(namespace).Watch(options)
		},
	}
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	stopCh := make(chan struct{})
	reflector := cache.NewReflector(lw, &api.Pod{}, store, 0)
	reflector.RunUntil(stopCh)
	return &PodStore{Store: store, stopCh: stopCh, Reflector: reflector}
}

func (s *PodStore) List() []*api.Pod {
	objects := s.Store.List()
	pods := make([]*api.Pod, 0)
	for _, o := range objects {
		pods = append(pods, o.(*api.Pod))
	}
	return pods
}

func (s *PodStore) Stop() {
	close(s.stopCh)
}
