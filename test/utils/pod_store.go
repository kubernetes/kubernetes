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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// Convenient wrapper around cache.Store that returns list of v1.Pod instead of interface{}.
type PodStore struct {
	cache.Store
	stopCh    chan struct{}
	Reflector *cache.Reflector
}

func NewPodStore(c clientset.Interface, namespace string, label labels.Selector, field fields.Selector) *PodStore {
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.LabelSelector = label.String()
			options.FieldSelector = field.String()
			obj, err := c.Core().Pods(namespace).List(options)
			return runtime.Object(obj), err
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.LabelSelector = label.String()
			options.FieldSelector = field.String()
			return c.Core().Pods(namespace).Watch(options)
		},
	}
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	stopCh := make(chan struct{})
	reflector := cache.NewReflector(lw, &v1.Pod{}, store, 0)
	reflector.RunUntil(stopCh)
	return &PodStore{Store: store, stopCh: stopCh, Reflector: reflector}
}

func (s *PodStore) List() []*v1.Pod {
	objects := s.Store.List()
	pods := make([]*v1.Pod, 0)
	for _, o := range objects {
		pods = append(pods, o.(*v1.Pod))
	}
	return pods
}

func (s *PodStore) Stop() {
	close(s.stopCh)
}
