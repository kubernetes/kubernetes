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

package cache

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/labels"
)

//  TODO: generate these classes and methods for all resources of interest using
// a script.  Can use "go generate" once 1.4 is supported by all users.

// StoreToPodLister makes a Store have the List method of the client.PodInterface
// The Store must contain (only) Pods.
//
// Example:
// s := cache.NewStore()
// lw := cache.ListWatch{Client: c, FieldSelector: sel, Resource: "pods"}
// r := cache.NewReflector(lw, &api.Pod{}, s).Run()
// l := StoreToPodLister{s}
// l.List()
type StoreToPodLister struct {
	Indexer Indexer
}

func (s *StoreToPodLister) List(selector labels.Selector) (pods []*api.Pod, err error) {
	err = ListAll(s.Indexer, selector, func(m interface{}) {
		pods = append(pods, m.(*api.Pod))
	})
	return pods, err
}

func (s *StoreToPodLister) Pods(namespace string) storePodsNamespacer {
	return storePodsNamespacer{Indexer: s.Indexer, namespace: namespace}
}

type storePodsNamespacer struct {
	Indexer   Indexer
	namespace string
}

func (s storePodsNamespacer) List(selector labels.Selector) (pods []*api.Pod, err error) {
	err = ListAllByNamespace(s.Indexer, s.namespace, selector, func(m interface{}) {
		pods = append(pods, m.(*api.Pod))
	})
	return pods, err
}

func (s storePodsNamespacer) Get(name string) (*api.Pod, error) {
	obj, exists, err := s.Indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(api.Resource("pod"), name)
	}
	return obj.(*api.Pod), nil
}
