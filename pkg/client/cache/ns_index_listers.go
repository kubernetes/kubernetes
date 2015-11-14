/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

const (
	NamespaceIndex string = "namespace"
)

// PodListerByNamespace makes a Store have the List method of the client.PodInterface
// The Store must contain (only) Pods, and just expects "namespace" index.
type PodListerByNamespace struct {
	Indexer
}

// List lists all pods in the store.
func (s PodListerByNamespace) List() (pods []*api.Pod, err error) {
	for _, c := range s.Indexer.List() {
		pods = append(pods, (c.(*api.Pod)))
	}
	return pods, nil
}

// Exists returns true if a pod matching the namespace/name of the given pod exists in the indexer.
func (s PodListerByNamespace) Exists(pod *api.Pod) (bool, error) {
	_, exists, err := s.Indexer.Get(pod)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// Pods takes a namespace, and returns a podsNamespacer, which has methods to work with Pod resources in a namespace.
func (s *PodListerByNamespace) Pods(namespace string) podsNamespacer {
	return podsNamespacer{s.Indexer, namespace}
}

// podsNamespacer has methods to work with Pod resources in a namespace.
type podsNamespacer struct {
	indexer   Indexer
	namespace string
}

// // List returns a list of pods that match the selectors in a namespace
func (s podsNamespacer) List(selector labels.Selector) (pods api.PodList, err error) {
	list := api.PodList{}
	if s.namespace == api.NamespaceAll {
		for _, m := range s.indexer.List() {
			pod := m.(*api.Pod)
			if selector.Matches(labels.Set(pod.Labels)) {
				list.Items = append(list.Items, *pod)
			}
		}
		return list, nil
	}

	key := &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: s.namespace}}
	items, err := s.indexer.Index(NamespaceIndex, key)
	if err != nil {
		return api.PodList{}, err
	}
	for _, m := range items {
		pod := m.(*api.Pod)
		if selector.Matches(labels.Set(pod.Labels)) {
			list.Items = append(list.Items, *pod)
		}
	}
	return list, nil
}

// ReplicationControllerListerByNamespace gives a store List and Exists methods.
// The store must contain only ReplicationControllers, and just expects "namespace" index.
type ReplicationControllerListerByNamespace struct {
	Indexer
}

// ReplicationControllerListerByNamespace lists all controllers in the store.
func (s ReplicationControllerListerByNamespace) List() (controllers []*api.ReplicationController, err error) {
	for _, c := range s.Indexer.List() {
		controllers = append(controllers, (c.(*api.ReplicationController)))
	}
	return controllers, nil
}

// Exists checks if the given rc exists in the store.
func (s ReplicationControllerListerByNamespace) Exists(controller *api.ReplicationController) (bool, error) {
	_, exists, err := s.Indexer.Get(controller)
	if err != nil {
		return false, err
	}
	return exists, nil
}

// GetPodControllers returns a list of replication controllers managing a pod. Returns an error only if no matching controllers are found.
func (s ReplicationControllerListerByNamespace) GetPodControllers(pod *api.Pod) (controllers []api.ReplicationController, err error) {
	var selector labels.Selector
	var rc api.ReplicationController

	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no controllers found for pod %v because it has no labels", pod.Name)
		return
	}

	key := &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace}}
	items, err := s.Indexer.Index(NamespaceIndex, key)
	if err != nil {
		return []api.ReplicationController{}, err
	}

	for _, m := range items {
		rc = *m.(*api.ReplicationController)
		selector = labels.Set(rc.Spec.Selector).AsSelector()
		// If an rc with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		controllers = append(controllers, rc)
	}
	if len(controllers) == 0 {
		err = fmt.Errorf("could not find controller for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}
