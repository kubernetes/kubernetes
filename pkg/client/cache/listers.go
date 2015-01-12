/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
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
	Store
}

// TODO Get rid of the selector because that is confusing because the user might not realize that there has already been
// some selection at the caching stage.  Also, consistency will facilitate code generation.  However, the pkg/client
// is inconsistent too.
func (s *StoreToPodLister) List(selector labels.Selector) (pods []api.Pod, err error) {
	for _, m := range s.Store.List() {
		pod := m.(*api.Pod)
		if selector.Matches(labels.Set(pod.Labels)) {
			pods = append(pods, *pod)
		}
	}
	return pods, nil
}

// StoreToNodeLister makes a Store have the List method of the client.NodeInterface
// The Store must contain (only) Nodes.
type StoreToNodeLister struct {
	Store
}

func (s *StoreToNodeLister) List() (machines api.NodeList, err error) {
	for _, m := range s.Store.List() {
		machines.Items = append(machines.Items, *(m.(*api.Node)))
	}
	return machines, nil
}

// TODO Move this back to scheduler as a helper function that takes a Store,
// rather than a method of StoreToNodeLister.
// GetNodeInfo returns cached data for the minion 'id'.
func (s *StoreToNodeLister) GetNodeInfo(id string) (*api.Node, error) {
	if minion, ok := s.Get(id); ok {
		return minion.(*api.Node), nil
	}
	return nil, fmt.Errorf("minion '%v' is not in cache", id)
}

// StoreToServiceLister makes a Store have the List method of the client.ServiceInterface
// The Store must contain (only) Services.
type StoreToServiceLister struct {
	Store
}

func (s *StoreToServiceLister) List() (svcs api.ServiceList, err error) {
	for _, m := range s.Store.List() {
		svcs.Items = append(svcs.Items, *(m.(*api.Service)))
	}
	return svcs, nil
}

// TODO: add StoreToEndpointsLister for use in kube-proxy.
