/*
Copyright 2017 The Kubernetes Authors.

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

package internalversion

import (
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api"
)

// NodeConditionPredicate is a function that indicates whether the given node's conditions meet
// some set of criteria defined by the function.
type NodeConditionPredicate func(node *api.Node) bool

// NodeListerExpansion allows custom methods to be added to
// NodeLister.
type NodeListerExpansion interface {
	ListWithPredicate(predicate NodeConditionPredicate) ([]*api.Node, error)
}

func (l *nodeLister) ListWithPredicate(predicate NodeConditionPredicate) ([]*api.Node, error) {
	nodes, err := l.List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var filtered []*api.Node
	for i := range nodes {
		if predicate(nodes[i]) {
			filtered = append(filtered, nodes[i])
		}
	}

	return filtered, nil
}
