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

package v1beta2

import (
	"fmt"

	apps "k8s.io/api/apps/v1beta2"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// ReplicaSetListerExpansion allows custom methods to be added to
// ReplicaSetLister.
type ReplicaSetListerExpansion interface {
	GetPodReplicaSets(pod *v1.Pod) ([]*apps.ReplicaSet, error)
}

// ReplicaSetNamespaceListerExpansion allows custom methods to be added to
// ReplicaSetNamespaceLister.
type ReplicaSetNamespaceListerExpansion interface{}

// GetPodReplicaSets returns a list of ReplicaSets that potentially match a pod.
// Only the one specified in the Pod's ControllerRef will actually manage it.
// Returns an error only if no matching ReplicaSets are found.
func (s *replicaSetLister) GetPodReplicaSets(pod *v1.Pod) ([]*apps.ReplicaSet, error) {
	if len(pod.Labels) == 0 {
		return nil, fmt.Errorf("no ReplicaSets found for pod %v because it has no labels", pod.Name)
	}

	list, err := s.ReplicaSets(pod.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var rss []*apps.ReplicaSet
	for _, rs := range list {
		if rs.Namespace != pod.Namespace {
			continue
		}
		selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
		if err != nil {
			// This object has an invalid selector, it does not match the pod
			continue
		}

		// If a ReplicaSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		rss = append(rss, rs)
	}

	if len(rss) == 0 {
		return nil, fmt.Errorf("could not find ReplicaSet for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return rss, nil
}
