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

package v1

import (
	"fmt"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/pkg/api/v1"
)

// ReplicationControllerListerExpansion allows custom methods to be added to
// ReplicationControllerLister.
type ReplicationControllerListerExpansion interface {
	GetPodControllers(pod *v1.Pod) ([]*v1.ReplicationController, error)
}

// ReplicationControllerNamespaceListerExpansion allows custom methods to be added to
// ReplicationControllerNamespaeLister.
type ReplicationControllerNamespaceListerExpansion interface{}

// GetPodControllers returns a list of ReplicationControllers that potentially match a pod.
// Only the one specified in the Pod's ControllerRef will actually manage it.
// Returns an error only if no matching ReplicationControllers are found.
func (s *replicationControllerLister) GetPodControllers(pod *v1.Pod) ([]*v1.ReplicationController, error) {
	if len(pod.Labels) == 0 {
		return nil, fmt.Errorf("no controllers found for pod %v because it has no labels", pod.Name)
	}

	items, err := s.ReplicationControllers(pod.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var controllers []*v1.ReplicationController
	for i := range items {
		rc := items[i]
		selector := labels.Set(rc.Spec.Selector).AsSelectorPreValidated()

		// If an rc with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		controllers = append(controllers, rc)
	}

	if len(controllers) == 0 {
		return nil, fmt.Errorf("could not find controller for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return controllers, nil
}
