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

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// DaemonSetListerExpansion allows custom methods to be added to
// DaemonSetLister.
type DaemonSetListerExpansion interface {
	GetPodDaemonSets(pod *v1.Pod) ([]*apps.DaemonSet, error)
	GetHistoryDaemonSets(history *apps.ControllerRevision) ([]*apps.DaemonSet, error)
}

// DaemonSetNamespaceListerExpansion allows custom methods to be added to
// DaemonSetNamespaceLister.
type DaemonSetNamespaceListerExpansion interface{}

// GetPodDaemonSets returns a list of DaemonSets that potentially match a pod.
// Only the one specified in the Pod's ControllerRef will actually manage it.
// Returns an error only if no matching DaemonSets are found.
func (s *daemonSetLister) GetPodDaemonSets(pod *v1.Pod) ([]*apps.DaemonSet, error) {
	var selector labels.Selector
	var daemonSet *apps.DaemonSet

	if len(pod.Labels) == 0 {
		return nil, fmt.Errorf("no daemon sets found for pod %v because it has no labels", pod.Name)
	}

	list, err := s.DaemonSets(pod.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var daemonSets []*apps.DaemonSet
	for i := range list {
		daemonSet = list[i]
		if daemonSet.Namespace != pod.Namespace {
			continue
		}
		selector, err = metav1.LabelSelectorAsSelector(daemonSet.Spec.Selector)
		if err != nil {
			// this should not happen if the DaemonSet passed validation
			return nil, err
		}

		// If a daemonSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		daemonSets = append(daemonSets, daemonSet)
	}

	if len(daemonSets) == 0 {
		return nil, fmt.Errorf("could not find daemon set for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return daemonSets, nil
}

// GetHistoryDaemonSets returns a list of DaemonSets that potentially
// match a ControllerRevision. Only the one specified in the ControllerRevision's ControllerRef
// will actually manage it.
// Returns an error only if no matching DaemonSets are found.
func (s *daemonSetLister) GetHistoryDaemonSets(history *apps.ControllerRevision) ([]*apps.DaemonSet, error) {
	if len(history.Labels) == 0 {
		return nil, fmt.Errorf("no DaemonSet found for ControllerRevision %s because it has no labels", history.Name)
	}

	list, err := s.DaemonSets(history.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var daemonSets []*apps.DaemonSet
	for _, ds := range list {
		selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		// If a DaemonSet with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(history.Labels)) {
			continue
		}
		daemonSets = append(daemonSets, ds)
	}

	if len(daemonSets) == 0 {
		return nil, fmt.Errorf("could not find DaemonSets for ControllerRevision %s in namespace %s with labels: %v", history.Name, history.Namespace, history.Labels)
	}

	return daemonSets, nil
}
