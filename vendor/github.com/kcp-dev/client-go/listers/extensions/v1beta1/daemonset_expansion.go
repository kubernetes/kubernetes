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

package v1beta1

import (
	"fmt"

	apps "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// DaemonSetClusterListerExpansion allows custom methods to be added to
// DaemonSetClusterLister.
type DaemonSetClusterListerExpansion interface{}

// DaemonSetListerExpansion allows custom methods to be added to
// DaemonSetLister.
type DaemonSetListerExpansion interface {
	GetPodDaemonSets(pod *v1.Pod) ([]*v1beta1.DaemonSet, error)
	GetHistoryDaemonSets(history *apps.ControllerRevision) ([]*v1beta1.DaemonSet, error)
}

// DaemonSetNamespaceListerExpansion allows custom methods to be added to
// DaemonSetNamespaceLister.
type DaemonSetNamespaceListerExpansion interface{}

// GetPodDaemonSets returns a list of DaemonSets that potentially match a pod.
// Only the one specified in the Pod's ControllerRef will actually manage it.
// Returns an error only if no matching DaemonSets are found.
func (s *daemonSetLister) GetPodDaemonSets(pod *v1.Pod) ([]*v1beta1.DaemonSet, error) {
	var selector labels.Selector
	var daemonSet *v1beta1.DaemonSet

	if len(pod.Labels) == 0 {
		return nil, fmt.Errorf("no daemon sets found for pod %v because it has no labels", pod.Name)
	}

	list, err := s.DaemonSets(pod.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var daemonSets []*v1beta1.DaemonSet
	for i := range list {
		daemonSet = list[i]
		if daemonSet.Namespace != pod.Namespace {
			continue
		}
		selector, err = metav1.LabelSelectorAsSelector(daemonSet.Spec.Selector)
		if err != nil {
			// This object has an invalid selector, it does not match the pod
			continue
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
func (s *daemonSetLister) GetHistoryDaemonSets(history *apps.ControllerRevision) ([]*v1beta1.DaemonSet, error) {
	if len(history.Labels) == 0 {
		return nil, fmt.Errorf("no DaemonSet found for ControllerRevision %s because it has no labels", history.Name)
	}

	list, err := s.DaemonSets(history.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var daemonSets []*v1beta1.DaemonSet
	for _, ds := range list {
		selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
		if err != nil {
			// This object has an invalid selector, it does not match the history object
			continue
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
