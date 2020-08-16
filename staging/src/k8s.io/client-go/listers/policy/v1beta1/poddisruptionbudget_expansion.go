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

	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/klog/v2"
)

// PodDisruptionBudgetListerExpansion allows custom methods to be added to
// PodDisruptionBudgetLister.
type PodDisruptionBudgetListerExpansion interface {
	GetPodPodDisruptionBudgets(pod *v1.Pod) ([]*policy.PodDisruptionBudget, error)
}

// PodDisruptionBudgetNamespaceListerExpansion allows custom methods to be added to
// PodDisruptionBudgetNamespaceLister.
type PodDisruptionBudgetNamespaceListerExpansion interface{}

// GetPodPodDisruptionBudgets returns a list of PodDisruptionBudgets matching a pod.  Returns an error only if no matching PodDisruptionBudgets are found.
func (s *podDisruptionBudgetLister) GetPodPodDisruptionBudgets(pod *v1.Pod) ([]*policy.PodDisruptionBudget, error) {
	var selector labels.Selector

	if len(pod.Labels) == 0 {
		return nil, fmt.Errorf("no PodDisruptionBudgets found for pod %v because it has no labels", pod.Name)
	}

	list, err := s.PodDisruptionBudgets(pod.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	var pdbList []*policy.PodDisruptionBudget
	for i := range list {
		pdb := list[i]
		selector, err = metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
		if err != nil {
			klog.Warningf("invalid selector: %v", err)
			// TODO(mml): add an event to the PDB
			continue
		}

		// If a PDB with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		pdbList = append(pdbList, pdb)
	}

	if len(pdbList) == 0 {
		return nil, fmt.Errorf("could not find PodDisruptionBudget for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}

	return pdbList, nil
}
