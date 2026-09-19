/*
Copyright 2020 The Kubernetes Authors.

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

package volume

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/component-helpers/scheduling/corev1"
)

// PersistentVolumeClaimHasClass returns true if given claim has set StorageClassName field.
func PersistentVolumeClaimHasClass(claim *v1.PersistentVolumeClaim) bool {
	// Use beta annotation first
	if _, found := claim.Annotations[v1.BetaStorageClassAnnotation]; found {
		return true
	}

	if claim.Spec.StorageClassName != nil {
		return true
	}

	return false
}

// GetPersistentVolumeClaimClass returns StorageClassName. If no storage class was
// requested, it returns "".
func GetPersistentVolumeClaimClass(claim *v1.PersistentVolumeClaim) string {
	// Use beta annotation first
	if class, found := claim.Annotations[v1.BetaStorageClassAnnotation]; found {
		return class
	}

	if claim.Spec.StorageClassName != nil {
		return *claim.Spec.StorageClassName
	}

	return ""
}

// GetPersistentVolumeClass returns StorageClassName.
func GetPersistentVolumeClass(volume *v1.PersistentVolume) string {
	// Use beta annotation first
	if class, found := volume.Annotations[v1.BetaStorageClassAnnotation]; found {
		return class
	}

	return volume.Spec.StorageClassName
}

// ContainsAccessMode returns true if the requested access mode is present.
func ContainsAccessMode(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// MatchTopologySelectorTerms checks whether the given labels match any of the
// topology selector terms. An empty term list matches all objects, while an
// empty term matches no objects.
func MatchTopologySelectorTerms(topologySelectorTerms []v1.TopologySelectorTerm, lbls labels.Set) bool {
	if len(topologySelectorTerms) == 0 {
		return true
	}

	for _, term := range topologySelectorTerms {
		if len(term.MatchLabelExpressions) == 0 {
			continue
		}

		selector, err := topologySelectorRequirementsAsSelector(term.MatchLabelExpressions)
		if err == nil && selector.Matches(lbls) {
			return true
		}
	}

	return false
}

func topologySelectorRequirementsAsSelector(requirements []v1.TopologySelectorLabelRequirement) (labels.Selector, error) {
	if len(requirements) == 0 {
		return labels.Nothing(), nil
	}

	selector := labels.NewSelector()
	for _, expression := range requirements {
		requirement, err := labels.NewRequirement(expression.Key, selection.In, expression.Values)
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*requirement)
	}

	return selector, nil
}

// CheckNodeAffinity looks at the PV node affinity, and checks if the node has the same corresponding labels
// This ensures that we don't mount a volume that doesn't belong to this node
func CheckNodeAffinity(pv *v1.PersistentVolume, nodeLabels map[string]string) error {
	if pv.Spec.NodeAffinity == nil {
		return nil
	}

	if pv.Spec.NodeAffinity.Required != nil {
		node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: nodeLabels}}
		terms := pv.Spec.NodeAffinity.Required
		if matches, err := corev1.MatchNodeSelectorTerms(node, terms); err != nil {
			return err
		} else if !matches {
			return fmt.Errorf("no matching NodeSelectorTerms")
		}
	}

	return nil
}
