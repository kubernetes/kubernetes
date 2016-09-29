package predicates

import "k8s.io/kubernetes/pkg/labels"
import "k8s.io/kubernetes/pkg/api"

// findLabelsInPod gets as many constraints as possible from the pod itself.
func findLabelsInPod(labelsToKeep []string, nodeSelector labels.Set) map[string]string {
	aL := make(map[string]string)
	for _, l := range labelsToKeep {
		if nodeSelector.Has(l) {
			aL[l] = nodeSelector.Get(l)
		}
		// No need to flag missing labels: if the length of this map < labelsToKeep,
		// we know that we need to introspect nodes to come up with a complet set of
		// homogenous labelling constraints.
	}
	return aL
}

// addUnsetLabelsToMap backfills missing constraints with constraints we find in a map.
func addUnsetLabelsToMap(aL map[string]string, labelsToAdd []string, labelSet labels.Set) {
	for _, l := range labelsToAdd {
		// If the pod being scheduled has the label value specified, do not override it
		if _, exists := aL[l]; exists {
			continue
		}
		// Otherwise
		if labelSet.Has(l) {
			aL[l] = labelSet.Get(l)
		}
	}
}

// filterPodsByNamespace appends only the pods
func filterPodsByNamespace(pods []*api.Pod, ns string) []*api.Pod {
	filtered := []*api.Pod{}
	for _, nsPod := range pods {
		if nsPod.Namespace == ns {
			filtered = append(filtered, nsPod)
		}
	}
	return filtered
}

// Helper function used to define the final selector which we will use to decide wether this pod
// does, or does not, match this node.
func createSelectorFromLabels(aL map[string]string) labels.Selector {
	if len(aL) == 0 {
		return labels.Everything()
	}
	return labels.Set(aL).AsSelector()
}
