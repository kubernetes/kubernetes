/*
Copyright 2016 The Kubernetes Authors.

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

package labels

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// Clones the given map and returns a new map with the given key and value added.
// Returns the given map, if labelKey is empty.
func CloneAndAddLabel(labels map[string]string, labelKey string, labelValue uint32) map[string]string {
	if labelKey == "" {
		// Don't need to add a label.
		return labels
	}
	// Clone.
	newLabels := map[string]string{}
	for key, value := range labels {
		newLabels[key] = value
	}
	newLabels[labelKey] = fmt.Sprintf("%d", labelValue)
	return newLabels
}

// CloneAndRemoveLabel clones the given map and returns a new map with the given key removed.
// Returns the given map, if labelKey is empty.
func CloneAndRemoveLabel(labels map[string]string, labelKey string) map[string]string {
	if labelKey == "" {
		// Don't need to add a label.
		return labels
	}
	// Clone.
	newLabels := map[string]string{}
	for key, value := range labels {
		newLabels[key] = value
	}
	delete(newLabels, labelKey)
	return newLabels
}

// AddLabel returns a map with the given key and value added to the given map.
func AddLabel(labels map[string]string, labelKey string, labelValue string) map[string]string {
	if labelKey == "" {
		// Don't need to add a label.
		return labels
	}
	if labels == nil {
		labels = make(map[string]string)
	}
	labels[labelKey] = labelValue
	return labels
}

// Clones the given selector and returns a new selector with the given key and value added.
// Returns the given selector, if labelKey is empty.
func CloneSelectorAndAddLabel(selector *unversioned.LabelSelector, labelKey string, labelValue uint32) *unversioned.LabelSelector {
	if labelKey == "" {
		// Don't need to add a label.
		return selector
	}

	// Clone.
	newSelector := new(unversioned.LabelSelector)

	// TODO(madhusudancs): Check if you can use deepCopy_extensions_LabelSelector here.
	newSelector.MatchLabels = make(map[string]string)
	if selector.MatchLabels != nil {
		for key, val := range selector.MatchLabels {
			newSelector.MatchLabels[key] = val
		}
	}
	newSelector.MatchLabels[labelKey] = fmt.Sprintf("%d", labelValue)

	if selector.MatchExpressions != nil {
		newMExps := make([]unversioned.LabelSelectorRequirement, len(selector.MatchExpressions))
		for i, me := range selector.MatchExpressions {
			newMExps[i].Key = me.Key
			newMExps[i].Operator = me.Operator
			if me.Values != nil {
				newMExps[i].Values = make([]string, len(me.Values))
				copy(newMExps[i].Values, me.Values)
			} else {
				newMExps[i].Values = nil
			}
		}
		newSelector.MatchExpressions = newMExps
	} else {
		newSelector.MatchExpressions = nil
	}

	return newSelector
}

// AddLabelToSelector returns a selector with the given key and value added to the given selector's MatchLabels.
func AddLabelToSelector(selector *unversioned.LabelSelector, labelKey string, labelValue string) *unversioned.LabelSelector {
	if labelKey == "" {
		// Don't need to add a label.
		return selector
	}
	if selector.MatchLabels == nil {
		selector.MatchLabels = make(map[string]string)
	}
	selector.MatchLabels[labelKey] = labelValue
	return selector
}

// SelectorHasLabel checks if the given selector contains the given label key in its MatchLabels
func SelectorHasLabel(selector *unversioned.LabelSelector, labelKey string) bool {
	return len(selector.MatchLabels[labelKey]) > 0
}
