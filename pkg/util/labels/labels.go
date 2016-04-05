/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"reflect"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	pkg_label "k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
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

// TODO: Find a better home for this
// MergeInto flags
const (
	OverwriteExistingDstKey = 1 << iota
	ErrorOnExistingDstKey
	ErrorOnDifferentDstKeyValue
)

// AddObjectLabels adds new label(s) to a single runtime.Object
func AddObjectLabels(obj runtime.Object, labels pkg_label.Set) error {
	if labels == nil {
		return nil
	}

	accessor, err := meta.Accessor(obj)

	if err != nil {
		if _, ok := obj.(*runtime.Unstructured); !ok {
			// error out if it's not possible to get an accessor and it's also not an unstructured object
			return err
		}
	} else {
		metaLabels := accessor.GetLabels()
		if metaLabels == nil {
			metaLabels = make(map[string]string)
		}

		if err := MergeInto(metaLabels, labels, ErrorOnDifferentDstKeyValue); err != nil {
			return fmt.Errorf("unable to add labels to %s/%s: %v", obj.GetObjectKind().GroupVersionKind(), accessor.GetName(), err)
		}
		accessor.SetLabels(metaLabels)

		return nil
	}

	// handle unstructured object
	// TODO: allow meta.Accessor to handle runtime.Unstructured
	if unstruct, ok := obj.(*runtime.Unstructured); ok && unstruct.Object != nil {
		// the presence of "metadata" is sufficient for us to apply the rules for Kube-like
		// objects.
		// TODO: add swagger detection to allow this to happen more effectively
		if obj, ok := unstruct.Object["metadata"]; ok {
			if m, ok := obj.(map[string]interface{}); ok {

				existing := make(map[string]string)
				if l, ok := m["labels"]; ok {
					if found, ok := interfaceToStringMap(l); ok {
						existing = found
					}
				}
				if err := MergeInto(existing, labels, OverwriteExistingDstKey); err != nil {
					return err
				}
				m["labels"] = mapToGeneric(existing)
			}
			return nil
		}

		// only attempt to set root labels if a root object called labels exists
		// TODO: add swagger detection to allow this to happen more effectively
		if obj, ok := unstruct.Object["labels"]; ok {
			existing := make(map[string]string)
			if found, ok := interfaceToStringMap(obj); ok {
				existing = found
			}
			if err := MergeInto(existing, labels, OverwriteExistingDstKey); err != nil {
				return err
			}
			unstruct.Object["labels"] = mapToGeneric(existing)
			return nil
		}
	}

	return nil
}

// interfaceToStringMap extracts a map[string]string from a map[string]interface{}
func interfaceToStringMap(obj interface{}) (map[string]string, bool) {
	if obj == nil {
		return nil, false
	}
	lm, ok := obj.(map[string]interface{})
	if !ok {
		return nil, false
	}
	existing := make(map[string]string)
	for k, v := range lm {
		switch t := v.(type) {
		case string:
			existing[k] = t
		}
	}
	return existing, true
}

// mapToGeneric converts a map[string]string into a map[string]interface{}
func mapToGeneric(obj map[string]string) map[string]interface{} {
	if obj == nil {
		return nil
	}
	res := make(map[string]interface{})
	for k, v := range obj {
		res[k] = v
	}
	return res
}

// MergeInto merges items from a src map into a dst map.
// Returns an error when the maps are not of the same type.
// Flags:
// - ErrorOnExistingDstKey
//     When set: Return an error if any of the dst keys is already set.
// - ErrorOnDifferentDstKeyValue
//     When set: Return an error if any of the dst keys is already set
//               to a different value than src key.
// - OverwriteDstKey
//     When set: Overwrite existing dst key value with src key value.
func MergeInto(dst, src interface{}, flags int) error {
	dstVal := reflect.ValueOf(dst)
	srcVal := reflect.ValueOf(src)

	if dstVal.Kind() != reflect.Map {
		return fmt.Errorf("dst is not a valid map: %v", dstVal.Kind())
	}
	if srcVal.Kind() != reflect.Map {
		return fmt.Errorf("src is not a valid map: %v", srcVal.Kind())
	}
	if dstTyp, srcTyp := dstVal.Type(), srcVal.Type(); !dstTyp.AssignableTo(srcTyp) {
		return fmt.Errorf("type mismatch, can't assign '%v' to '%v'", srcTyp, dstTyp)
	}

	if dstVal.IsNil() {
		return fmt.Errorf("dst value is nil")
	}
	if srcVal.IsNil() {
		// Nothing to merge
		return nil
	}

	for _, k := range srcVal.MapKeys() {
		if dstVal.MapIndex(k).IsValid() {
			if flags&ErrorOnExistingDstKey != 0 {
				return fmt.Errorf("dst key already set (ErrorOnExistingDstKey=1), '%v'='%v'", k, dstVal.MapIndex(k))
			}
			if dstVal.MapIndex(k).String() != srcVal.MapIndex(k).String() {
				if flags&ErrorOnDifferentDstKeyValue != 0 {
					return fmt.Errorf("dst key already set to a different value (ErrorOnDifferentDstKeyValue=1), '%v'='%v'", k, dstVal.MapIndex(k))
				}
				if flags&OverwriteExistingDstKey != 0 {
					dstVal.SetMapIndex(k, srcVal.MapIndex(k))
				}
			}
		} else {
			dstVal.SetMapIndex(k, srcVal.MapIndex(k))
		}
	}

	return nil
}
