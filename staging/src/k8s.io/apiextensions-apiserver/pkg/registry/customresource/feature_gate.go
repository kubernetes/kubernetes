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

package customresource

import (
	"strings"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// fieldPathStateTracker is used to store state about
// what fields are considered "enabled" based on the
// feature gate that guards them. Note that validation
// takes care of the fact that each fieldPath should
// be guarded by at-most 1 feature gate.
//
// A fieldPath is considered as enabled iff all parent
// fieldPaths are enabled as well. A fieldPath that is
// not guarded by a feature gate is considered enabled
// and therefore we track only fieldPaths that do come
// under a fetaure gate.
//
// We can use this to judge if a feature gate should be
// be considered or not. A gate can be considered iff
// all fieldPaths that it guards are enabled.
type fieldPathStateTracker struct {
	fieldPathState map[string]bool
}

func constructFieldPathTracker(gates *apiextensionsv1.CustomResourceDefinitionFeatureGates) fieldPathStateTracker {
	fieldPathState := make(map[string]bool)

	for _, gate := range gates.FeatureGates {
		enabled := isFeatureGateEnabled(gate)
		for _, fieldPath := range gate.FieldPaths {
			fieldPathState[fieldPath] = enabled
		}
	}

	for path := range fieldPathState {
		pathEnabled := true
		for potentialParent := range fieldPathState {
			if isParent(potentialParent, path) {
				if !fieldPathState[potentialParent] {
					pathEnabled = false
					break
				}
			}
		}
		fieldPathState[path] = pathEnabled
	}

	return fieldPathStateTracker{fieldPathState: fieldPathState}
}

func (t fieldPathStateTracker) isPathEnabled(path string) bool {
	return t.fieldPathState[path]
}

func (t fieldPathStateTracker) shouldConsiderFeatureGate(gate apiextensionsv1.CustomResourceDefinitionFeatureGate) bool {
	for _, path := range gate.FieldPaths {
		if !t.isPathEnabled(path) {
			return false
		}
	}

	return true
}

// TODO(MadhavJivrajani): use the utils repo for this, keeping strings.Contains
// for now till functionality is included in the utils repo.
func isParent(fieldPathA, fieldPathB string) bool {
	return len(fieldPathB) < len(fieldPathA) && strings.Contains(fieldPathA, fieldPathB)
}

// isFeatureGateEnabled determines if a feature gate is enabled or not according to
// the following set of rules:
// 1. If the preRelease of the feature gate is stable, then it is enabled.
// 2. If the feature gate has an enabled value specified, then this will
//    determine whether the feature gate is enabled or not.
// 3. If the feature gate does not have an enabled value specified but has
//    a default value specified, then this will determine whether the gate
//    is enabled or not.
// 4. If neither enabled nor default values are specified, the enabled state
//    of the feature gate takes after the default values of alpha and beta
//    preRelease stages, which is of state disabled.
func isFeatureGateEnabled(featureGate apiextensionsv1.CustomResourceDefinitionFeatureGate) bool {
	enabled := false
	switch {
	case featureGate.PreRelease == "stable": // TODO(MadhavJivrajani): create consts for these stages
		enabled = true
	case featureGate.Enabled != nil:
		enabled = *featureGate.Enabled
	case featureGate.Default != nil:
		enabled = *featureGate.Default
	}

	return enabled
}

func hasField(obj map[string]interface{}, fieldPath string) bool {
	fieldNames := getFieldNamesFromFieldPath(fieldPath)
	_, fieldExists, _ := unstructured.NestedFieldNoCopy(obj, fieldNames...)
	return fieldExists
}

func attemptDroppingField(obj map[string]interface{}, fieldPath string) {
	fieldNames := getFieldNamesFromFieldPath(fieldPath)
	_, fieldExists, _ := unstructured.NestedFieldNoCopy(obj, fieldNames...)
	if fieldExists {
		// To drop the field, we get a reference to the parent map of the field
		// and delete it from there. This is guaranteed to exist since at this
		// point, the entirety of the fieldPath exists.
		namesUptoLastField := fieldNames[:len(fieldNames)-1]
		parentOfField, _, _ := unstructured.NestedFieldNoCopy(obj, namesUptoLastField...)
		parentOfFieldMap := parentOfField.(map[string]interface{})
		delete(parentOfFieldMap, fieldNames[len(fieldNames)-1])
	}
}

func ensureNoUpdateToField(old, new map[string]interface{}, fieldPath string) {
	fieldNames := getFieldNamesFromFieldPath(fieldPath)
	copyOfField, existOld, _ := unstructured.NestedFieldCopy(old, fieldNames...)
	if !existOld {
		return
	}
	namesUptoLastField := fieldNames[:len(fieldNames)-1]
	lastField := fieldNames[len(fieldNames)-1]
	refToParentOfField, existsNew, _ := unstructured.NestedFieldNoCopy(new, namesUptoLastField...)
	if !existsNew {
		return
	}
	parentOfFieldMap := refToParentOfField.(map[string]interface{})
	parentOfFieldMap[lastField] = copyOfField
}

func getFieldNamesFromFieldPath(fieldPath string) []string {
	trimmedPath := strings.TrimPrefix(fieldPath, ".")
	return strings.Split(trimmedPath, ".")
}
