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

// package taints implements utilities for working with taints
package taints

import (
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
)

const (
	MODIFIED  = "modified"
	TAINTED   = "tainted"
	UNTAINTED = "untainted"
)

// parseTaint parses a taint from a string, whose form must be either
// '<key>=<value>:<effect>', '<key>:<effect>', or '<key>'.
func parseTaint(st string) (v1.Taint, error) {
	var taint v1.Taint

	// Check for multiple ':' separators
	colonCount := strings.Count(st, ":")
	if colonCount > 1 {
		return taint, fmt.Errorf("invalid taint spec: %v", st)
	}

	// Split by ':' to separate key-value from effect
	var keyValue, effect string
	if colonCount == 1 {
		parts := strings.SplitN(st, ":", 2)
		keyValue = parts[0]
		effect = parts[1]
	} else {
		// No colon, just key
		keyValue = st
	}

	// Parse key and value
	var key, value string
	equalCount := strings.Count(keyValue, "=")
	if equalCount > 1 {
		return taint, fmt.Errorf("invalid taint spec: %v", st)
	}

	if equalCount == 1 {
		parts := strings.SplitN(keyValue, "=", 2)
		key = parts[0]
		value = parts[1]
	} else {
		key = keyValue
	}

	// Validate key
	if errs := validation.IsQualifiedName(key); len(errs) > 0 {
		return taint, fmt.Errorf("invalid taint key: %v", strings.Join(errs, "; "))
	}

	// Validate value if present
	if value != "" {
		if errs := validation.IsValidLabelValue(value); len(errs) > 0 {
			return taint, fmt.Errorf("invalid taint value: %v", strings.Join(errs, "; "))
		}
	}

	// Validate effect if present
	var taintEffect v1.TaintEffect
	if effect != "" {
		taintEffect = v1.TaintEffect(effect)
		if err := validateTaintEffect(taintEffect); err != nil {
			return taint, err
		}
	}

	taint.Key = key
	taint.Value = value
	taint.Effect = taintEffect

	return taint, nil
}

func validateTaintEffect(effect v1.TaintEffect) error {
	if effect != v1.TaintEffectNoSchedule &&
		effect != v1.TaintEffectPreferNoSchedule &&
		effect != v1.TaintEffectNoExecute {
		return fmt.Errorf("invalid taint effect: %v, unsupported taint effect", effect)
	}
	return nil
}

// ParseTaints takes a spec which is an array and creates slices for new taints to be added, taints to be deleted.
// It also validates the spec. For example, the form `<key>` may be used to remove a taint, but not to add one.
func ParseTaints(spec []string) ([]v1.Taint, []v1.Taint, error) {
	var taintsToAdd []v1.Taint
	var taintsToRemove []v1.Taint

	// Track taints to be added by key:effect to detect duplicates
	addedTaints := make(map[string]bool)

	for _, taintSpec := range spec {
		if len(taintSpec) == 0 {
			continue
		}

		// Check if it's a removal operation (ends with -)
		isRemove := strings.HasSuffix(taintSpec, "-")
		var taintStr string
		if isRemove {
			taintStr = taintSpec[:len(taintSpec)-1]
		} else {
			taintStr = taintSpec
		}

		// Parse the taint
		taint, err := parseTaint(taintStr)
		if err != nil {
			return nil, nil, err
		}

		if isRemove {
			// For removal, we only need key and effect
			taintsToRemove = append(taintsToRemove, v1.Taint{
				Key:    taint.Key,
				Effect: taint.Effect,
			})
		} else {
			// For add operation, effect is required (form `key` is not allowed)
			if taint.Effect == "" {
				return nil, nil, fmt.Errorf("invalid taint spec: %v", taintSpec)
			}

			// Check for duplicates (same key:effect)
			taintKey := fmt.Sprintf("%s:%s", taint.Key, taint.Effect)
			if addedTaints[taintKey] {
				return nil, nil, fmt.Errorf("duplicated taints with the same key and effect: %v", taint)
			}
			addedTaints[taintKey] = true

			taintsToAdd = append(taintsToAdd, taint)
		}
	}

	return taintsToAdd, taintsToRemove, nil
}

// CheckIfTaintsAlreadyExists checks if the node already has taints that we want to add and returns a string with taint keys.
func CheckIfTaintsAlreadyExists(oldTaints []v1.Taint, taints []v1.Taint) string {
	var existingKeys []string
	for _, taint := range taints {
		if TaintExists(oldTaints, &taint) {
			existingKeys = append(existingKeys, taint.Key)
		}
	}
	return strings.Join(existingKeys, ",")
}

// DeleteTaintsByKey removes all the taints that have the same key to given taintKey
func DeleteTaintsByKey(taints []v1.Taint, taintKey string) ([]v1.Taint, bool) {
	var newTaints []v1.Taint
	deleted := false
	for _, taint := range taints {
		if taint.Key != taintKey {
			newTaints = append(newTaints, taint)
		} else {
			deleted = true
		}
	}
	return newTaints, deleted
}

// DeleteTaint removes all the taints that have the same key and effect to given taintToDelete.
func DeleteTaint(taints []v1.Taint, taintToDelete *v1.Taint) ([]v1.Taint, bool) {
	var newTaints []v1.Taint
	deleted := false
	for _, taint := range taints {
		if taint.Key == taintToDelete.Key && taint.Effect == taintToDelete.Effect {
			deleted = true
			continue
		}
		newTaints = append(newTaints, taint)
	}
	return newTaints, deleted
}

// RemoveTaint tries to remove a taint from annotations list. Returns a new copy of updated Node and true if something was updated
// false otherwise.
func RemoveTaint(node *v1.Node, taint *v1.Taint) (*v1.Node, bool, error) {
	newNode := node.DeepCopy()
	nodeTaints := newNode.Spec.Taints
	if len(nodeTaints) == 0 {
		return newNode, false, nil
	}

	newTaints, deleted := DeleteTaint(nodeTaints, taint)
	if !deleted {
		return newNode, false, nil
	}

	newNode.Spec.Taints = newTaints
	return newNode, true, nil
}

// AddOrUpdateTaint tries to add a taint to annotations list. Returns a new copy of updated Node and true if something was updated
// false otherwise.
func AddOrUpdateTaint(node *v1.Node, taint *v1.Taint) (*v1.Node, bool, error) {
	newNode := node.DeepCopy()
	nodeTaints := newNode.Spec.Taints

	var newTaints []v1.Taint
	updated := false

	// Check if taint already exists
	for i := range nodeTaints {
		if nodeTaints[i].Key == taint.Key && nodeTaints[i].Effect == taint.Effect {
			// Found existing taint with same key and effect
			if nodeTaints[i].Value == taint.Value {
				// Identical taint, no update needed
				return newNode, false, nil
			}
			// Update the value
			updated = true
			newTaints = append(newTaints, *taint)
		} else {
			newTaints = append(newTaints, nodeTaints[i])
		}
	}

	if !updated {
		// Taint doesn't exist, add it
		newTaints = append(nodeTaints, *taint)
		updated = true
	}

	newNode.Spec.Taints = newTaints
	return newNode, updated, nil
}

// TaintExists checks if the given taint exists in list of taints. Returns true if exists false otherwise.
func TaintExists(taints []v1.Taint, taintToFind *v1.Taint) bool {
	for i := range taints {
		if taints[i].Key == taintToFind.Key && taints[i].Effect == taintToFind.Effect {
			return true
		}
	}
	return false
}

// TaintKeyExists checks if the given taint key exists in list of taints. Returns true if exists false otherwise.
func TaintKeyExists(taints []v1.Taint, taintKeyToMatch string) bool {
	for i := range taints {
		if taints[i].Key == taintKeyToMatch {
			return true
		}
	}
	return false
}

// TaintSetDiff finds the difference between two taint slices and
// returns all new and removed elements of the new slice relative to the old slice.
// for example:
// input: taintsNew=[a b] taintsOld=[a c]
// output: taintsToAdd=[b] taintsToRemove=[c]
func TaintSetDiff(taintsNew, taintsOld []v1.Taint) (taintsToAdd []*v1.Taint, taintsToRemove []*v1.Taint) {
	// Find taints to add (in new but not in old)
	for i := range taintsNew {
		if !TaintExists(taintsOld, &taintsNew[i]) {
			taintsToAdd = append(taintsToAdd, &taintsNew[i])
		}
	}

	// Find taints to remove (in old but not in new)
	for i := range taintsOld {
		if !TaintExists(taintsNew, &taintsOld[i]) {
			taintsToRemove = append(taintsToRemove, &taintsOld[i])
		}
	}

	return taintsToAdd, taintsToRemove
}

// TaintSetFilter filters from the taint slice according to the passed fn function to get the filtered taint slice.
func TaintSetFilter(taints []v1.Taint, fn func(*v1.Taint) bool) []v1.Taint {
	var result []v1.Taint
	for i := range taints {
		if fn(&taints[i]) {
			result = append(result, taints[i])
		}
	}
	return result
}

// CheckTaintValidation checks if the given taint is valid.
// Returns error if the given taint is invalid.
func CheckTaintValidation(taint v1.Taint) error {
	// Empty key is invalid
	if len(taint.Key) == 0 {
		return fmt.Errorf("invalid taint key: %v: %v", taint.Key, "name part must be non-empty")
	}

	// Validate key format
	if errs := validation.IsQualifiedName(taint.Key); len(errs) > 0 {
		return fmt.Errorf("invalid taint key: %v: %v", taint.Key, strings.Join(errs, "; "))
	}

	// Value exceeding 63 characters is invalid
	if len(taint.Value) > 63 {
		return fmt.Errorf("invalid taint value: %v: %v", taint.Value, "must be no more than 63 characters")
	}

	// Validate value if present
	if len(taint.Value) > 0 {
		if errs := validation.IsValidLabelValue(taint.Value); len(errs) > 0 {
			return fmt.Errorf("invalid taint value: %v: %v", taint.Value, strings.Join(errs, "; "))
		}
	}

	// Effect must be valid when non-empty
	if len(taint.Effect) > 0 {
		if err := validateTaintEffect(taint.Effect); err != nil {
			return err
		}
	}

	return nil
}
