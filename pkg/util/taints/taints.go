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

	// Split by ':' to separate key-value from effect
	parts := strings.Split(st, ":")
	if len(parts) > 2 {
		return taint, fmt.Errorf("invalid taint spec: %v", st)
	}

	// Parse key and value
	keyValue := parts[0]
	if strings.Contains(keyValue, "=") {
		kvParts := strings.SplitN(keyValue, "=", 2)
		if len(kvParts) != 2 {
			return taint, fmt.Errorf("invalid taint spec: %v", st)
		}
		taint.Key = kvParts[0]
		taint.Value = kvParts[1]
	} else {
		taint.Key = keyValue
	}

	// Validate key
	if errs := validation.IsQualifiedName(taint.Key); len(errs) > 0 {
		return taint, fmt.Errorf("invalid taint key: %s, %s", taint.Key, strings.Join(errs, "; "))
	}

	// Validate value
	if taint.Value != "" {
		if errs := validation.IsValidLabelValue(taint.Value); len(errs) > 0 {
			return taint, fmt.Errorf("invalid taint value: %s, %s", taint.Value, strings.Join(errs, "; "))
		}
	}

	// Parse effect if present
	if len(parts) == 2 {
		taint.Effect = v1.TaintEffect(parts[1])
		if err := validateTaintEffect(taint.Effect); err != nil {
			return taint, err
		}
	}

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

	// Track taints to add by key:effect to detect duplicates
	uniqueTaints := make(map[string]bool)

	for _, taintSpec := range spec {
		if strings.HasSuffix(taintSpec, "-") {
			// Remove operation
			taintSpec = strings.TrimSuffix(taintSpec, "-")
			taint, err := parseTaint(taintSpec)
			if err != nil {
				return nil, nil, err
			}
			taintsToRemove = append(taintsToRemove, taint)
		} else {
			// Add operation
			taint, err := parseTaint(taintSpec)
			if err != nil {
				return nil, nil, err
			}

			// Validate that add operations have an effect
			if taint.Effect == "" {
				return nil, nil, fmt.Errorf("invalid taint spec: %v", taintSpec)
			}

			// Check for duplicate taints
			taintKey := fmt.Sprintf("%s:%s", taint.Key, taint.Effect)
			if uniqueTaints[taintKey] {
				return nil, nil, fmt.Errorf("duplicated taints with the same key and effect: %v", taintKey)
			}
			uniqueTaints[taintKey] = true

			taintsToAdd = append(taintsToAdd, taint)
		}
	}

	// Check for conflicts between add and remove operations
	for _, taintToAdd := range taintsToAdd {
		for _, taintToRemove := range taintsToRemove {
			if taintToAdd.MatchTaint(&taintToRemove) {
				return nil, nil, fmt.Errorf("can not both modify and remove the following taint(s) in the same command: %v", taintToAdd.ToString())
			}
		}
	}

	return taintsToAdd, taintsToRemove, nil
}

// CheckIfTaintsAlreadyExists checks if the node already has taints that we want to add and returns a string with taint keys.
func CheckIfTaintsAlreadyExists(oldTaints []v1.Taint, taints []v1.Taint) string {
	var existingTaints []string
	for _, taint := range taints {
		for _, oldTaint := range oldTaints {
			if taint.MatchTaint(&oldTaint) {
				existingTaints = append(existingTaints, taint.Key)
				break
			}
		}
	}
	return strings.Join(existingTaints, ",")
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
		if taint.MatchTaint(taintToDelete) {
			deleted = true
		} else {
			newTaints = append(newTaints, taint)
		}
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
	found := false

	for _, nodeTaint := range nodeTaints {
		if nodeTaint.MatchTaint(taint) {
			found = true
			// Update existing taint if value differs
			if nodeTaint.Value != taint.Value {
				newTaints = append(newTaints, *taint)
				updated = true
			} else {
				// No change needed
				newTaints = append(newTaints, nodeTaint)
			}
		} else {
			newTaints = append(newTaints, nodeTaint)
		}
	}

	// If not found, add the taint
	if !found {
		newTaints = append(newTaints, *taint)
		updated = true
	}

	if updated {
		newNode.Spec.Taints = newTaints
	}
	return newNode, updated, nil
}

// TaintExists checks if the given taint exists in list of taints. Returns true if exists false otherwise.
func TaintExists(taints []v1.Taint, taintToFind *v1.Taint) bool {
	for _, taint := range taints {
		if taint.MatchTaint(taintToFind) {
			return true
		}
	}
	return false
}

// TaintKeyExists checks if the given taint key exists in list of taints. Returns true if exists false otherwise.
func TaintKeyExists(taints []v1.Taint, taintKeyToMatch string) bool {
	for _, taint := range taints {
		if taint.Key == taintKeyToMatch {
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
		found := false
		for _, oldTaint := range taintsOld {
			if taintsNew[i].MatchTaint(&oldTaint) {
				found = true
				break
			}
		}
		if !found {
			taintsToAdd = append(taintsToAdd, &taintsNew[i])
		}
	}

	// Find taints to remove (in old but not in new)
	for i := range taintsOld {
		found := false
		for _, newTaint := range taintsNew {
			if taintsOld[i].MatchTaint(&newTaint) {
				found = true
				break
			}
		}
		if !found {
			taintsToRemove = append(taintsToRemove, &taintsOld[i])
		}
	}

	return taintsToAdd, taintsToRemove
}

// TaintSetFilter filters from the taint slice according to the passed fn function to get the filtered taint slice.
func TaintSetFilter(taints []v1.Taint, fn func(*v1.Taint) bool) []v1.Taint {
	var filteredTaints []v1.Taint
	for i := range taints {
		if fn(&taints[i]) {
			filteredTaints = append(filteredTaints, taints[i])
		}
	}
	return filteredTaints
}

// CheckTaintValidation checks if the given taint is valid.
// Returns error if the given taint is invalid.
func CheckTaintValidation(taint v1.Taint) error {
	// Empty key is invalid
	if taint.Key == "" {
		return fmt.Errorf("invalid taint key: Key is empty")
	}

	// Value exceeding 63 characters is invalid
	if len(taint.Value) > 63 {
		return fmt.Errorf("invalid taint value: Value is too long")
	}

	// Effect must be valid enum value when non-empty
	if taint.Effect != "" {
		if err := validateTaintEffect(taint.Effect); err != nil {
			return err
		}
	}

	return nil
}
