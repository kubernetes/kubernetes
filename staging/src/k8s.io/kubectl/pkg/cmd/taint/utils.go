/*
Copyright 2018 The Kubernetes Authors.

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

// Package taints implements utilites for working with taints
package taint

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
)

// Exported taint constant strings
const (
	MODIFIED  = "modified"
	TAINTED   = "tainted"
	UNTAINTED = "untainted"
)

// parseTaints takes a spec which is an array and creates slices for new taints to be added, taints to be deleted.
// It also validates the spec. For example, the form `<key>` may be used to remove a taint, but not to add one.
func parseTaints(spec []string) ([]corev1.Taint, []corev1.Taint, error) {
	var taints, taintsToRemove []corev1.Taint
	uniqueTaints := map[corev1.TaintEffect]sets.String{}

	for _, taintSpec := range spec {
		if strings.HasSuffix(taintSpec, "-") {
			taintToRemove, err := parseTaint(strings.TrimSuffix(taintSpec, "-"))
			if err != nil {
				return nil, nil, err
			}
			taintsToRemove = append(taintsToRemove, corev1.Taint{Key: taintToRemove.Key, Effect: taintToRemove.Effect})
		} else {
			newTaint, err := parseTaint(taintSpec)
			if err != nil {
				return nil, nil, err
			}
			// validate that the taint has an effect, which is required to add the taint
			if len(newTaint.Effect) == 0 {
				return nil, nil, fmt.Errorf("invalid taint spec: %v", taintSpec)
			}
			// validate if taint is unique by <key, effect>
			if len(uniqueTaints[newTaint.Effect]) > 0 && uniqueTaints[newTaint.Effect].Has(newTaint.Key) {
				return nil, nil, fmt.Errorf("duplicated taints with the same key and effect: %v", newTaint)
			}
			// add taint to existingTaints for uniqueness check
			if len(uniqueTaints[newTaint.Effect]) == 0 {
				uniqueTaints[newTaint.Effect] = sets.String{}
			}
			uniqueTaints[newTaint.Effect].Insert(newTaint.Key)

			taints = append(taints, newTaint)
		}
	}
	return taints, taintsToRemove, nil
}

// parseTaint parses a taint from a string, whose form must be either
// '<key>=<value>:<effect>', '<key>:<effect>', or '<key>'.
func parseTaint(st string) (corev1.Taint, error) {
	var taint corev1.Taint

	var key string
	var value string
	var effect corev1.TaintEffect

	parts := strings.Split(st, ":")
	switch len(parts) {
	case 1:
		key = parts[0]
	case 2:
		effect = corev1.TaintEffect(parts[1])
		if err := validateTaintEffect(effect); err != nil {
			return taint, err
		}

		partsKV := strings.Split(parts[0], "=")
		if len(partsKV) > 2 {
			return taint, fmt.Errorf("invalid taint spec: %v", st)
		}
		key = partsKV[0]
		if len(partsKV) == 2 {
			value = partsKV[1]
			if errs := validation.IsValidLabelValue(value); len(errs) > 0 {
				return taint, fmt.Errorf("invalid taint spec: %v, %s", st, strings.Join(errs, "; "))
			}
		}
	default:
		return taint, fmt.Errorf("invalid taint spec: %v", st)
	}

	if errs := validation.IsQualifiedName(key); len(errs) > 0 {
		return taint, fmt.Errorf("invalid taint spec: %v, %s", st, strings.Join(errs, "; "))
	}

	taint.Key = key
	taint.Value = value
	taint.Effect = effect

	return taint, nil
}

func validateTaintEffect(effect corev1.TaintEffect) error {
	if effect != corev1.TaintEffectNoSchedule && effect != corev1.TaintEffectPreferNoSchedule && effect != corev1.TaintEffectNoExecute {
		return fmt.Errorf("invalid taint effect: %v, unsupported taint effect", effect)
	}

	return nil
}

// reorganizeTaints returns the updated set of taints, taking into account old taints that were not updated,
// old taints that were updated, old taints that were deleted, and new taints.
func reorganizeTaints(node *corev1.Node, overwrite bool, taintsToAdd []corev1.Taint, taintsToRemove []corev1.Taint) (string, []corev1.Taint, error) {
	newTaints := append([]corev1.Taint{}, taintsToAdd...)
	oldTaints := node.Spec.Taints
	// add taints that already existing but not updated to newTaints
	added := addTaints(oldTaints, &newTaints)
	allErrs, deleted := deleteTaints(taintsToRemove, &newTaints)
	if (added && deleted) || overwrite {
		return MODIFIED, newTaints, utilerrors.NewAggregate(allErrs)
	} else if added {
		return TAINTED, newTaints, utilerrors.NewAggregate(allErrs)
	}
	return UNTAINTED, newTaints, utilerrors.NewAggregate(allErrs)
}

// deleteTaints deletes the given taints from the node's taintlist.
func deleteTaints(taintsToRemove []corev1.Taint, newTaints *[]corev1.Taint) ([]error, bool) {
	allErrs := []error{}
	var removed bool
	for _, taintToRemove := range taintsToRemove {
		removed = false
		if len(taintToRemove.Effect) > 0 {
			*newTaints, removed = deleteTaint(*newTaints, &taintToRemove)
		} else {
			*newTaints, removed = deleteTaintsByKey(*newTaints, taintToRemove.Key)
		}
		if !removed {
			allErrs = append(allErrs, fmt.Errorf("taint %q not found", taintToRemove.ToString()))
		}
	}
	return allErrs, removed
}

// addTaints adds the newTaints list to existing ones and updates the newTaints List.
// TODO: This needs a rewrite to take only the new values instead of appended newTaints list to be consistent.
func addTaints(oldTaints []corev1.Taint, newTaints *[]corev1.Taint) bool {
	for _, oldTaint := range oldTaints {
		existsInNew := false
		for _, taint := range *newTaints {
			if taint.MatchTaint(&oldTaint) {
				existsInNew = true
				break
			}
		}
		if !existsInNew {
			*newTaints = append(*newTaints, oldTaint)
		}
	}
	return len(oldTaints) != len(*newTaints)
}

// checkIfTaintsAlreadyExists checks if the node already has taints that we want to add and returns a string with taint keys.
func checkIfTaintsAlreadyExists(oldTaints []corev1.Taint, taints []corev1.Taint) string {
	var existingTaintList = make([]string, 0)
	for _, taint := range taints {
		for _, oldTaint := range oldTaints {
			if taint.Key == oldTaint.Key && taint.Effect == oldTaint.Effect {
				existingTaintList = append(existingTaintList, taint.Key)
			}
		}
	}
	return strings.Join(existingTaintList, ",")
}

// deleteTaintsByKey removes all the taints that have the same key to given taintKey
func deleteTaintsByKey(taints []corev1.Taint, taintKey string) ([]corev1.Taint, bool) {
	newTaints := []corev1.Taint{}
	for i := range taints {
		if taintKey == taints[i].Key {
			continue
		}
		newTaints = append(newTaints, taints[i])
	}
	return newTaints, len(taints) != len(newTaints)
}

// deleteTaint removes all the taints that have the same key and effect to given taintToDelete.
func deleteTaint(taints []corev1.Taint, taintToDelete *corev1.Taint) ([]corev1.Taint, bool) {
	newTaints := []corev1.Taint{}
	for i := range taints {
		if taintToDelete.MatchTaint(&taints[i]) {
			continue
		}
		newTaints = append(newTaints, taints[i])
	}
	return newTaints, len(taints) != len(newTaints)
}
