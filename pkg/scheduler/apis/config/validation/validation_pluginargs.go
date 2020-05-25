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

package validation

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// ValidateInterPodAffinityArgs validates that InterPodAffinityArgs are correct.
func ValidateInterPodAffinityArgs(args config.InterPodAffinityArgs) error {
	return ValidateHardPodAffinityWeight(field.NewPath("hardPodAffinityWeight"), args.HardPodAffinityWeight)
}

// ValidateHardPodAffinityWeight validates that weight is within allowed range.
func ValidateHardPodAffinityWeight(path *field.Path, w int32) error {
	const (
		minHardPodAffinityWeight = 0
		maxHardPodAffinityWeight = 100
	)

	if w < minHardPodAffinityWeight || w > maxHardPodAffinityWeight {
		msg := fmt.Sprintf("not in valid range [%d-%d]", minHardPodAffinityWeight, maxHardPodAffinityWeight)
		return field.Invalid(path, w, msg)
	}
	return nil
}

// ValidateNodeLabelArgs validates that NodeLabelArgs are correct.
func ValidateNodeLabelArgs(args config.NodeLabelArgs) error {
	if err := validateNoConflict(args.PresentLabels, args.AbsentLabels); err != nil {
		return err
	}
	if err := validateNoConflict(args.PresentLabelsPreference, args.AbsentLabelsPreference); err != nil {
		return err
	}
	return nil
}

// validateNoConflict validates that presentLabels and absentLabels do not conflict.
func validateNoConflict(presentLabels []string, absentLabels []string) error {
	m := make(map[string]struct{}, len(presentLabels))
	for _, l := range presentLabels {
		m[l] = struct{}{}
	}
	for _, l := range absentLabels {
		if _, ok := m[l]; ok {
			return fmt.Errorf("detecting at least one label (e.g., %q) that exist in both the present(%+v) and absent(%+v) label list", l, presentLabels, absentLabels)
		}
	}
	return nil
}

// ValidatePodTopologySpreadArgs validates that PodTopologySpreadArgs are correct.
// It replicates the validation from pkg/apis/core/validation.validateTopologySpreadConstraints
// with an additional check for .labelSelector to be nil.
func ValidatePodTopologySpreadArgs(args *config.PodTopologySpreadArgs) error {
	var allErrs field.ErrorList
	path := field.NewPath("defaultConstraints")

	for i, c := range args.DefaultConstraints {
		p := path.Index(i)
		if c.MaxSkew <= 0 {
			f := p.Child("maxSkew")
			allErrs = append(allErrs, field.Invalid(f, c.MaxSkew, "must be greater than zero"))
		}
		allErrs = append(allErrs, validateTopologyKey(p.Child("topologyKey"), c.TopologyKey)...)
		if err := validateWhenUnsatisfiable(p.Child("whenUnsatisfiable"), c.WhenUnsatisfiable); err != nil {
			allErrs = append(allErrs, err)
		}
		if c.LabelSelector != nil {
			f := field.Forbidden(p.Child("labelSelector"), "constraint must not define a selector, as they deduced for each pod")
			allErrs = append(allErrs, f)
		}
		if err := validateConstraintNotRepeat(path, args.DefaultConstraints, i); err != nil {
			allErrs = append(allErrs, err)
		}
	}
	if len(allErrs) == 0 {
		return nil
	}
	return allErrs.ToAggregate()
}

func validateTopologyKey(p *field.Path, v string) field.ErrorList {
	var allErrs field.ErrorList
	if len(v) == 0 {
		allErrs = append(allErrs, field.Required(p, "can not be empty"))
	} else {
		allErrs = append(allErrs, metav1validation.ValidateLabelName(v, p)...)
	}
	return allErrs
}

func validateWhenUnsatisfiable(p *field.Path, v v1.UnsatisfiableConstraintAction) *field.Error {
	supportedScheduleActions := sets.NewString(string(v1.DoNotSchedule), string(v1.ScheduleAnyway))

	if len(v) == 0 {
		return field.Required(p, "can not be empty")
	}
	if !supportedScheduleActions.Has(string(v)) {
		return field.NotSupported(p, v, supportedScheduleActions.List())
	}
	return nil
}

func validateConstraintNotRepeat(path *field.Path, constraints []v1.TopologySpreadConstraint, idx int) *field.Error {
	c := &constraints[idx]
	for i := range constraints[:idx] {
		other := &constraints[i]
		if c.TopologyKey == other.TopologyKey && c.WhenUnsatisfiable == other.WhenUnsatisfiable {
			return field.Duplicate(path.Index(idx), fmt.Sprintf("{%v, %v}", c.TopologyKey, c.WhenUnsatisfiable))
		}
	}
	return nil
}
