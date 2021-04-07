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
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// ValidateDefaultPreemptionArgs validates that DefaultPreemptionArgs are correct.
func ValidateDefaultPreemptionArgs(args config.DefaultPreemptionArgs) error {
	var path *field.Path
	var allErrs field.ErrorList
	percentagePath := path.Child("minCandidateNodesPercentage")
	absolutePath := path.Child("minCandidateNodesAbsolute")
	if err := validateMinCandidateNodesPercentage(args.MinCandidateNodesPercentage, percentagePath); err != nil {
		allErrs = append(allErrs, err)
	}
	if err := validateMinCandidateNodesAbsolute(args.MinCandidateNodesAbsolute, absolutePath); err != nil {
		allErrs = append(allErrs, err)
	}
	if args.MinCandidateNodesPercentage == 0 && args.MinCandidateNodesAbsolute == 0 {
		allErrs = append(allErrs,
			field.Invalid(percentagePath, args.MinCandidateNodesPercentage, "cannot be zero at the same time as minCandidateNodesAbsolute"),
			field.Invalid(absolutePath, args.MinCandidateNodesAbsolute, "cannot be zero at the same time as minCandidateNodesPercentage"))
	}
	return allErrs.ToAggregate()
}

// validateMinCandidateNodesPercentage validates that
// minCandidateNodesPercentage is within the allowed range.
func validateMinCandidateNodesPercentage(minCandidateNodesPercentage int32, p *field.Path) *field.Error {
	if minCandidateNodesPercentage < 0 || minCandidateNodesPercentage > 100 {
		return field.Invalid(p, minCandidateNodesPercentage, "not in valid range [0, 100]")
	}
	return nil
}

// validateMinCandidateNodesAbsolute validates that minCandidateNodesAbsolute
// is within the allowed range.
func validateMinCandidateNodesAbsolute(minCandidateNodesAbsolute int32, p *field.Path) *field.Error {
	if minCandidateNodesAbsolute < 0 {
		return field.Invalid(p, minCandidateNodesAbsolute, "not in valid range [0, inf)")
	}
	return nil
}

// ValidateInterPodAffinityArgs validates that InterPodAffinityArgs are correct.
func ValidateInterPodAffinityArgs(args config.InterPodAffinityArgs) error {
	var path *field.Path
	return ValidateHardPodAffinityWeight(path.Child("hardPodAffinityWeight"), args.HardPodAffinityWeight)
}

// ValidateHardPodAffinityWeight validates that weight is within allowed range.
func ValidateHardPodAffinityWeight(path *field.Path, w int32) error {
	const (
		minHardPodAffinityWeight = 0
		maxHardPodAffinityWeight = 100
	)

	if w < minHardPodAffinityWeight || w > maxHardPodAffinityWeight {
		msg := fmt.Sprintf("not in valid range [%d, %d]", minHardPodAffinityWeight, maxHardPodAffinityWeight)
		return field.Invalid(path, w, msg)
	}
	return nil
}

// ValidateNodeLabelArgs validates that NodeLabelArgs are correct.
func ValidateNodeLabelArgs(args config.NodeLabelArgs) error {
	var path *field.Path
	var allErrs field.ErrorList

	allErrs = append(allErrs, validateNoConflict(args.PresentLabels, args.AbsentLabels,
		path.Child("presentLabels"), path.Child("absentLabels"))...)
	allErrs = append(allErrs, validateNoConflict(args.PresentLabelsPreference, args.AbsentLabelsPreference,
		path.Child("presentLabelsPreference"), path.Child("absentLabelsPreference"))...)

	return allErrs.ToAggregate()
}

// validateNoConflict validates that presentLabels and absentLabels do not conflict.
func validateNoConflict(presentLabels, absentLabels []string, presentPath, absentPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	m := make(map[string]int, len(presentLabels)) // label -> index
	for i, l := range presentLabels {
		m[l] = i
	}
	for i, l := range absentLabels {
		if j, ok := m[l]; ok {
			allErrs = append(allErrs, field.Invalid(presentPath.Index(j), l,
				fmt.Sprintf("conflict with %v", absentPath.Index(i).String())))
		}
	}
	return allErrs
}

// ValidatePodTopologySpreadArgs validates that PodTopologySpreadArgs are correct.
// It replicates the validation from pkg/apis/core/validation.validateTopologySpreadConstraints
// with an additional check for .labelSelector to be nil.
func ValidatePodTopologySpreadArgs(args *config.PodTopologySpreadArgs) error {
	var path *field.Path
	var allErrs field.ErrorList
	if err := validateDefaultingType(path.Child("defaultingType"), args.DefaultingType, args.DefaultConstraints); err != nil {
		allErrs = append(allErrs, err)
	}

	defaultConstraintsPath := path.Child("defaultConstraints")
	for i, c := range args.DefaultConstraints {
		p := defaultConstraintsPath.Index(i)
		if c.MaxSkew <= 0 {
			f := p.Child("maxSkew")
			allErrs = append(allErrs, field.Invalid(f, c.MaxSkew, "not in valid range (0, inf)"))
		}
		allErrs = append(allErrs, validateTopologyKey(p.Child("topologyKey"), c.TopologyKey)...)
		if err := validateWhenUnsatisfiable(p.Child("whenUnsatisfiable"), c.WhenUnsatisfiable); err != nil {
			allErrs = append(allErrs, err)
		}
		if c.LabelSelector != nil {
			f := field.Forbidden(p.Child("labelSelector"), "constraint must not define a selector, as they deduced for each pod")
			allErrs = append(allErrs, f)
		}
		if err := validateConstraintNotRepeat(defaultConstraintsPath, args.DefaultConstraints, i); err != nil {
			allErrs = append(allErrs, err)
		}
	}
	if len(allErrs) == 0 {
		return nil
	}
	return allErrs.ToAggregate()
}

func validateDefaultingType(p *field.Path, v config.PodTopologySpreadConstraintsDefaulting, constraints []v1.TopologySpreadConstraint) *field.Error {
	if v != config.SystemDefaulting && v != config.ListDefaulting {
		return field.NotSupported(p, v, []string{string(config.SystemDefaulting), string(config.ListDefaulting)})
	}
	if v == config.SystemDefaulting && len(constraints) > 0 {
		return field.Invalid(p, v, "when .defaultConstraints are not empty")
	}
	return nil
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

// ValidateRequestedToCapacityRatioArgs validates that RequestedToCapacityRatioArgs are correct.
func ValidateRequestedToCapacityRatioArgs(args config.RequestedToCapacityRatioArgs) error {
	var path *field.Path
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateFunctionShape(args.Shape, path.Child("shape"))...)
	allErrs = append(allErrs, validateResourcesNoMax(args.Resources, path.Child("resources"))...)
	return allErrs.ToAggregate()
}

func validateFunctionShape(shape []config.UtilizationShapePoint, path *field.Path) field.ErrorList {
	const (
		minUtilization = 0
		maxUtilization = 100
		minScore       = 0
		maxScore       = int32(config.MaxCustomPriorityScore)
	)

	var allErrs field.ErrorList

	if len(shape) == 0 {
		allErrs = append(allErrs, field.Required(path, "at least one point must be specified"))
		return allErrs
	}

	for i := 1; i < len(shape); i++ {
		if shape[i-1].Utilization >= shape[i].Utilization {
			allErrs = append(allErrs, field.Invalid(path.Index(i).Child("utilization"), shape[i].Utilization, "utilization values must be sorted in increasing order"))
			break
		}
	}

	for i, point := range shape {
		if point.Utilization < minUtilization || point.Utilization > maxUtilization {
			msg := fmt.Sprintf("not in valid range [%d, %d]", minUtilization, maxUtilization)
			allErrs = append(allErrs, field.Invalid(path.Index(i).Child("utilization"), point.Utilization, msg))
		}

		if point.Score < minScore || point.Score > maxScore {
			msg := fmt.Sprintf("not in valid range [%d, %d]", minScore, maxScore)
			allErrs = append(allErrs, field.Invalid(path.Index(i).Child("score"), point.Score, msg))
		}
	}

	return allErrs
}

// TODO potentially replace with validateResources
func validateResourcesNoMax(resources []config.ResourceSpec, p *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for i, r := range resources {
		if r.Weight < 1 {
			allErrs = append(allErrs, field.Invalid(p.Index(i).Child("weight"), r.Weight,
				fmt.Sprintf("resource weight of %s not in valid range [1, inf)", r.Name)))
		}
	}
	return allErrs
}

// ValidateNodeResourcesLeastAllocatedArgs validates that NodeResourcesLeastAllocatedArgs are correct.
func ValidateNodeResourcesLeastAllocatedArgs(args *config.NodeResourcesLeastAllocatedArgs) error {
	var path *field.Path
	return validateResources(args.Resources, path.Child("resources")).ToAggregate()
}

// ValidateNodeResourcesMostAllocatedArgs validates that NodeResourcesMostAllocatedArgs are correct.
func ValidateNodeResourcesMostAllocatedArgs(args *config.NodeResourcesMostAllocatedArgs) error {
	var path *field.Path
	return validateResources(args.Resources, path.Child("resources")).ToAggregate()
}

func validateResources(resources []config.ResourceSpec, p *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for i, resource := range resources {
		if resource.Weight <= 0 || resource.Weight > 100 {
			msg := fmt.Sprintf("resource weight of %v not in valid range (0, 100]", resource.Name)
			allErrs = append(allErrs, field.Invalid(p.Index(i).Child("weight"), resource.Weight, msg))
		}
	}
	return allErrs
}

// ValidateNodeAffinityArgs validates that NodeAffinityArgs are correct.
func ValidateNodeAffinityArgs(args *config.NodeAffinityArgs) error {
	if args.AddedAffinity == nil {
		return nil
	}
	affinity := args.AddedAffinity
	path := field.NewPath("addedAffinity")
	var errs []error
	if ns := affinity.RequiredDuringSchedulingIgnoredDuringExecution; ns != nil {
		_, err := nodeaffinity.NewNodeSelector(ns, field.WithPath(path.Child("requiredDuringSchedulingIgnoredDuringExecution")))
		if err != nil {
			errs = append(errs, err)
		}
	}
	// TODO: Add validation for requiredDuringSchedulingRequiredDuringExecution when it gets added to the API.
	if terms := affinity.PreferredDuringSchedulingIgnoredDuringExecution; len(terms) != 0 {
		_, err := nodeaffinity.NewPreferredSchedulingTerms(terms, field.WithPath(path.Child("preferredDuringSchedulingIgnoredDuringExecution")))
		if err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Flatten(errors.NewAggregate(errs))
}

// ValidateVolumeBindingArgs validates that VolumeBindingArgs are set correctly.
func ValidateVolumeBindingArgs(args *config.VolumeBindingArgs) error {
	var path *field.Path
	var err error

	if args.BindTimeoutSeconds < 0 {
		err = field.Invalid(path.Child("bindTimeoutSeconds"), args.BindTimeoutSeconds, "invalid BindTimeoutSeconds, should not be a negative value")
	}

	return err
}
