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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
)

// supportedScoringStrategyTypes has to be a set of strings for use with field.Unsupported
var supportedScoringStrategyTypes = sets.New(
	string(config.LeastAllocated),
	string(config.MostAllocated),
	string(config.RequestedToCapacityRatio),
)

// ValidateDefaultPreemptionArgs validates that DefaultPreemptionArgs are correct.
func ValidateDefaultPreemptionArgs(path *field.Path, args *config.DefaultPreemptionArgs) error {
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
func ValidateInterPodAffinityArgs(path *field.Path, args *config.InterPodAffinityArgs) error {
	return validateHardPodAffinityWeight(path.Child("hardPodAffinityWeight"), args.HardPodAffinityWeight)
}

// validateHardPodAffinityWeight validates that weight is within allowed range.
func validateHardPodAffinityWeight(path *field.Path, w int32) error {
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

// ValidatePodTopologySpreadArgs validates that PodTopologySpreadArgs are correct.
// It replicates the validation from pkg/apis/core/validation.validateTopologySpreadConstraints
// with an additional check for .labelSelector to be nil.
func ValidatePodTopologySpreadArgs(path *field.Path, args *config.PodTopologySpreadArgs) error {
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
	supportedScheduleActions := sets.New(string(v1.DoNotSchedule), string(v1.ScheduleAnyway))

	if len(v) == 0 {
		return field.Required(p, "can not be empty")
	}
	if !supportedScheduleActions.Has(string(v)) {
		return field.NotSupported(p, v, sets.List(supportedScheduleActions))
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
			allErrs = append(allErrs, field.Invalid(path.Index(i).Child("utilization"), shape[i].Utilization, "values must be sorted in increasing order"))
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

// ValidateNodeResourcesBalancedAllocationArgs validates that NodeResourcesBalancedAllocationArgs are set correctly.
func ValidateNodeResourcesBalancedAllocationArgs(path *field.Path, args *config.NodeResourcesBalancedAllocationArgs) error {
	var allErrs field.ErrorList
	seenResources := sets.New[string]()
	for i, resource := range args.Resources {
		if seenResources.Has(resource.Name) {
			allErrs = append(allErrs, field.Duplicate(path.Child("resources").Index(i).Child("name"), resource.Name))
		} else {
			seenResources.Insert(resource.Name)
		}
		if resource.Weight != 1 {
			allErrs = append(allErrs, field.Invalid(path.Child("resources").Index(i).Child("weight"), resource.Weight, "must be 1"))
		}
	}
	return allErrs.ToAggregate()
}

// ValidateNodeAffinityArgs validates that NodeAffinityArgs are correct.
func ValidateNodeAffinityArgs(path *field.Path, args *config.NodeAffinityArgs) error {
	if args.AddedAffinity == nil {
		return nil
	}
	affinity := args.AddedAffinity
	var errs []error
	if ns := affinity.RequiredDuringSchedulingIgnoredDuringExecution; ns != nil {
		_, err := nodeaffinity.NewNodeSelector(ns, field.WithPath(path.Child("addedAffinity", "requiredDuringSchedulingIgnoredDuringExecution")))
		if err != nil {
			errs = append(errs, err)
		}
	}
	// TODO: Add validation for requiredDuringSchedulingRequiredDuringExecution when it gets added to the API.
	if terms := affinity.PreferredDuringSchedulingIgnoredDuringExecution; len(terms) != 0 {
		_, err := nodeaffinity.NewPreferredSchedulingTerms(terms, field.WithPath(path.Child("addedAffinity", "preferredDuringSchedulingIgnoredDuringExecution")))
		if err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Flatten(errors.NewAggregate(errs))
}

// VolumeBindingArgsValidationOptions contains the different settings for validation.
type VolumeBindingArgsValidationOptions struct {
	AllowStorageCapacityScoring bool
}

// ValidateVolumeBindingArgs validates that VolumeBindingArgs are set correctly.
func ValidateVolumeBindingArgs(path *field.Path, args *config.VolumeBindingArgs) error {
	return ValidateVolumeBindingArgsWithOptions(path, args, VolumeBindingArgsValidationOptions{
		AllowStorageCapacityScoring: utilfeature.DefaultFeatureGate.Enabled(features.StorageCapacityScoring),
	})
}

// ValidateVolumeBindingArgsWithOptions validates that VolumeBindingArgs and VolumeBindingArgsValidationOptions with scheduler features.
func ValidateVolumeBindingArgsWithOptions(path *field.Path, args *config.VolumeBindingArgs, opts VolumeBindingArgsValidationOptions) error {
	var allErrs field.ErrorList

	if args.BindTimeoutSeconds < 0 {
		allErrs = append(allErrs, field.Invalid(path.Child("bindTimeoutSeconds"), args.BindTimeoutSeconds, "invalid BindTimeoutSeconds, should not be a negative value"))
	}

	if opts.AllowStorageCapacityScoring {
		allErrs = append(allErrs, validateFunctionShape(args.Shape, path.Child("shape"))...)
	} else if args.Shape != nil {
		// When the feature is off, return an error if the config is not nil.
		// This prevents unexpected configuration from taking effect when the
		// feature turns on in the future.
		allErrs = append(allErrs, field.Invalid(path.Child("shape"), args.Shape, "unexpected field `shape`, remove it or turn on the feature gate StorageCapacityScoring"))
	}
	return allErrs.ToAggregate()
}

func ValidateNodeResourcesFitArgs(path *field.Path, args *config.NodeResourcesFitArgs) error {
	var allErrs field.ErrorList
	resPath := path.Child("ignoredResources")
	for i, res := range args.IgnoredResources {
		path := resPath.Index(i)
		if errs := metav1validation.ValidateLabelName(res, path); len(errs) != 0 {
			allErrs = append(allErrs, errs...)
		}
	}

	groupPath := path.Child("ignoredResourceGroups")
	for i, group := range args.IgnoredResourceGroups {
		path := groupPath.Index(i)
		if strings.Contains(group, "/") {
			allErrs = append(allErrs, field.Invalid(path, group, "resource group name can't contain '/'"))
		}
		if errs := metav1validation.ValidateLabelName(group, path); len(errs) != 0 {
			allErrs = append(allErrs, errs...)
		}
	}

	strategyPath := path.Child("scoringStrategy")
	if args.ScoringStrategy != nil {
		if !supportedScoringStrategyTypes.Has(string(args.ScoringStrategy.Type)) {
			allErrs = append(allErrs, field.NotSupported(strategyPath.Child("type"), args.ScoringStrategy.Type, sets.List(supportedScoringStrategyTypes)))
		}
		allErrs = append(allErrs, validateResources(args.ScoringStrategy.Resources, strategyPath.Child("resources"))...)
		if args.ScoringStrategy.RequestedToCapacityRatio != nil {
			allErrs = append(allErrs, validateFunctionShape(args.ScoringStrategy.RequestedToCapacityRatio.Shape, strategyPath.Child("shape"))...)
		}
	}

	if len(allErrs) == 0 {
		return nil
	}
	return allErrs.ToAggregate()
}

// ValidateDynamicResourcesArgs validates that DynamicResourcesArgs are correct.
// In contrast to the REST API, setting fields that have no effect because
// the corresponding feature is disabled is considered an error.
func ValidateDynamicResourcesArgs(path *field.Path, args *config.DynamicResourcesArgs, fts feature.Features) error {
	var allErrs field.ErrorList
	if fts.EnableDRASchedulerFilterTimeout {
		if args.FilterTimeout != nil && args.FilterTimeout.Duration < 0 {
			allErrs = append(allErrs, field.Invalid(path.Child("filterTimeout"), args.FilterTimeout, "must be zero or positive"))
		}
	} else {
		if args.FilterTimeout != nil {
			allErrs = append(allErrs, field.Forbidden(path.Child("filterTimeout"), "DRASchedulingFilterTimeout feature gate is disabled"))
		}
	}

	if fts.EnableDRADeviceBindingConditions && fts.EnableDRAResourceClaimDeviceStatus {
		if args.BindingTimeout != nil && args.BindingTimeout.Duration < 1*time.Second {
			allErrs = append(allErrs, field.Invalid(
				path.Child("bindingTimeout"),
				args.BindingTimeout,
				"must be at least 1 second",
			))
		}
	} else {
		if args.BindingTimeout != nil {
			allErrs = append(allErrs, field.Forbidden(
				path.Child("bindingTimeout"),
				"DRADeviceBindingConditions or DRAResourceClaimDeviceStatus feature gate is disabled",
			))
		}
	}
	return allErrs.ToAggregate()
}
