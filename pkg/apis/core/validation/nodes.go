/*
Copyright 2014 The Kubernetes Authors.

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
	"net"
	"strings"

	"github.com/golang/glog"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"
)

// ValidateNodeName can be used to check whether the given node name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateNodeName = NameIsDNSSubdomain

// ValidateNode tests if required fields in the node are set.
func ValidateNode(node *core.Node) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMeta(&node.ObjectMeta, false, ValidateNodeName, fldPath)
	allErrs = append(allErrs, ValidateNodeSpecificAnnotations(node.ObjectMeta.Annotations, fldPath.Child("annotations"))...)
	if len(node.Spec.Taints) > 0 {
		allErrs = append(allErrs, validateNodeTaints(node.Spec.Taints, fldPath.Child("taints"))...)
	}

	// Only validate spec.
	// All status fields are optional and can be updated later.
	// That said, if specified, we need to ensure they are valid.
	allErrs = append(allErrs, ValidateNodeResources(node)...)

	// external ID is required.
	if len(node.Spec.ExternalID) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "externalID"), ""))
	}

	// Only allow Node.Spec.ConfigSource to be set if the DynamicKubeletConfig feature gate is enabled
	if node.Spec.ConfigSource != nil && !utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec", "configSource"), "configSource may only be set if the DynamicKubeletConfig feature gate is enabled)"))
	}

	if len(node.Spec.PodCIDR) != 0 {
		_, err := ValidateCIDR(node.Spec.PodCIDR)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "podCIDR"), node.Spec.PodCIDR, "not a valid CIDR"))
		}
	}
	return allErrs
}

// ValidateNodeUpdate tests to make sure a node update can be applied.  Modifies oldNode.
func ValidateNodeUpdate(node, oldNode *core.Node) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMetaUpdate(&node.ObjectMeta, &oldNode.ObjectMeta, fldPath)
	allErrs = append(allErrs, ValidateNodeSpecificAnnotations(node.ObjectMeta.Annotations, fldPath.Child("annotations"))...)

	// TODO: Enable the code once we have better core object.status update model. Currently,
	// anyone can update node status.
	// if !apiequality.Semantic.DeepEqual(node.Status, core.NodeStatus{}) {
	// 	allErrs = append(allErrs, field.Invalid("status", node.Status, "must be empty"))
	// }

	allErrs = append(allErrs, ValidateNodeResources(node)...)

	// Validate no duplicate addresses in node status.
	addresses := make(map[core.NodeAddress]bool)
	for i, address := range node.Status.Addresses {
		if _, ok := addresses[address]; ok {
			allErrs = append(allErrs, field.Duplicate(field.NewPath("status", "addresses").Index(i), address))
		}
		addresses[address] = true
	}

	if len(oldNode.Spec.PodCIDR) == 0 {
		// Allow the controller manager to assign a CIDR to a node if it doesn't have one.
		oldNode.Spec.PodCIDR = node.Spec.PodCIDR
	} else {
		if oldNode.Spec.PodCIDR != node.Spec.PodCIDR {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("spec", "podCIDR"), "node updates may not change podCIDR except from \"\" to valid"))
		}
	}

	// Allow controller manager updating provider ID when not set
	if len(oldNode.Spec.ProviderID) == 0 {
		oldNode.Spec.ProviderID = node.Spec.ProviderID
	} else {
		if oldNode.Spec.ProviderID != node.Spec.ProviderID {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("spec", "providerID"), "node updates may not change providerID except from \"\" to valid"))
		}
	}

	// TODO: move reset function to its own location
	// Ignore metadata changes now that they have been tested
	oldNode.ObjectMeta = node.ObjectMeta
	// Allow users to update capacity
	oldNode.Status.Capacity = node.Status.Capacity
	// Allow users to unschedule node
	oldNode.Spec.Unschedulable = node.Spec.Unschedulable
	// Clear status
	oldNode.Status = node.Status

	// update taints
	if len(node.Spec.Taints) > 0 {
		allErrs = append(allErrs, validateNodeTaints(node.Spec.Taints, fldPath.Child("taints"))...)
	}
	oldNode.Spec.Taints = node.Spec.Taints

	// Allow updates to Node.Spec.ConfigSource if DynamicKubeletConfig feature gate is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		oldNode.Spec.ConfigSource = node.Spec.ConfigSource
	}

	// We made allowed changes to oldNode, and now we compare oldNode to node. Any remaining differences indicate changes to protected fields.
	// TODO: Add a 'real' error type for this error and provide print actual diffs.
	if !apiequality.Semantic.DeepEqual(oldNode, node) {
		glog.V(4).Infof("Update failed validation %#v vs %#v", oldNode, node)
		allErrs = append(allErrs, field.Forbidden(field.NewPath(""), "node updates may only change labels, taints, or capacity (or configSource, if the DynamicKubeletConfig feature gate is enabled)"))
	}

	return allErrs
}

// ValidateCIDR validates whether a CIDR matches the conventions expected by net.ParseCIDR
func ValidateCIDR(cidr string) (*net.IPNet, error) {
	_, net, err := net.ParseCIDR(cidr)
	if err != nil {
		return nil, err
	}
	return net, nil
}

// ValidateTaintsInNodeAnnotations tests that the serialized taints in Node.Annotations has valid data
func ValidateTaintsInNodeAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	taints, err := helper.GetTaintsFromNodeAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, core.TaintsAnnotationKey, err.Error()))
		return allErrs
	}

	if len(taints) > 0 {
		allErrs = append(allErrs, validateNodeTaints(taints, fldPath.Child(core.TaintsAnnotationKey))...)
	}

	return allErrs
}

func ValidateNodeSpecificAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if annotations[core.TaintsAnnotationKey] != "" {
		allErrs = append(allErrs, ValidateTaintsInNodeAnnotations(annotations, fldPath)...)
	}

	if annotations[core.PreferAvoidPodsAnnotationKey] != "" {
		allErrs = append(allErrs, ValidateAvoidPodsInNodeAnnotations(annotations, fldPath)...)
	}
	return allErrs
}

// ValidateNodeResources is used to make sure a node has valid capacity and allocatable values.
func ValidateNodeResources(node *core.Node) field.ErrorList {
	allErrs := field.ErrorList{}
	// Validate resource quantities in capacity.
	hugePageSizes := sets.NewString()
	for k, v := range node.Status.Capacity {
		resPath := field.NewPath("status", "capacity", string(k))
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(k), v, resPath)...)
		// track any huge page size that has a positive value
		if helper.IsHugePageResourceName(k) && v.Value() > int64(0) {
			hugePageSizes.Insert(string(k))
		}
		if len(hugePageSizes) > 1 {
			allErrs = append(allErrs, field.Invalid(resPath, v, "may not have pre-allocated hugepages for multiple page sizes"))
		}
	}
	// Validate resource quantities in allocatable.
	hugePageSizes = sets.NewString()
	for k, v := range node.Status.Allocatable {
		resPath := field.NewPath("status", "allocatable", string(k))
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(k), v, resPath)...)
		// track any huge page size that has a positive value
		if helper.IsHugePageResourceName(k) && v.Value() > int64(0) {
			hugePageSizes.Insert(string(k))
		}
		if len(hugePageSizes) > 1 {
			allErrs = append(allErrs, field.Invalid(resPath, v, "may not have pre-allocated hugepages for multiple page sizes"))
		}
	}
	return allErrs
}

// ValidateNodeSelectorRequirement tests that the specified NodeSelectorRequirement fields has valid data
func ValidateNodeSelectorRequirement(rq core.NodeSelectorRequirement, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch rq.Operator {
	case core.NodeSelectorOpIn, core.NodeSelectorOpNotIn:
		if len(rq.Values) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("values"), "must be specified when `operator` is 'In' or 'NotIn'"))
		}
	case core.NodeSelectorOpExists, core.NodeSelectorOpDoesNotExist:
		if len(rq.Values) > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("values"), "may not be specified when `operator` is 'Exists' or 'DoesNotExist'"))
		}

	case core.NodeSelectorOpGt, core.NodeSelectorOpLt:
		if len(rq.Values) != 1 {
			allErrs = append(allErrs, field.Required(fldPath.Child("values"), "must be specified single value when `operator` is 'Lt' or 'Gt'"))
		}
	default:
		allErrs = append(allErrs, field.Invalid(fldPath.Child("operator"), rq.Operator, "not a valid selector operator"))
	}
	allErrs = append(allErrs, unversionedvalidation.ValidateLabelName(rq.Key, fldPath.Child("key"))...)
	return allErrs
}

// ValidateNodeSelectorTerm tests that the specified node selector term has valid data
func ValidateNodeSelectorTerm(term core.NodeSelectorTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(term.MatchExpressions) == 0 {
		return append(allErrs, field.Required(fldPath.Child("matchExpressions"), "must have at least one node selector requirement"))
	}
	for j, req := range term.MatchExpressions {
		allErrs = append(allErrs, ValidateNodeSelectorRequirement(req, fldPath.Child("matchExpressions").Index(j))...)
	}
	return allErrs
}

// ValidateNodeSelector tests that the specified nodeSelector fields has valid data
func ValidateNodeSelector(nodeSelector *core.NodeSelector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	termFldPath := fldPath.Child("nodeSelectorTerms")
	if len(nodeSelector.NodeSelectorTerms) == 0 {
		return append(allErrs, field.Required(termFldPath, "must have at least one node selector term"))
	}

	for i, term := range nodeSelector.NodeSelectorTerms {
		allErrs = append(allErrs, ValidateNodeSelectorTerm(term, termFldPath.Index(i))...)
	}

	return allErrs
}

// ValidatePreferredSchedulingTerms tests that the specified SoftNodeAffinity fields has valid data
func ValidatePreferredSchedulingTerms(terms []core.PreferredSchedulingTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for i, term := range terms {
		if term.Weight <= 0 || term.Weight > 100 {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("weight"), term.Weight, "must be in the range 1-100"))
		}

		allErrs = append(allErrs, ValidateNodeSelectorTerm(term.Preference, fldPath.Index(i).Child("preference"))...)
	}
	return allErrs
}

// validateNodeTaints tests if given taints have valid data.
func validateNodeTaints(taints []core.Taint, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}

	uniqueTaints := map[core.TaintEffect]sets.String{}

	for i, currTaint := range taints {
		idxPath := fldPath.Index(i)
		// validate the taint key
		allErrors = append(allErrors, unversionedvalidation.ValidateLabelName(currTaint.Key, idxPath.Child("key"))...)
		// validate the taint value
		if errs := validation.IsValidLabelValue(currTaint.Value); len(errs) != 0 {
			allErrors = append(allErrors, field.Invalid(idxPath.Child("value"), currTaint.Value, strings.Join(errs, ";")))
		}
		// validate the taint effect
		allErrors = append(allErrors, validateTaintEffect(&currTaint.Effect, false, idxPath.Child("effect"))...)

		// validate if taint is unique by <key, effect>
		if len(uniqueTaints[currTaint.Effect]) > 0 && uniqueTaints[currTaint.Effect].Has(currTaint.Key) {
			duplicatedError := field.Duplicate(idxPath, currTaint)
			duplicatedError.Detail = "taints must be unique by key and effect pair"
			allErrors = append(allErrors, duplicatedError)
			continue
		}

		// add taint to existingTaints for uniqueness check
		if len(uniqueTaints[currTaint.Effect]) == 0 {
			uniqueTaints[currTaint.Effect] = sets.String{}
		}
		uniqueTaints[currTaint.Effect].Insert(currTaint.Key)
	}
	return allErrors
}

// validateNodeAffinity tests that the specified nodeAffinity fields have valid data
func validateNodeAffinity(na *core.NodeAffinity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO: Uncomment the next three lines once RequiredDuringSchedulingRequiredDuringExecution is implemented.
	// if na.RequiredDuringSchedulingRequiredDuringExecution != nil {
	//	allErrs = append(allErrs, ValidateNodeSelector(na.RequiredDuringSchedulingRequiredDuringExecution, fldPath.Child("requiredDuringSchedulingRequiredDuringExecution"))...)
	// }
	if na.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		allErrs = append(allErrs, ValidateNodeSelector(na.RequiredDuringSchedulingIgnoredDuringExecution, fldPath.Child("requiredDuringSchedulingIgnoredDuringExecution"))...)
	}
	if len(na.PreferredDuringSchedulingIgnoredDuringExecution) > 0 {
		allErrs = append(allErrs, ValidatePreferredSchedulingTerms(na.PreferredDuringSchedulingIgnoredDuringExecution, fldPath.Child("preferredDuringSchedulingIgnoredDuringExecution"))...)
	}
	return allErrs
}
