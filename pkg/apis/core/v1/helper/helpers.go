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

package helper

import (
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

// IsExtendedResourceName returns true if:
// 1. the resource name is not in the default namespace;
// 2. resource name does not have "requests." prefix,
// to avoid confusion with the convention in quota
// 3. it satisfies the rules in IsQualifiedName() after converted into quota resource name
func IsExtendedResourceName(name v1.ResourceName) bool {
	if IsNativeResource(name) || strings.HasPrefix(string(name), v1.DefaultResourceRequestsPrefix) {
		return false
	}
	// Ensure it satisfies the rules in IsQualifiedName() after converted into quota resource name
	nameForQuota := fmt.Sprintf("%s%s", v1.DefaultResourceRequestsPrefix, string(name))
	if errs := validation.IsQualifiedName(nameForQuota); len(errs) != 0 {
		return false
	}
	return true
}

// IsPrefixedNativeResource returns true if the resource name is in the
// *kubernetes.io/ namespace.
func IsPrefixedNativeResource(name v1.ResourceName) bool {
	return strings.Contains(string(name), v1.ResourceDefaultNamespacePrefix)
}

// IsNativeResource returns true if the resource name is in the
// *kubernetes.io/ namespace. Partially-qualified (unprefixed) names are
// implicitly in the kubernetes.io/ namespace.
func IsNativeResource(name v1.ResourceName) bool {
	return !strings.Contains(string(name), "/") ||
		IsPrefixedNativeResource(name)
}

// IsHugePageResourceName returns true if the resource name has the huge page
// resource prefix.
func IsHugePageResourceName(name v1.ResourceName) bool {
	return strings.HasPrefix(string(name), v1.ResourceHugePagesPrefix)
}

// HugePageResourceName returns a ResourceName with the canonical hugepage
// prefix prepended for the specified page size.  The page size is converted
// to its canonical representation.
func HugePageResourceName(pageSize resource.Quantity) v1.ResourceName {
	return v1.ResourceName(fmt.Sprintf("%s%s", v1.ResourceHugePagesPrefix, pageSize.String()))
}

// HugePageSizeFromResourceName returns the page size for the specified huge page
// resource name.  If the specified input is not a valid huge page resource name
// an error is returned.
func HugePageSizeFromResourceName(name v1.ResourceName) (resource.Quantity, error) {
	if !IsHugePageResourceName(name) {
		return resource.Quantity{}, fmt.Errorf("resource name: %s is an invalid hugepage name", name)
	}
	pageSize := strings.TrimPrefix(string(name), v1.ResourceHugePagesPrefix)
	return resource.ParseQuantity(pageSize)
}

// HugePageUnitSizeFromByteSize returns hugepage size has the format.
// `size` must be guaranteed to divisible into the largest units that can be expressed.
// <size><unit-prefix>B (1024 = "1KB", 1048576 = "1MB", etc).
func HugePageUnitSizeFromByteSize(size int64) (string, error) {
	// hugePageSizeUnitList is borrowed from opencontainers/runc/libcontainer/cgroups/utils.go
	var hugePageSizeUnitList = []string{"B", "KB", "MB", "GB", "TB", "PB"}
	idx := 0
	len := len(hugePageSizeUnitList) - 1
	for size%1024 == 0 && idx < len {
		size /= 1024
		idx++
	}
	if size > 1024 && idx < len {
		return "", fmt.Errorf("size: %d%s must be guaranteed to divisible into the largest units", size, hugePageSizeUnitList[idx])
	}
	return fmt.Sprintf("%d%s", size, hugePageSizeUnitList[idx]), nil
}

// IsHugePageMedium returns true if the volume medium is in 'HugePages[-size]' format
func IsHugePageMedium(medium v1.StorageMedium) bool {
	if medium == v1.StorageMediumHugePages {
		return true
	}
	return strings.HasPrefix(string(medium), string(v1.StorageMediumHugePagesPrefix))
}

// HugePageSizeFromMedium returns the page size for the specified huge page medium.
// If the specified input is not a valid huge page medium an error is returned.
func HugePageSizeFromMedium(medium v1.StorageMedium) (resource.Quantity, error) {
	if !IsHugePageMedium(medium) {
		return resource.Quantity{}, fmt.Errorf("medium: %s is not a hugepage medium", medium)
	}
	if medium == v1.StorageMediumHugePages {
		return resource.Quantity{}, fmt.Errorf("medium: %s doesn't have size information", medium)
	}
	pageSize := strings.TrimPrefix(string(medium), string(v1.StorageMediumHugePagesPrefix))
	return resource.ParseQuantity(pageSize)
}

// IsOvercommitAllowed returns true if the resource is in the default
// namespace and is not hugepages.
func IsOvercommitAllowed(name v1.ResourceName) bool {
	return IsNativeResource(name) &&
		!IsHugePageResourceName(name)
}

// IsAttachableVolumeResourceName returns true when the resource name is prefixed in attachable volume
func IsAttachableVolumeResourceName(name v1.ResourceName) bool {
	return strings.HasPrefix(string(name), v1.ResourceAttachableVolumesPrefix)
}

// IsServiceIPSet aims to check if the service's ClusterIP is set or not
// the objective is not to perform validation here
func IsServiceIPSet(service *v1.Service) bool {
	return service.Spec.ClusterIP != v1.ClusterIPNone && service.Spec.ClusterIP != ""
}

// GetAccessModesAsString returns a string representation of an array of access modes.
// modes, when present, are always in the same order: RWO,ROX,RWX,RWOP.
func GetAccessModesAsString(modes []v1.PersistentVolumeAccessMode) string {
	modes = removeDuplicateAccessModes(modes)
	modesStr := []string{}
	if ContainsAccessMode(modes, v1.ReadWriteOnce) {
		modesStr = append(modesStr, "RWO")
	}
	if ContainsAccessMode(modes, v1.ReadOnlyMany) {
		modesStr = append(modesStr, "ROX")
	}
	if ContainsAccessMode(modes, v1.ReadWriteMany) {
		modesStr = append(modesStr, "RWX")
	}
	if ContainsAccessMode(modes, v1.ReadWriteOncePod) {
		modesStr = append(modesStr, "RWOP")
	}
	return strings.Join(modesStr, ",")
}

// GetAccessModesFromString returns an array of AccessModes from a string created by GetAccessModesAsString
func GetAccessModesFromString(modes string) []v1.PersistentVolumeAccessMode {
	strmodes := strings.Split(modes, ",")
	accessModes := []v1.PersistentVolumeAccessMode{}
	for _, s := range strmodes {
		s = strings.Trim(s, " ")
		switch {
		case s == "RWO":
			accessModes = append(accessModes, v1.ReadWriteOnce)
		case s == "ROX":
			accessModes = append(accessModes, v1.ReadOnlyMany)
		case s == "RWX":
			accessModes = append(accessModes, v1.ReadWriteMany)
		case s == "RWOP":
			accessModes = append(accessModes, v1.ReadWriteOncePod)
		}
	}
	return accessModes
}

// removeDuplicateAccessModes returns an array of access modes without any duplicates
func removeDuplicateAccessModes(modes []v1.PersistentVolumeAccessMode) []v1.PersistentVolumeAccessMode {
	accessModes := []v1.PersistentVolumeAccessMode{}
	for _, m := range modes {
		if !ContainsAccessMode(accessModes, m) {
			accessModes = append(accessModes, m)
		}
	}
	return accessModes
}

func ContainsAccessMode(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// NodeSelectorRequirementKeysExistInNodeSelectorTerms checks if a NodeSelectorTerm with key is already specified in terms
func NodeSelectorRequirementKeysExistInNodeSelectorTerms(reqs []v1.NodeSelectorRequirement, terms []v1.NodeSelectorTerm) bool {
	for _, req := range reqs {
		for _, term := range terms {
			for _, r := range term.MatchExpressions {
				if r.Key == req.Key {
					return true
				}
			}
		}
	}
	return false
}

// TopologySelectorRequirementsAsSelector converts the []TopologySelectorLabelRequirement api type into a struct
// that implements labels.Selector.
func TopologySelectorRequirementsAsSelector(tsm []v1.TopologySelectorLabelRequirement) (labels.Selector, error) {
	if len(tsm) == 0 {
		return labels.Nothing(), nil
	}

	selector := labels.NewSelector()
	for _, expr := range tsm {
		r, err := labels.NewRequirement(expr.Key, selection.In, expr.Values)
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}

	return selector, nil
}

// MatchTopologySelectorTerms checks whether given labels match topology selector terms in ORed;
// nil or empty term matches no objects; while empty term list matches all objects.
func MatchTopologySelectorTerms(topologySelectorTerms []v1.TopologySelectorTerm, lbls labels.Set) bool {
	if len(topologySelectorTerms) == 0 {
		// empty term list matches all objects
		return true
	}

	for _, req := range topologySelectorTerms {
		// nil or empty term selects no objects
		if len(req.MatchLabelExpressions) == 0 {
			continue
		}

		labelSelector, err := TopologySelectorRequirementsAsSelector(req.MatchLabelExpressions)
		if err != nil || !labelSelector.Matches(lbls) {
			continue
		}

		return true
	}

	return false
}

// AddOrUpdateTolerationInPodSpec tries to add a toleration to the toleration list in PodSpec.
// Returns true if something was updated, false otherwise.
func AddOrUpdateTolerationInPodSpec(spec *v1.PodSpec, toleration *v1.Toleration) bool {
	podTolerations := spec.Tolerations

	var newTolerations []v1.Toleration
	updated := false
	for i := range podTolerations {
		if toleration.MatchToleration(&podTolerations[i]) {
			if helper.Semantic.DeepEqual(toleration, podTolerations[i]) {
				return false
			}
			newTolerations = append(newTolerations, *toleration)
			updated = true
			continue
		}

		newTolerations = append(newTolerations, podTolerations[i])
	}

	if !updated {
		newTolerations = append(newTolerations, *toleration)
	}

	spec.Tolerations = newTolerations
	return true
}

// GetMatchingTolerations returns true and list of Tolerations matching all Taints if all are tolerated, or false otherwise.
func GetMatchingTolerations(taints []v1.Taint, tolerations []v1.Toleration) (bool, []v1.Toleration) {
	if len(taints) == 0 {
		return true, []v1.Toleration{}
	}
	if len(tolerations) == 0 && len(taints) > 0 {
		return false, []v1.Toleration{}
	}
	result := []v1.Toleration{}
	for i := range taints {
		tolerated := false
		for j := range tolerations {
			if tolerations[j].ToleratesTaint(&taints[i]) {
				result = append(result, tolerations[j])
				tolerated = true
				break
			}
		}
		if !tolerated {
			return false, []v1.Toleration{}
		}
	}
	return true, result
}

// ScopedResourceSelectorRequirementsAsSelector converts the ScopedResourceSelectorRequirement api type into a struct that implements
// labels.Selector.
func ScopedResourceSelectorRequirementsAsSelector(ssr v1.ScopedResourceSelectorRequirement) (labels.Selector, error) {
	selector := labels.NewSelector()
	var op selection.Operator
	switch ssr.Operator {
	case v1.ScopeSelectorOpIn:
		op = selection.In
	case v1.ScopeSelectorOpNotIn:
		op = selection.NotIn
	case v1.ScopeSelectorOpExists:
		op = selection.Exists
	case v1.ScopeSelectorOpDoesNotExist:
		op = selection.DoesNotExist
	default:
		return nil, fmt.Errorf("%q is not a valid scope selector operator", ssr.Operator)
	}
	r, err := labels.NewRequirement(string(ssr.ScopeName), op, ssr.Values)
	if err != nil {
		return nil, err
	}
	selector = selector.Add(*r)
	return selector, nil
}
