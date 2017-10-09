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
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/helper"
)

// IsExtendedResourceName returns true if the resource name is not in the
// default namespace, or it has the opaque integer resource prefix.
func IsExtendedResourceName(name v1.ResourceName) bool {
	// TODO: Remove OIR part following deprecation.
	return !IsDefaultNamespaceResource(name) || IsOpaqueIntResourceName(name)
}

// IsDefaultNamespaceResource returns true if the resource name is in the
// *kubernetes.io/ namespace. Partially-qualified (unprefixed) names are
// implicitly in the kubernetes.io/ namespace.
func IsDefaultNamespaceResource(name v1.ResourceName) bool {
	return !strings.Contains(string(name), "/") ||
		strings.Contains(string(name), v1.ResourceDefaultNamespacePrefix)

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
		return resource.Quantity{}, fmt.Errorf("resource name: %s is not valid hugepage name", name)
	}
	pageSize := strings.TrimPrefix(string(name), v1.ResourceHugePagesPrefix)
	return resource.ParseQuantity(pageSize)
}

// IsOpaqueIntResourceName returns true if the resource name has the opaque
// integer resource prefix.
func IsOpaqueIntResourceName(name v1.ResourceName) bool {
	return strings.HasPrefix(string(name), v1.ResourceOpaqueIntPrefix)
}

// OpaqueIntResourceName returns a ResourceName with the canonical opaque
// integer prefix prepended. If the argument already has the prefix, it is
// returned unmodified.
func OpaqueIntResourceName(name string) v1.ResourceName {
	if IsOpaqueIntResourceName(v1.ResourceName(name)) {
		return v1.ResourceName(name)
	}
	return v1.ResourceName(fmt.Sprintf("%s%s", v1.ResourceOpaqueIntPrefix, name))
}

var overcommitBlacklist = sets.NewString(string(v1.ResourceNvidiaGPU))

// IsOvercommitAllowed returns true if the resource is in the default
// namespace and not blacklisted and is not hugepages.
func IsOvercommitAllowed(name v1.ResourceName) bool {
	return IsDefaultNamespaceResource(name) &&
		!IsHugePageResourceName(name) &&
		!overcommitBlacklist.Has(string(name))
}

// Extended and Hugepages resources
func IsScalarResourceName(name v1.ResourceName) bool {
	return IsExtendedResourceName(name) || IsHugePageResourceName(name)
}

// this function aims to check if the service's ClusterIP is set or not
// the objective is not to perform validation here
func IsServiceIPSet(service *v1.Service) bool {
	return service.Spec.ClusterIP != v1.ClusterIPNone && service.Spec.ClusterIP != ""
}

// this function aims to check if the service's cluster IP is requested or not
func IsServiceIPRequested(service *v1.Service) bool {
	// ExternalName services are CNAME aliases to external ones. Ignore the IP.
	if service.Spec.Type == v1.ServiceTypeExternalName {
		return false
	}
	return service.Spec.ClusterIP == ""
}

// AddToNodeAddresses appends the NodeAddresses to the passed-by-pointer slice,
// only if they do not already exist
func AddToNodeAddresses(addresses *[]v1.NodeAddress, addAddresses ...v1.NodeAddress) {
	for _, add := range addAddresses {
		exists := false
		for _, existing := range *addresses {
			if existing.Address == add.Address && existing.Type == add.Type {
				exists = true
				break
			}
		}
		if !exists {
			*addresses = append(*addresses, add)
		}
	}
}

// TODO: make method on LoadBalancerStatus?
func LoadBalancerStatusEqual(l, r *v1.LoadBalancerStatus) bool {
	return ingressSliceEqual(l.Ingress, r.Ingress)
}

func ingressSliceEqual(lhs, rhs []v1.LoadBalancerIngress) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	for i := range lhs {
		if !ingressEqual(&lhs[i], &rhs[i]) {
			return false
		}
	}
	return true
}

func ingressEqual(lhs, rhs *v1.LoadBalancerIngress) bool {
	if lhs.IP != rhs.IP {
		return false
	}
	if lhs.Hostname != rhs.Hostname {
		return false
	}
	return true
}

// TODO: make method on LoadBalancerStatus?
func LoadBalancerStatusDeepCopy(lb *v1.LoadBalancerStatus) *v1.LoadBalancerStatus {
	c := &v1.LoadBalancerStatus{}
	c.Ingress = make([]v1.LoadBalancerIngress, len(lb.Ingress))
	for i := range lb.Ingress {
		c.Ingress[i] = lb.Ingress[i]
	}
	return c
}

// GetAccessModesAsString returns a string representation of an array of access modes.
// modes, when present, are always in the same order: RWO,ROX,RWX.
func GetAccessModesAsString(modes []v1.PersistentVolumeAccessMode) string {
	modes = removeDuplicateAccessModes(modes)
	modesStr := []string{}
	if containsAccessMode(modes, v1.ReadWriteOnce) {
		modesStr = append(modesStr, "RWO")
	}
	if containsAccessMode(modes, v1.ReadOnlyMany) {
		modesStr = append(modesStr, "ROX")
	}
	if containsAccessMode(modes, v1.ReadWriteMany) {
		modesStr = append(modesStr, "RWX")
	}
	return strings.Join(modesStr, ",")
}

// GetAccessModesAsString returns an array of AccessModes from a string created by GetAccessModesAsString
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
		}
	}
	return accessModes
}

// removeDuplicateAccessModes returns an array of access modes without any duplicates
func removeDuplicateAccessModes(modes []v1.PersistentVolumeAccessMode) []v1.PersistentVolumeAccessMode {
	accessModes := []v1.PersistentVolumeAccessMode{}
	for _, m := range modes {
		if !containsAccessMode(accessModes, m) {
			accessModes = append(accessModes, m)
		}
	}
	return accessModes
}

func containsAccessMode(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// NodeSelectorRequirementsAsSelector converts the []NodeSelectorRequirement api type into a struct that implements
// labels.Selector.
func NodeSelectorRequirementsAsSelector(nsm []v1.NodeSelectorRequirement) (labels.Selector, error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	selector := labels.NewSelector()
	for _, expr := range nsm {
		var op selection.Operator
		switch expr.Operator {
		case v1.NodeSelectorOpIn:
			op = selection.In
		case v1.NodeSelectorOpNotIn:
			op = selection.NotIn
		case v1.NodeSelectorOpExists:
			op = selection.Exists
		case v1.NodeSelectorOpDoesNotExist:
			op = selection.DoesNotExist
		case v1.NodeSelectorOpGt:
			op = selection.GreaterThan
		case v1.NodeSelectorOpLt:
			op = selection.LessThan
		default:
			return nil, fmt.Errorf("%q is not a valid node selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, expr.Values)
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	return selector, nil
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

// AddOrUpdateTolerationInPod tries to add a toleration to the pod's toleration list.
// Returns true if something was updated, false otherwise.
func AddOrUpdateTolerationInPod(pod *v1.Pod, toleration *v1.Toleration) bool {
	return AddOrUpdateTolerationInPodSpec(&pod.Spec, toleration)
}

// TolerationsTolerateTaint checks if taint is tolerated by any of the tolerations.
func TolerationsTolerateTaint(tolerations []v1.Toleration, taint *v1.Taint) bool {
	for i := range tolerations {
		if tolerations[i].ToleratesTaint(taint) {
			return true
		}
	}
	return false
}

type taintsFilterFunc func(*v1.Taint) bool

// TolerationsTolerateTaintsWithFilter checks if given tolerations tolerates
// all the taints that apply to the filter in given taint list.
func TolerationsTolerateTaintsWithFilter(tolerations []v1.Toleration, taints []v1.Taint, applyFilter taintsFilterFunc) bool {
	if len(taints) == 0 {
		return true
	}

	for i := range taints {
		if applyFilter != nil && !applyFilter(&taints[i]) {
			continue
		}

		if !TolerationsTolerateTaint(tolerations, &taints[i]) {
			return false
		}
	}

	return true
}

// Returns true and list of Tolerations matching all Taints if all are tolerated, or false otherwise.
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

func GetAvoidPodsFromNodeAnnotations(annotations map[string]string) (v1.AvoidPods, error) {
	var avoidPods v1.AvoidPods
	if len(annotations) > 0 && annotations[v1.PreferAvoidPodsAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[v1.PreferAvoidPodsAnnotationKey]), &avoidPods)
		if err != nil {
			return avoidPods, err
		}
	}
	return avoidPods, nil
}

// SysctlsFromPodAnnotations parses the sysctl annotations into a slice of safe Sysctls
// and a slice of unsafe Sysctls. This is only a convenience wrapper around
// SysctlsFromPodAnnotation.
func SysctlsFromPodAnnotations(a map[string]string) ([]v1.Sysctl, []v1.Sysctl, error) {
	safe, err := SysctlsFromPodAnnotation(a[v1.SysctlsPodAnnotationKey])
	if err != nil {
		return nil, nil, err
	}
	unsafe, err := SysctlsFromPodAnnotation(a[v1.UnsafeSysctlsPodAnnotationKey])
	if err != nil {
		return nil, nil, err
	}

	return safe, unsafe, nil
}

// SysctlsFromPodAnnotation parses an annotation value into a slice of Sysctls.
func SysctlsFromPodAnnotation(annotation string) ([]v1.Sysctl, error) {
	if len(annotation) == 0 {
		return nil, nil
	}

	kvs := strings.Split(annotation, ",")
	sysctls := make([]v1.Sysctl, len(kvs))
	for i, kv := range kvs {
		cs := strings.Split(kv, "=")
		if len(cs) != 2 || len(cs[0]) == 0 {
			return nil, fmt.Errorf("sysctl %q not of the format sysctl_name=value", kv)
		}
		sysctls[i].Name = cs[0]
		sysctls[i].Value = cs[1]
	}
	return sysctls, nil
}

// PodAnnotationsFromSysctls creates an annotation value for a slice of Sysctls.
func PodAnnotationsFromSysctls(sysctls []v1.Sysctl) string {
	if len(sysctls) == 0 {
		return ""
	}

	kvs := make([]string, len(sysctls))
	for i := range sysctls {
		kvs[i] = fmt.Sprintf("%s=%s", sysctls[i].Name, sysctls[i].Value)
	}
	return strings.Join(kvs, ",")
}

// GetPersistentVolumeClass returns StorageClassName.
func GetPersistentVolumeClass(volume *v1.PersistentVolume) string {
	// Use beta annotation first
	if class, found := volume.Annotations[v1.BetaStorageClassAnnotation]; found {
		return class
	}

	return volume.Spec.StorageClassName
}

// GetPersistentVolumeClaimClass returns StorageClassName. If no storage class was
// requested, it returns "".
func GetPersistentVolumeClaimClass(claim *v1.PersistentVolumeClaim) string {
	// Use beta annotation first
	if class, found := claim.Annotations[v1.BetaStorageClassAnnotation]; found {
		return class
	}

	if claim.Spec.StorageClassName != nil {
		return *claim.Spec.StorageClassName
	}

	return ""
}

// PersistentVolumeClaimHasClass returns true if given claim has set StorageClassName field.
func PersistentVolumeClaimHasClass(claim *v1.PersistentVolumeClaim) bool {
	// Use beta annotation first
	if _, found := claim.Annotations[v1.BetaStorageClassAnnotation]; found {
		return true
	}

	if claim.Spec.StorageClassName != nil {
		return true
	}

	return false
}

// GetStorageNodeAffinityFromAnnotation gets the json serialized data from PersistentVolume.Annotations
// and converts it to the NodeAffinity type in api.
// TODO: update when storage node affinity graduates to beta
func GetStorageNodeAffinityFromAnnotation(annotations map[string]string) (*v1.NodeAffinity, error) {
	if len(annotations) > 0 && annotations[v1.AlphaStorageNodeAffinityAnnotation] != "" {
		var affinity v1.NodeAffinity
		err := json.Unmarshal([]byte(annotations[v1.AlphaStorageNodeAffinityAnnotation]), &affinity)
		if err != nil {
			return nil, err
		}
		return &affinity, nil
	}
	return nil, nil
}

// Converts NodeAffinity type to Alpha annotation for use in PersistentVolumes
// TODO: update when storage node affinity graduates to beta
func StorageNodeAffinityToAlphaAnnotation(annotations map[string]string, affinity *v1.NodeAffinity) error {
	if affinity == nil {
		return nil
	}

	json, err := json.Marshal(*affinity)
	if err != nil {
		return err
	}
	annotations[v1.AlphaStorageNodeAffinityAnnotation] = string(json)
	return nil
}
