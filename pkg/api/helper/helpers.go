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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
)

// NonConvertibleFields iterates over the provided map and filters out all but
// any keys with the "non-convertible.kubernetes.io" prefix.
func NonConvertibleFields(annotations map[string]string) map[string]string {
	nonConvertibleKeys := map[string]string{}
	for key, value := range annotations {
		if strings.HasPrefix(key, api.NonConvertibleAnnotationPrefix) {
			nonConvertibleKeys[key] = value
		}
	}
	return nonConvertibleKeys
}

// Semantic can do semantic deep equality checks for api objects.
// Example: apiequality.Semantic.DeepEqual(aPod, aPodWithNonNilButEmptyMaps) == true
var Semantic = conversion.EqualitiesOrDie(
	func(a, b resource.Quantity) bool {
		// Ignore formatting, only care that numeric value stayed the same.
		// TODO: if we decide it's important, it should be safe to start comparing the format.
		//
		// Uninitialized quantities are equivalent to 0 quantities.
		return a.Cmp(b) == 0
	},
	func(a, b metav1.MicroTime) bool {
		return a.UTC() == b.UTC()
	},
	func(a, b metav1.Time) bool {
		return a.UTC() == b.UTC()
	},
	func(a, b labels.Selector) bool {
		return a.String() == b.String()
	},
	func(a, b fields.Selector) bool {
		return a.String() == b.String()
	},
)

var standardResourceQuotaScopes = sets.NewString(
	string(api.ResourceQuotaScopeTerminating),
	string(api.ResourceQuotaScopeNotTerminating),
	string(api.ResourceQuotaScopeBestEffort),
	string(api.ResourceQuotaScopeNotBestEffort),
)

// IsStandardResourceQuotaScope returns true if the scope is a standard value
func IsStandardResourceQuotaScope(str string) bool {
	return standardResourceQuotaScopes.Has(str)
}

var podObjectCountQuotaResources = sets.NewString(
	string(api.ResourcePods),
)

var podComputeQuotaResources = sets.NewString(
	string(api.ResourceCPU),
	string(api.ResourceMemory),
	string(api.ResourceLimitsCPU),
	string(api.ResourceLimitsMemory),
	string(api.ResourceRequestsCPU),
	string(api.ResourceRequestsMemory),
)

// IsResourceQuotaScopeValidForResource returns true if the resource applies to the specified scope
func IsResourceQuotaScopeValidForResource(scope api.ResourceQuotaScope, resource string) bool {
	switch scope {
	case api.ResourceQuotaScopeTerminating, api.ResourceQuotaScopeNotTerminating, api.ResourceQuotaScopeNotBestEffort:
		return podObjectCountQuotaResources.Has(resource) || podComputeQuotaResources.Has(resource)
	case api.ResourceQuotaScopeBestEffort:
		return podObjectCountQuotaResources.Has(resource)
	default:
		return true
	}
}

var standardContainerResources = sets.NewString(
	string(api.ResourceCPU),
	string(api.ResourceMemory),
)

// IsStandardContainerResourceName returns true if the container can make a resource request
// for the specified resource
func IsStandardContainerResourceName(str string) bool {
	return standardContainerResources.Has(str)
}

// IsExtendedResourceName returns true if the resource name is not in the
// default namespace, or it has the opaque integer resource prefix.
func IsExtendedResourceName(name api.ResourceName) bool {
	// TODO: Remove OIR part following deprecation.
	return !IsDefaultNamespaceResource(name) || IsOpaqueIntResourceName(name)
}

// IsDefaultNamespaceResource returns true if the resource name is in the
// *kubernetes.io/ namespace. Partially-qualified (unprefixed) names are
// implicitly in the kubernetes.io/ namespace.
func IsDefaultNamespaceResource(name api.ResourceName) bool {
	return !strings.Contains(string(name), "/") ||
		strings.Contains(string(name), api.ResourceDefaultNamespacePrefix)
}

// IsOpaqueIntResourceName returns true if the resource name has the opaque
// integer resource prefix.
func IsOpaqueIntResourceName(name api.ResourceName) bool {
	return strings.HasPrefix(string(name), api.ResourceOpaqueIntPrefix)
}

// OpaqueIntResourceName returns a ResourceName with the canonical opaque
// integer prefix prepended. If the argument already has the prefix, it is
// returned unmodified.
func OpaqueIntResourceName(name string) api.ResourceName {
	if IsOpaqueIntResourceName(api.ResourceName(name)) {
		return api.ResourceName(name)
	}
	return api.ResourceName(fmt.Sprintf("%s%s", api.ResourceOpaqueIntPrefix, name))
}

var overcommitBlacklist = sets.NewString(string(api.ResourceNvidiaGPU))

// IsOvercommitAllowed returns true if the resource is in the default
// namespace and not blacklisted.
func IsOvercommitAllowed(name api.ResourceName) bool {
	return IsDefaultNamespaceResource(name) &&
		!overcommitBlacklist.Has(string(name))
}

var standardLimitRangeTypes = sets.NewString(
	string(api.LimitTypePod),
	string(api.LimitTypeContainer),
	string(api.LimitTypePersistentVolumeClaim),
)

// IsStandardLimitRangeType returns true if the type is Pod or Container
func IsStandardLimitRangeType(str string) bool {
	return standardLimitRangeTypes.Has(str)
}

var standardQuotaResources = sets.NewString(
	string(api.ResourceCPU),
	string(api.ResourceMemory),
	string(api.ResourceRequestsCPU),
	string(api.ResourceRequestsMemory),
	string(api.ResourceRequestsStorage),
	string(api.ResourceLimitsCPU),
	string(api.ResourceLimitsMemory),
	string(api.ResourcePods),
	string(api.ResourceQuotas),
	string(api.ResourceServices),
	string(api.ResourceReplicationControllers),
	string(api.ResourceSecrets),
	string(api.ResourcePersistentVolumeClaims),
	string(api.ResourceConfigMaps),
	string(api.ResourceServicesNodePorts),
	string(api.ResourceServicesLoadBalancers),
)

// IsStandardQuotaResourceName returns true if the resource is known to
// the quota tracking system
func IsStandardQuotaResourceName(str string) bool {
	return standardQuotaResources.Has(str)
}

var standardResources = sets.NewString(
	string(api.ResourceCPU),
	string(api.ResourceMemory),
	string(api.ResourceRequestsCPU),
	string(api.ResourceRequestsMemory),
	string(api.ResourceLimitsCPU),
	string(api.ResourceLimitsMemory),
	string(api.ResourcePods),
	string(api.ResourceQuotas),
	string(api.ResourceServices),
	string(api.ResourceReplicationControllers),
	string(api.ResourceSecrets),
	string(api.ResourceConfigMaps),
	string(api.ResourcePersistentVolumeClaims),
	string(api.ResourceStorage),
	string(api.ResourceRequestsStorage),
)

// IsStandardResourceName returns true if the resource is known to the system
func IsStandardResourceName(str string) bool {
	return standardResources.Has(str)
}

var integerResources = sets.NewString(
	string(api.ResourcePods),
	string(api.ResourceQuotas),
	string(api.ResourceServices),
	string(api.ResourceReplicationControllers),
	string(api.ResourceSecrets),
	string(api.ResourceConfigMaps),
	string(api.ResourcePersistentVolumeClaims),
	string(api.ResourceServicesNodePorts),
	string(api.ResourceServicesLoadBalancers),
)

// IsIntegerResourceName returns true if the resource is measured in integer values
func IsIntegerResourceName(str string) bool {
	return integerResources.Has(str) || IsExtendedResourceName(api.ResourceName(str))
}

// this function aims to check if the service's ClusterIP is set or not
// the objective is not to perform validation here
func IsServiceIPSet(service *api.Service) bool {
	return service.Spec.ClusterIP != api.ClusterIPNone && service.Spec.ClusterIP != ""
}

// this function aims to check if the service's cluster IP is requested or not
func IsServiceIPRequested(service *api.Service) bool {
	// ExternalName services are CNAME aliases to external ones. Ignore the IP.
	if service.Spec.Type == api.ServiceTypeExternalName {
		return false
	}
	return service.Spec.ClusterIP == ""
}

var standardFinalizers = sets.NewString(
	string(api.FinalizerKubernetes),
	metav1.FinalizerOrphanDependents,
	metav1.FinalizerDeleteDependents,
)

// HasAnnotation returns a bool if passed in annotation exists
func HasAnnotation(obj api.ObjectMeta, ann string) bool {
	_, found := obj.Annotations[ann]
	return found
}

// SetMetaDataAnnotation sets the annotation and value
func SetMetaDataAnnotation(obj *api.ObjectMeta, ann string, value string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	obj.Annotations[ann] = value
}

func IsStandardFinalizerName(str string) bool {
	return standardFinalizers.Has(str)
}

// AddToNodeAddresses appends the NodeAddresses to the passed-by-pointer slice,
// only if they do not already exist
func AddToNodeAddresses(addresses *[]api.NodeAddress, addAddresses ...api.NodeAddress) {
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
func LoadBalancerStatusEqual(l, r *api.LoadBalancerStatus) bool {
	return ingressSliceEqual(l.Ingress, r.Ingress)
}

func ingressSliceEqual(lhs, rhs []api.LoadBalancerIngress) bool {
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

func ingressEqual(lhs, rhs *api.LoadBalancerIngress) bool {
	if lhs.IP != rhs.IP {
		return false
	}
	if lhs.Hostname != rhs.Hostname {
		return false
	}
	return true
}

// TODO: make method on LoadBalancerStatus?
func LoadBalancerStatusDeepCopy(lb *api.LoadBalancerStatus) *api.LoadBalancerStatus {
	c := &api.LoadBalancerStatus{}
	c.Ingress = make([]api.LoadBalancerIngress, len(lb.Ingress))
	for i := range lb.Ingress {
		c.Ingress[i] = lb.Ingress[i]
	}
	return c
}

// GetAccessModesAsString returns a string representation of an array of access modes.
// modes, when present, are always in the same order: RWO,ROX,RWX.
func GetAccessModesAsString(modes []api.PersistentVolumeAccessMode) string {
	modes = removeDuplicateAccessModes(modes)
	modesStr := []string{}
	if containsAccessMode(modes, api.ReadWriteOnce) {
		modesStr = append(modesStr, "RWO")
	}
	if containsAccessMode(modes, api.ReadOnlyMany) {
		modesStr = append(modesStr, "ROX")
	}
	if containsAccessMode(modes, api.ReadWriteMany) {
		modesStr = append(modesStr, "RWX")
	}
	return strings.Join(modesStr, ",")
}

// GetAccessModesAsString returns an array of AccessModes from a string created by GetAccessModesAsString
func GetAccessModesFromString(modes string) []api.PersistentVolumeAccessMode {
	strmodes := strings.Split(modes, ",")
	accessModes := []api.PersistentVolumeAccessMode{}
	for _, s := range strmodes {
		s = strings.Trim(s, " ")
		switch {
		case s == "RWO":
			accessModes = append(accessModes, api.ReadWriteOnce)
		case s == "ROX":
			accessModes = append(accessModes, api.ReadOnlyMany)
		case s == "RWX":
			accessModes = append(accessModes, api.ReadWriteMany)
		}
	}
	return accessModes
}

// removeDuplicateAccessModes returns an array of access modes without any duplicates
func removeDuplicateAccessModes(modes []api.PersistentVolumeAccessMode) []api.PersistentVolumeAccessMode {
	accessModes := []api.PersistentVolumeAccessMode{}
	for _, m := range modes {
		if !containsAccessMode(accessModes, m) {
			accessModes = append(accessModes, m)
		}
	}
	return accessModes
}

func containsAccessMode(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// NodeSelectorRequirementsAsSelector converts the []NodeSelectorRequirement api type into a struct that implements
// labels.Selector.
func NodeSelectorRequirementsAsSelector(nsm []api.NodeSelectorRequirement) (labels.Selector, error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	selector := labels.NewSelector()
	for _, expr := range nsm {
		var op selection.Operator
		switch expr.Operator {
		case api.NodeSelectorOpIn:
			op = selection.In
		case api.NodeSelectorOpNotIn:
			op = selection.NotIn
		case api.NodeSelectorOpExists:
			op = selection.Exists
		case api.NodeSelectorOpDoesNotExist:
			op = selection.DoesNotExist
		case api.NodeSelectorOpGt:
			op = selection.GreaterThan
		case api.NodeSelectorOpLt:
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

// GetTolerationsFromPodAnnotations gets the json serialized tolerations data from Pod.Annotations
// and converts it to the []Toleration type in api.
func GetTolerationsFromPodAnnotations(annotations map[string]string) ([]api.Toleration, error) {
	var tolerations []api.Toleration
	if len(annotations) > 0 && annotations[api.TolerationsAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[api.TolerationsAnnotationKey]), &tolerations)
		if err != nil {
			return tolerations, err
		}
	}
	return tolerations, nil
}

// AddOrUpdateTolerationInPod tries to add a toleration to the pod's toleration list.
// Returns true if something was updated, false otherwise.
func AddOrUpdateTolerationInPod(pod *api.Pod, toleration *api.Toleration) bool {
	podTolerations := pod.Spec.Tolerations

	var newTolerations []api.Toleration
	updated := false
	for i := range podTolerations {
		if toleration.MatchToleration(&podTolerations[i]) {
			if Semantic.DeepEqual(toleration, podTolerations[i]) {
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

	pod.Spec.Tolerations = newTolerations
	return true
}

// TolerationToleratesTaint checks if the toleration tolerates the taint.
func TolerationToleratesTaint(toleration *api.Toleration, taint *api.Taint) bool {
	if len(toleration.Effect) != 0 && toleration.Effect != taint.Effect {
		return false
	}

	if toleration.Key != taint.Key {
		return false
	}
	// TODO: Use proper defaulting when Toleration becomes a field of PodSpec
	if (len(toleration.Operator) == 0 || toleration.Operator == api.TolerationOpEqual) && toleration.Value == taint.Value {
		return true
	}
	if toleration.Operator == api.TolerationOpExists {
		return true
	}
	return false
}

// TaintToleratedByTolerations checks if taint is tolerated by any of the tolerations.
func TaintToleratedByTolerations(taint *api.Taint, tolerations []api.Toleration) bool {
	tolerated := false
	for i := range tolerations {
		if TolerationToleratesTaint(&tolerations[i], taint) {
			tolerated = true
			break
		}
	}
	return tolerated
}

// GetTaintsFromNodeAnnotations gets the json serialized taints data from Pod.Annotations
// and converts it to the []Taint type in api.
func GetTaintsFromNodeAnnotations(annotations map[string]string) ([]api.Taint, error) {
	var taints []api.Taint
	if len(annotations) > 0 && annotations[api.TaintsAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[api.TaintsAnnotationKey]), &taints)
		if err != nil {
			return []api.Taint{}, err
		}
	}
	return taints, nil
}

// SysctlsFromPodAnnotations parses the sysctl annotations into a slice of safe Sysctls
// and a slice of unsafe Sysctls. This is only a convenience wrapper around
// SysctlsFromPodAnnotation.
func SysctlsFromPodAnnotations(a map[string]string) ([]api.Sysctl, []api.Sysctl, error) {
	safe, err := SysctlsFromPodAnnotation(a[api.SysctlsPodAnnotationKey])
	if err != nil {
		return nil, nil, err
	}
	unsafe, err := SysctlsFromPodAnnotation(a[api.UnsafeSysctlsPodAnnotationKey])
	if err != nil {
		return nil, nil, err
	}

	return safe, unsafe, nil
}

// SysctlsFromPodAnnotation parses an annotation value into a slice of Sysctls.
func SysctlsFromPodAnnotation(annotation string) ([]api.Sysctl, error) {
	if len(annotation) == 0 {
		return nil, nil
	}

	kvs := strings.Split(annotation, ",")
	sysctls := make([]api.Sysctl, len(kvs))
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
func PodAnnotationsFromSysctls(sysctls []api.Sysctl) string {
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
func GetPersistentVolumeClass(volume *api.PersistentVolume) string {
	// Use beta annotation first
	if class, found := volume.Annotations[api.BetaStorageClassAnnotation]; found {
		return class
	}

	return volume.Spec.StorageClassName
}

// GetPersistentVolumeClaimClass returns StorageClassName. If no storage class was
// requested, it returns "".
func GetPersistentVolumeClaimClass(claim *api.PersistentVolumeClaim) string {
	// Use beta annotation first
	if class, found := claim.Annotations[api.BetaStorageClassAnnotation]; found {
		return class
	}

	if claim.Spec.StorageClassName != nil {
		return *claim.Spec.StorageClassName
	}

	return ""
}

// PersistentVolumeClaimHasClass returns true if given claim has set StorageClassName field.
func PersistentVolumeClaimHasClass(claim *api.PersistentVolumeClaim) bool {
	// Use beta annotation first
	if _, found := claim.Annotations[api.BetaStorageClassAnnotation]; found {
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
func GetStorageNodeAffinityFromAnnotation(annotations map[string]string) (*api.NodeAffinity, error) {
	if len(annotations) > 0 && annotations[api.AlphaStorageNodeAffinityAnnotation] != "" {
		var affinity api.NodeAffinity
		err := json.Unmarshal([]byte(annotations[api.AlphaStorageNodeAffinityAnnotation]), &affinity)
		if err != nil {
			return nil, err
		}
		return &affinity, nil
	}
	return nil, nil
}

// Converts NodeAffinity type to Alpha annotation for use in PersistentVolumes
// TODO: update when storage node affinity graduates to beta
func StorageNodeAffinityToAlphaAnnotation(annotations map[string]string, affinity *api.NodeAffinity) error {
	if affinity == nil {
		return nil
	}

	json, err := json.Marshal(*affinity)
	if err != nil {
		return err
	}
	annotations[api.AlphaStorageNodeAffinityAnnotation] = string(json)
	return nil
}
