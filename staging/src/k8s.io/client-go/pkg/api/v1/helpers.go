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

package v1

import (
	"encoding/json"
	"fmt"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/pkg/api"
)

// IsOpaqueIntResourceName returns true if the resource name has the opaque
// integer resource prefix.
func IsOpaqueIntResourceName(name ResourceName) bool {
	return strings.HasPrefix(string(name), ResourceOpaqueIntPrefix)
}

// OpaqueIntResourceName returns a ResourceName with the canonical opaque
// integer prefix prepended. If the argument already has the prefix, it is
// returned unmodified.
func OpaqueIntResourceName(name string) ResourceName {
	if IsOpaqueIntResourceName(ResourceName(name)) {
		return ResourceName(name)
	}
	return ResourceName(fmt.Sprintf("%s%s", api.ResourceOpaqueIntPrefix, name))
}

// NewDeleteOptions returns a DeleteOptions indicating the resource should
// be deleted within the specified grace period. Use zero to indicate
// immediate deletion. If you would prefer to use the default grace period,
// use &metav1.DeleteOptions{} directly.
func NewDeleteOptions(grace int64) *DeleteOptions {
	return &DeleteOptions{GracePeriodSeconds: &grace}
}

// NewPreconditionDeleteOptions returns a DeleteOptions with a UID precondition set.
func NewPreconditionDeleteOptions(uid string) *DeleteOptions {
	u := types.UID(uid)
	p := Preconditions{UID: &u}
	return &DeleteOptions{Preconditions: &p}
}

// NewUIDPreconditions returns a Preconditions with UID set.
func NewUIDPreconditions(uid string) *Preconditions {
	u := types.UID(uid)
	return &Preconditions{UID: &u}
}

// this function aims to check if the service's ClusterIP is set or not
// the objective is not to perform validation here
func IsServiceIPSet(service *Service) bool {
	return service.Spec.ClusterIP != ClusterIPNone && service.Spec.ClusterIP != ""
}

// this function aims to check if the service's cluster IP is requested or not
func IsServiceIPRequested(service *Service) bool {
	// ExternalName services are CNAME aliases to external ones. Ignore the IP.
	if service.Spec.Type == ServiceTypeExternalName {
		return false
	}
	return service.Spec.ClusterIP == ""
}

var standardFinalizers = sets.NewString(
	string(FinalizerKubernetes),
	metav1.FinalizerOrphan,
)

func IsStandardFinalizerName(str string) bool {
	return standardFinalizers.Has(str)
}

// AddToNodeAddresses appends the NodeAddresses to the passed-by-pointer slice,
// only if they do not already exist
func AddToNodeAddresses(addresses *[]NodeAddress, addAddresses ...NodeAddress) {
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
func LoadBalancerStatusEqual(l, r *LoadBalancerStatus) bool {
	return ingressSliceEqual(l.Ingress, r.Ingress)
}

func ingressSliceEqual(lhs, rhs []LoadBalancerIngress) bool {
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

func ingressEqual(lhs, rhs *LoadBalancerIngress) bool {
	if lhs.IP != rhs.IP {
		return false
	}
	if lhs.Hostname != rhs.Hostname {
		return false
	}
	return true
}

// TODO: make method on LoadBalancerStatus?
func LoadBalancerStatusDeepCopy(lb *LoadBalancerStatus) *LoadBalancerStatus {
	c := &LoadBalancerStatus{}
	c.Ingress = make([]LoadBalancerIngress, len(lb.Ingress))
	for i := range lb.Ingress {
		c.Ingress[i] = lb.Ingress[i]
	}
	return c
}

// GetAccessModesAsString returns a string representation of an array of access modes.
// modes, when present, are always in the same order: RWO,ROX,RWX.
func GetAccessModesAsString(modes []PersistentVolumeAccessMode) string {
	modes = removeDuplicateAccessModes(modes)
	modesStr := []string{}
	if containsAccessMode(modes, ReadWriteOnce) {
		modesStr = append(modesStr, "RWO")
	}
	if containsAccessMode(modes, ReadOnlyMany) {
		modesStr = append(modesStr, "ROX")
	}
	if containsAccessMode(modes, ReadWriteMany) {
		modesStr = append(modesStr, "RWX")
	}
	return strings.Join(modesStr, ",")
}

// GetAccessModesAsString returns an array of AccessModes from a string created by GetAccessModesAsString
func GetAccessModesFromString(modes string) []PersistentVolumeAccessMode {
	strmodes := strings.Split(modes, ",")
	accessModes := []PersistentVolumeAccessMode{}
	for _, s := range strmodes {
		s = strings.Trim(s, " ")
		switch {
		case s == "RWO":
			accessModes = append(accessModes, ReadWriteOnce)
		case s == "ROX":
			accessModes = append(accessModes, ReadOnlyMany)
		case s == "RWX":
			accessModes = append(accessModes, ReadWriteMany)
		}
	}
	return accessModes
}

// removeDuplicateAccessModes returns an array of access modes without any duplicates
func removeDuplicateAccessModes(modes []PersistentVolumeAccessMode) []PersistentVolumeAccessMode {
	accessModes := []PersistentVolumeAccessMode{}
	for _, m := range modes {
		if !containsAccessMode(accessModes, m) {
			accessModes = append(accessModes, m)
		}
	}
	return accessModes
}

func containsAccessMode(modes []PersistentVolumeAccessMode, mode PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// NodeSelectorRequirementsAsSelector converts the []NodeSelectorRequirement api type into a struct that implements
// labels.Selector.
func NodeSelectorRequirementsAsSelector(nsm []NodeSelectorRequirement) (labels.Selector, error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	selector := labels.NewSelector()
	for _, expr := range nsm {
		var op selection.Operator
		switch expr.Operator {
		case NodeSelectorOpIn:
			op = selection.In
		case NodeSelectorOpNotIn:
			op = selection.NotIn
		case NodeSelectorOpExists:
			op = selection.Exists
		case NodeSelectorOpDoesNotExist:
			op = selection.DoesNotExist
		case NodeSelectorOpGt:
			op = selection.GreaterThan
		case NodeSelectorOpLt:
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

const (
	// TolerationsAnnotationKey represents the key of tolerations data (json serialized)
	// in the Annotations of a Pod.
	TolerationsAnnotationKey string = "scheduler.alpha.kubernetes.io/tolerations"

	// TaintsAnnotationKey represents the key of taints data (json serialized)
	// in the Annotations of a Node.
	TaintsAnnotationKey string = "scheduler.alpha.kubernetes.io/taints"

	// SeccompPodAnnotationKey represents the key of a seccomp profile applied
	// to all containers of a pod.
	SeccompPodAnnotationKey string = "seccomp.security.alpha.kubernetes.io/pod"

	// SeccompContainerAnnotationKeyPrefix represents the key of a seccomp profile applied
	// to one container of a pod.
	SeccompContainerAnnotationKeyPrefix string = "container.seccomp.security.alpha.kubernetes.io/"

	// CreatedByAnnotation represents the key used to store the spec(json)
	// used to create the resource.
	CreatedByAnnotation = "kubernetes.io/created-by"

	// PreferAvoidPodsAnnotationKey represents the key of preferAvoidPods data (json serialized)
	// in the Annotations of a Node.
	PreferAvoidPodsAnnotationKey string = "scheduler.alpha.kubernetes.io/preferAvoidPods"

	// SysctlsPodAnnotationKey represents the key of sysctls which are set for the infrastructure
	// container of a pod. The annotation value is a comma separated list of sysctl_name=value
	// key-value pairs. Only a limited set of whitelisted and isolated sysctls is supported by
	// the kubelet. Pods with other sysctls will fail to launch.
	SysctlsPodAnnotationKey string = "security.alpha.kubernetes.io/sysctls"

	// UnsafeSysctlsPodAnnotationKey represents the key of sysctls which are set for the infrastructure
	// container of a pod. The annotation value is a comma separated list of sysctl_name=value
	// key-value pairs. Unsafe sysctls must be explicitly enabled for a kubelet. They are properly
	// namespaced to a pod or a container, but their isolation is usually unclear or weak. Their use
	// is at-your-own-risk. Pods that attempt to set an unsafe sysctl that is not enabled for a kubelet
	// will fail to launch.
	UnsafeSysctlsPodAnnotationKey string = "security.alpha.kubernetes.io/unsafe-sysctls"

	// ObjectTTLAnnotations represents a suggestion for kubelet for how long it can cache
	// an object (e.g. secret, config map) before fetching it again from apiserver.
	// This annotation can be attached to node.
	ObjectTTLAnnotationKey string = "node.alpha.kubernetes.io/ttl"
)

// GetTolerationsFromPodAnnotations gets the json serialized tolerations data from Pod.Annotations
// and converts it to the []Toleration type in api.
func GetTolerationsFromPodAnnotations(annotations map[string]string) ([]Toleration, error) {
	var tolerations []Toleration
	if len(annotations) > 0 && annotations[TolerationsAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[TolerationsAnnotationKey]), &tolerations)
		if err != nil {
			return tolerations, err
		}
	}
	return tolerations, nil
}

func GetPodTolerations(pod *Pod) ([]Toleration, error) {
	return GetTolerationsFromPodAnnotations(pod.Annotations)
}

// GetTaintsFromNodeAnnotations gets the json serialized taints data from Pod.Annotations
// and converts it to the []Taint type in api.
func GetTaintsFromNodeAnnotations(annotations map[string]string) ([]Taint, error) {
	var taints []Taint
	if len(annotations) > 0 && annotations[TaintsAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[TaintsAnnotationKey]), &taints)
		if err != nil {
			return []Taint{}, err
		}
	}
	return taints, nil
}

func GetNodeTaints(node *Node) ([]Taint, error) {
	return GetTaintsFromNodeAnnotations(node.Annotations)
}

// ToleratesTaint checks if the toleration tolerates the taint.
// The matching follows the rules below:
// (1) Empty toleration.effect means to match all taint effects,
//     otherwise taint effect must equal to toleration.effect.
// (2) If toleration.operator is 'Exists', it means to match all taint values.
// (3) Empty toleration.key means to match all taint keys.
//     If toleration.key is empty, toleration.operator must be 'Exists';
//     this combination means to match all taint values and all taint keys.
func (t *Toleration) ToleratesTaint(taint *Taint) bool {
	if len(t.Effect) > 0 && t.Effect != taint.Effect {
		return false
	}

	if len(t.Key) > 0 && t.Key != taint.Key {
		return false
	}

	// TODO: Use proper defaulting when Toleration becomes a field of PodSpec
	switch t.Operator {
	// empty operator means Equal
	case "", TolerationOpEqual:
		return t.Value == taint.Value
	case TolerationOpExists:
		return true
	default:
		return false
	}
}

// TolerationsTolerateTaint checks if taint is tolerated by any of the tolerations.
func TolerationsTolerateTaint(tolerations []Toleration, taint *Taint) bool {
	for i := range tolerations {
		if tolerations[i].ToleratesTaint(taint) {
			return true
		}
	}
	return false
}

type taintsFilterFunc func(*Taint) bool

// TolerationsTolerateTaintsWithFilter checks if given tolerations tolerates
// all the taints that apply to the filter in given taint list.
func TolerationsTolerateTaintsWithFilter(tolerations []Toleration, taints []Taint, applyFilter taintsFilterFunc) bool {
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

// DeleteTaintsByKey removes all the taints that have the same key to given taintKey
func DeleteTaintsByKey(taints []Taint, taintKey string) ([]Taint, bool) {
	newTaints := []Taint{}
	deleted := false
	for i := range taints {
		if taintKey == taints[i].Key {
			deleted = true
			continue
		}
		newTaints = append(newTaints, taints[i])
	}
	return newTaints, deleted
}

// DeleteTaint removes all the the taints that have the same key and effect to given taintToDelete.
func DeleteTaint(taints []Taint, taintToDelete *Taint) ([]Taint, bool) {
	newTaints := []Taint{}
	deleted := false
	for i := range taints {
		if taintToDelete.MatchTaint(taints[i]) {
			deleted = true
			continue
		}
		newTaints = append(newTaints, taints[i])
	}
	return newTaints, deleted
}

// Returns true and list of Tolerations matching all Taints if all are tolerated, or false otherwise.
func GetMatchingTolerations(taints []Taint, tolerations []Toleration) (bool, []Toleration) {
	if len(tolerations) == 0 && len(taints) > 0 {
		return false, []Toleration{}
	}
	result := []Toleration{}
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
			return false, []Toleration{}
		}
	}
	return true, result
}

// MatchTaint checks if the taint matches taintToMatch. Taints are unique by key:effect,
// if the two taints have same key:effect, regard as they match.
func (t *Taint) MatchTaint(taintToMatch Taint) bool {
	return t.Key == taintToMatch.Key && t.Effect == taintToMatch.Effect
}

// taint.ToString() converts taint struct to string in format key=value:effect or key:effect.
func (t *Taint) ToString() string {
	if len(t.Value) == 0 {
		return fmt.Sprintf("%v:%v", t.Key, t.Effect)
	}
	return fmt.Sprintf("%v=%v:%v", t.Key, t.Value, t.Effect)
}

func GetAvoidPodsFromNodeAnnotations(annotations map[string]string) (AvoidPods, error) {
	var avoidPods AvoidPods
	if len(annotations) > 0 && annotations[PreferAvoidPodsAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[PreferAvoidPodsAnnotationKey]), &avoidPods)
		if err != nil {
			return avoidPods, err
		}
	}
	return avoidPods, nil
}

// SysctlsFromPodAnnotations parses the sysctl annotations into a slice of safe Sysctls
// and a slice of unsafe Sysctls. This is only a convenience wrapper around
// SysctlsFromPodAnnotation.
func SysctlsFromPodAnnotations(a map[string]string) ([]Sysctl, []Sysctl, error) {
	safe, err := SysctlsFromPodAnnotation(a[SysctlsPodAnnotationKey])
	if err != nil {
		return nil, nil, err
	}
	unsafe, err := SysctlsFromPodAnnotation(a[UnsafeSysctlsPodAnnotationKey])
	if err != nil {
		return nil, nil, err
	}

	return safe, unsafe, nil
}

// SysctlsFromPodAnnotation parses an annotation value into a slice of Sysctls.
func SysctlsFromPodAnnotation(annotation string) ([]Sysctl, error) {
	if len(annotation) == 0 {
		return nil, nil
	}

	kvs := strings.Split(annotation, ",")
	sysctls := make([]Sysctl, len(kvs))
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
func PodAnnotationsFromSysctls(sysctls []Sysctl) string {
	if len(sysctls) == 0 {
		return ""
	}

	kvs := make([]string, len(sysctls))
	for i := range sysctls {
		kvs[i] = fmt.Sprintf("%s=%s", sysctls[i].Name, sysctls[i].Value)
	}
	return strings.Join(kvs, ",")
}

type Sysctl struct {
	Name  string `protobuf:"bytes,1,opt,name=name"`
	Value string `protobuf:"bytes,2,opt,name=value"`
}

// NodeResources is an object for conveying resource information about a node.
// see http://releases.k8s.io/HEAD/docs/design/resources.md for more details.
type NodeResources struct {
	// Capacity represents the available resources of a node
	Capacity ResourceList `protobuf:"bytes,1,rep,name=capacity,casttype=ResourceList,castkey=ResourceName"`
}
