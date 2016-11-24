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

	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/fields"
	"k8s.io/client-go/pkg/labels"
	"k8s.io/client-go/pkg/selection"
	"k8s.io/client-go/pkg/types"
	"k8s.io/client-go/pkg/util/sets"
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
// use &api.DeleteOptions{} directly.
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
	FinalizerOrphan,
)

// HasAnnotation returns a bool if passed in annotation exists
func HasAnnotation(obj ObjectMeta, ann string) bool {
	_, found := obj.Annotations[ann]
	return found
}

// SetMetaDataAnnotation sets the annotation and value
func SetMetaDataAnnotation(obj *ObjectMeta, ann string, value string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	obj.Annotations[ann] = value
}

func IsStandardFinalizerName(str string) bool {
	return standardFinalizers.Has(str)
}

// SingleObject returns a ListOptions for watching a single object.
func SingleObject(meta ObjectMeta) ListOptions {
	return ListOptions{
		FieldSelector:   fields.OneTermEqualSelector("metadata.name", meta.Name).String(),
		ResourceVersion: meta.ResourceVersion,
	}
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
	// AffinityAnnotationKey represents the key of affinity data (json serialized)
	// in the Annotations of a Pod.
	AffinityAnnotationKey string = "scheduler.alpha.kubernetes.io/affinity"

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
)

// GetAffinityFromPod gets the json serialized affinity data from Pod.Annotations
// and converts it to the Affinity type in api.
func GetAffinityFromPodAnnotations(annotations map[string]string) (*Affinity, error) {
	if len(annotations) > 0 && annotations[AffinityAnnotationKey] != "" {
		var affinity Affinity
		err := json.Unmarshal([]byte(annotations[AffinityAnnotationKey]), &affinity)
		if err != nil {
			return nil, err
		}
		return &affinity, nil
	}
	return nil, nil
}

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

// TolerationToleratesTaint checks if the toleration tolerates the taint.
func TolerationToleratesTaint(toleration *Toleration, taint *Taint) bool {
	if len(toleration.Effect) != 0 && toleration.Effect != taint.Effect {
		return false
	}

	if toleration.Key != taint.Key {
		return false
	}
	// TODO: Use proper defaulting when Toleration becomes a field of PodSpec
	if (len(toleration.Operator) == 0 || toleration.Operator == TolerationOpEqual) && toleration.Value == taint.Value {
		return true
	}
	if toleration.Operator == TolerationOpExists {
		return true
	}
	return false
}

// TaintToleratedByTolerations checks if taint is tolerated by any of the tolerations.
func TaintToleratedByTolerations(taint *Taint, tolerations []Toleration) bool {
	tolerated := false
	for i := range tolerations {
		if TolerationToleratesTaint(&tolerations[i], taint) {
			tolerated = true
			break
		}
	}
	return tolerated
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
