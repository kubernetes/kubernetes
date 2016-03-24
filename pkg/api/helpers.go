/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/davecgh/go-spew/spew"
)

// Conversion error conveniently packages up errors in conversions.
type ConversionError struct {
	In, Out interface{}
	Message string
}

// Return a helpful string about the error
func (c *ConversionError) Error() string {
	return spew.Sprintf(
		"Conversion error: %s. (in: %v(%+v) out: %v)",
		c.Message, reflect.TypeOf(c.In), c.In, reflect.TypeOf(c.Out),
	)
}

// Semantic can do semantic deep equality checks for api objects.
// Example: api.Semantic.DeepEqual(aPod, aPodWithNonNilButEmptyMaps) == true
var Semantic = conversion.EqualitiesOrDie(
	func(a, b resource.Quantity) bool {
		// Ignore formatting, only care that numeric value stayed the same.
		// TODO: if we decide it's important, it should be safe to start comparing the format.
		//
		// Uninitialized quantities are equivalent to 0 quantities.
		if a.Amount == nil && b.MilliValue() == 0 {
			return true
		}
		if b.Amount == nil && a.MilliValue() == 0 {
			return true
		}
		if a.Amount == nil || b.Amount == nil {
			return false
		}
		return a.Amount.Cmp(b.Amount) == 0
	},
	func(a, b unversioned.Time) bool {
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
	string(ResourceQuotaScopeTerminating),
	string(ResourceQuotaScopeNotTerminating),
	string(ResourceQuotaScopeBestEffort),
	string(ResourceQuotaScopeNotBestEffort),
)

// IsStandardResourceQuotaScope returns true if the scope is a standard value
func IsStandardResourceQuotaScope(str string) bool {
	return standardResourceQuotaScopes.Has(str)
}

var podObjectCountQuotaResources = sets.NewString(
	string(ResourcePods),
)

var podComputeQuotaResources = sets.NewString(
	string(ResourceCPU),
	string(ResourceMemory),
	string(ResourceLimitsCPU),
	string(ResourceLimitsMemory),
	string(ResourceRequestsCPU),
	string(ResourceRequestsMemory),
)

// IsResourceQuotaScopeValidForResource returns true if the resource applies to the specified scope
func IsResourceQuotaScopeValidForResource(scope ResourceQuotaScope, resource string) bool {
	switch scope {
	case ResourceQuotaScopeTerminating, ResourceQuotaScopeNotTerminating, ResourceQuotaScopeNotBestEffort:
		return podObjectCountQuotaResources.Has(resource) || podComputeQuotaResources.Has(resource)
	case ResourceQuotaScopeBestEffort:
		return podObjectCountQuotaResources.Has(resource)
	default:
		return true
	}
}

var standardContainerResources = sets.NewString(
	string(ResourceCPU),
	string(ResourceMemory),
)

// IsStandardContainerResourceName returns true if the container can make a resource request
// for the specified resource
func IsStandardContainerResourceName(str string) bool {
	return standardContainerResources.Has(str)
}

var standardLimitRangeTypes = sets.NewString(
	string(LimitTypePod),
	string(LimitTypeContainer),
)

// IsStandardLimitRangeType returns true if the type is Pod or Container
func IsStandardLimitRangeType(str string) bool {
	return standardLimitRangeTypes.Has(str)
}

var standardQuotaResources = sets.NewString(
	string(ResourceCPU),
	string(ResourceMemory),
	string(ResourceRequestsCPU),
	string(ResourceRequestsMemory),
	string(ResourceLimitsCPU),
	string(ResourceLimitsMemory),
	string(ResourcePods),
	string(ResourceQuotas),
	string(ResourceServices),
	string(ResourceReplicationControllers),
	string(ResourceSecrets),
	string(ResourcePersistentVolumeClaims),
	string(ResourceConfigMaps),
)

// IsStandardQuotaResourceName returns true if the resource is known to
// the quota tracking system
func IsStandardQuotaResourceName(str string) bool {
	return standardQuotaResources.Has(str)
}

var standardResources = sets.NewString(
	string(ResourceCPU),
	string(ResourceMemory),
	string(ResourceRequestsCPU),
	string(ResourceRequestsMemory),
	string(ResourceLimitsCPU),
	string(ResourceLimitsMemory),
	string(ResourcePods),
	string(ResourceQuotas),
	string(ResourceServices),
	string(ResourceReplicationControllers),
	string(ResourceSecrets),
	string(ResourceConfigMaps),
	string(ResourcePersistentVolumeClaims),
	string(ResourceStorage),
)

// IsStandardResourceName returns true if the resource is known to the system
func IsStandardResourceName(str string) bool {
	return standardResources.Has(str)
}

var integerResources = sets.NewString(
	string(ResourcePods),
	string(ResourceQuotas),
	string(ResourceServices),
	string(ResourceReplicationControllers),
	string(ResourceSecrets),
	string(ResourceConfigMaps),
	string(ResourcePersistentVolumeClaims),
)

// IsIntegerResourceName returns true if the resource is measured in integer values
func IsIntegerResourceName(str string) bool {
	return integerResources.Has(str)
}

// NewDeleteOptions returns a DeleteOptions indicating the resource should
// be deleted within the specified grace period. Use zero to indicate
// immediate deletion. If you would prefer to use the default grace period,
// use &api.DeleteOptions{} directly.
func NewDeleteOptions(grace int64) *DeleteOptions {
	return &DeleteOptions{GracePeriodSeconds: &grace}
}

// this function aims to check if the service's ClusterIP is set or not
// the objective is not to perform validation here
func IsServiceIPSet(service *Service) bool {
	return service.Spec.ClusterIP != ClusterIPNone && service.Spec.ClusterIP != ""
}

// this function aims to check if the service's cluster IP is requested or not
func IsServiceIPRequested(service *Service) bool {
	return service.Spec.ClusterIP == ""
}

var standardFinalizers = sets.NewString(
	string(FinalizerKubernetes))

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

func HashObject(obj runtime.Object, codec runtime.Codec) (string, error) {
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", md5.Sum(data)), nil
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

// ParseRFC3339 parses an RFC3339 date in either RFC3339Nano or RFC3339 format.
func ParseRFC3339(s string, nowFn func() unversioned.Time) (unversioned.Time, error) {
	if t, timeErr := time.Parse(time.RFC3339Nano, s); timeErr == nil {
		return unversioned.Time{t}, nil
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return unversioned.Time{}, err
	}
	return unversioned.Time{t}, nil
}

// NodeSelectorRequirementsAsSelector converts the []NodeSelectorRequirement api type into a struct that implements
// labels.Selector.
func NodeSelectorRequirementsAsSelector(nsm []NodeSelectorRequirement) (labels.Selector, error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	selector := labels.NewSelector()
	for _, expr := range nsm {
		var op labels.Operator
		switch expr.Operator {
		case NodeSelectorOpIn:
			op = labels.InOperator
		case NodeSelectorOpNotIn:
			op = labels.NotInOperator
		case NodeSelectorOpExists:
			op = labels.ExistsOperator
		case NodeSelectorOpDoesNotExist:
			op = labels.DoesNotExistOperator
		case NodeSelectorOpGt:
			op = labels.GreaterThanOperator
		case NodeSelectorOpLt:
			op = labels.LessThanOperator
		default:
			return nil, fmt.Errorf("%q is not a valid node selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, sets.NewString(expr.Values...))
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	return selector, nil
}

// AffinityAnnotationKey represents the key of affinity data (json serialized)
// in the Annotations of a Pod.
const AffinityAnnotationKey string = "scheduler.alpha.kubernetes.io/affinity"

// GetAffinityFromPod gets the json serialized affinity data from Pod.Annotations
// and converts it to the Affinity type in api.
func GetAffinityFromPodAnnotations(annotations map[string]string) (Affinity, error) {
	var affinity Affinity
	if len(annotations) > 0 && annotations[AffinityAnnotationKey] != "" {
		err := json.Unmarshal([]byte(annotations[AffinityAnnotationKey]), &affinity)
		if err != nil {
			return affinity, err
		}
	}
	return affinity, nil
}
