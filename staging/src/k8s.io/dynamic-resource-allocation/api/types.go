/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"iter"
	"slices"
	"strings"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	apiservercel "k8s.io/apiserver/pkg/cel"
)

// FullyQualifiedName is a fully-qualified name (i.e. domain is non-empty) with both parts already split apart.
//
// +k8s:conversion-gen=false
type FullyQualifiedName struct {
	Domain     UniqueString
	Identifier UniqueString
}

func (n FullyQualifiedName) String() string {
	return n.Domain.String() + "/" + n.Identifier.String()
}

// MakeFullyQualifiedName creates a [FullyQualifiedName] from a [resourceapi.QualifiedName].
// The name may or may not have an explicit domain; if it doesn't, the default domain is used.
func MakeFullyQualifiedName(name resourceapi.QualifiedName, defaultDomain UniqueString, makeUnique func(string) UniqueString) FullyQualifiedName {
	domain, identifier, hasDomain := strings.Cut(string(name), "/")
	if !hasDomain {
		return FullyQualifiedName{Domain: defaultDomain, Identifier: makeUnique(string(name))}
	}
	return FullyQualifiedName{Domain: makeUnique(domain), Identifier: makeUnique(identifier)}
}

// JSON tags exist to make the output more readable (klog, diff.Diff).
// They are intentionally not compatible with the normal encoding
// of a ResourceSlice to avoid accidentally using them with an apiserver
// request:
// - TypeMeta does not get encoded.
// - Fields from this package use upper case whereas types from the
//   real API use lower case.

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type ResourceSlice struct {
	metav1.TypeMeta `json:"-"` // Not needed, not set consistently.
	metav1.ObjectMeta

	Spec ResourceSliceSpec

	// uniqueStringMap is a cache for looking up the UniqueString instances
	// used in this slice. MakeUniqueString would yield the same result,
	// but must lock.
	uniqueStringMap map[string]UniqueString
}

// MakeUniqueString ensures that the string is in the per-slice
// unique string cache and returns the unique string for it.
//
// Conversion of a resourceapi.ResourceSlice into this ResourceSlice
// uses this method, therefore all unique strings used for this
// instance are cached after conversion.
func (r *ResourceSlice) MakeUniqueString(str string) UniqueString {
	if r.uniqueStringMap == nil {
		r.uniqueStringMap = make(map[string]UniqueString)
	}
	u, ok := r.uniqueStringMap[str]
	if ok {
		return u
	}
	u = MakeUniqueString(str)
	r.uniqueStringMap[str] = u
	return u
}

// LookupUniqueString returns a unique string if the string is in
// the cache populated by MakeUniqueString, otherwise [NullUniqueString].
func (r *ResourceSlice) LookupUniqueString(str string) UniqueString {
	// Most likely string: the driver name. It's at the root of most
	// attribute and capacity lookups.
	if str == r.Spec.Driver.String() {
		return r.Spec.Driver
	}
	u, ok := r.uniqueStringMap[str]
	if ok {
		return u
	}
	return NullUniqueString
}

type ResourceSliceSpec struct {
	Driver                 UniqueString
	Pool                   ResourcePool
	NodeName               *string                         `json:",omitempty"`
	NodeSelector           *v1.NodeSelector                `json:",omitempty"`
	AllNodes               bool                            `json:",omitempty"`
	Devices                []Device                        `json:",omitempty"`
	PerDeviceNodeSelection *bool                           `json:",omitempty"`
	SharedCounters         []CounterSet                    `json:",omitempty"`
	PartitionTypeAttribute *resourceapi.FullyQualifiedName `json:",omitempty"`
	SkipNodeOperations     []resourceapi.SkipNodeOperation `json:",omitempty"`
}

type CounterSet struct {
	Name     UniqueString
	Counters map[string]resourceapi.Counter `json:",omitempty"`
}

type ResourcePool struct {
	Name               UniqueString
	Generation         int64
	ResourceSliceCount int64
}

type Device struct {
	Name                     UniqueString
	Attributes               DeviceAttributes                                        `json:",omitempty"`
	Capacity                 DeviceCapacities                                        `json:",omitempty"`
	ConsumesCounters         []DeviceCounterConsumption                              `json:",omitempty"`
	NodeName                 *string                                                 `json:",omitempty"`
	NodeSelector             *v1.NodeSelector                                        `json:",omitempty"`
	AllNodes                 *bool                                                   `json:",omitempty"`
	Taints                   []resourceapi.DeviceTaint                               `json:",omitempty"`
	BindsToNode              bool                                                    `json:",omitempty"`
	BindingConditions        []string                                                `json:",omitempty"`
	BindingFailureConditions []string                                                `json:",omitempty"`
	AllowMultipleAllocations *bool                                                   `json:",omitempty"`
	NodeAllocatableResources map[v1.ResourceName]resourceapi.NodeAllocatableResource `json:",omitempty"`
}

type DeviceCounterConsumption struct {
	CounterSet          UniqueString
	Counters            map[string]resourceapi.Counter `json:",omitempty"`
	CompatibilityGroups []string                       `json:",omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type ResourceSliceList struct {
	metav1.TypeMeta
	metav1.ListMeta
	Items []ResourceSlice
}

// DeviceAttributes maps domain + id from a FullyQualifiedName to the attribute value.
// String/bool/int values are stored as such. Version values are stored as
// apiservercel.Semver. Lists are stored as slices of their values, again using
// apiservercel.Semver.
type DeviceAttributes struct {
	Nested map[UniqueString]map[UniqueString]any

	// DriverNameQualifiedIDs contains those identifiers (e.g. "version")
	// which were unnecessarily qualified with the driver name as domain ("dra.example.com/foo")
	// in the original ResourceSlice. This is unnecessary and not recommended,
	// so typically this will be nil.
	//
	// This set is used for two purposes:
	// - Restoring the original name during round-tripping - not particularly important.
	// - Writing consumed capacity into a ResourceClaim status with the exact same name
	//   as used by the driver in the ResourceSlice (https://github.com/kubernetes/kubernetes/pull/142202#discussion_r4070913314).
	//   That usage might go away once we no longer need to support downgrades to
	//   Kubernetes 1.37, in which case DriverNameQualifiedIDs can be removed.
	DriverNameQualifiedIDs sets.Set[UniqueString]

	// DriverName is the driver name used when constructing this instance.
	DriverName UniqueString
}

func (m DeviceAttributes) DeepCopy() DeviceAttributes {
	if m.Nested == nil {
		return DeviceAttributes{}
	}
	out := DeviceAttributes{
		Nested:     make(map[UniqueString]map[UniqueString]any, len(m.Nested)),
		DriverName: m.DriverName,
	}
	if m.DriverNameQualifiedIDs != nil {
		out.DriverNameQualifiedIDs = m.DriverNameQualifiedIDs.Clone()
	}
	for k, v := range m.Nested {
		if v == nil {
			out.Nested[k] = nil
			continue
		}
		inner := make(map[UniqueString]any, len(v))
		for k, v := range v {
			switch v := v.(type) {
			case int64:
				inner[k] = v
			case bool:
				inner[k] = v
			case string:
				inner[k] = v
			case apiservercel.Semver:
				inner[k] = v
			case []int64:
				inner[k] = slices.Clone(v)
			case []bool:
				inner[k] = slices.Clone(v)
			case []string:
				inner[k] = slices.Clone(v)
			case []apiservercel.Semver:
				inner[k] = slices.Clone(v)
			default:
				panic(fmt.Sprintf("internal error, missing case for %T", v))
			}
		}
		out.Nested[k] = inner
	}

	return out
}

func (m DeviceAttributes) Lookup(name FullyQualifiedName) any {
	return m.Nested[name.Domain][name.Identifier]
}

// DeviceCapacity holds the capacity value for a single capacity entry.
// The value uses apiservercel.Quantity for direct use in CEL evaluation.
type DeviceCapacity struct {
	Value         apiservercel.Quantity
	RequestPolicy *resourceapi.CapacityRequestPolicy
}

// DeepCopy returns a deep copy of DeviceCapacity.
func (in *DeviceCapacity) DeepCopy() *DeviceCapacity {
	if in == nil {
		return nil
	}
	out := new(DeviceCapacity)
	valueCopy := in.Value.DeepCopy()
	out.Value = apiservercel.Quantity{Quantity: &valueCopy}
	out.RequestPolicy = in.RequestPolicy.DeepCopy()
	return out
}

// DeviceCapacities maps domain + unqualified name to the capacity value,
// see DeviceAttributes
type DeviceCapacities struct {
	Nested                 map[UniqueString]map[UniqueString]DeviceCapacity
	DriverNameQualifiedIDs sets.Set[UniqueString]
	DriverName             UniqueString
}

func (m DeviceCapacities) DeepCopy() DeviceCapacities {
	if m.Nested == nil {
		return DeviceCapacities{}
	}
	out := DeviceCapacities{
		Nested:     make(map[UniqueString]map[UniqueString]DeviceCapacity, len(m.Nested)),
		DriverName: m.DriverName,
	}
	if m.DriverNameQualifiedIDs != nil {
		out.DriverNameQualifiedIDs = m.DriverNameQualifiedIDs.Clone()
	}
	for domain, inner := range m.Nested {
		if inner == nil {
			out.Nested[domain] = nil
			continue
		}
		innerCopy := make(map[UniqueString]DeviceCapacity, len(inner))
		for name, cap := range inner {
			innerCopy[name] = *cap.DeepCopy()
		}
		out.Nested[domain] = innerCopy
	}
	return out
}

func (m DeviceCapacities) Lookup(name FullyQualifiedName) (DeviceCapacity, bool) {
	inner, ok := m.Nested[name.Domain]
	if !ok {
		return DeviceCapacity{}, false
	}
	cap, ok := inner[name.Identifier]
	return cap, ok
}

// Lookup returns the entry for a fully qualified name.
func (m DeviceCapacities) LookupByFullName(name resourceapi.FullyQualifiedName) (DeviceCapacity, bool) {
	domain, id, _ := strings.Cut(string(name), "/")

	// Creating unique strings is probably slower than direct string comparisons,
	// despite having to loop here.
	for d, inner := range m.Nested {
		if d.String() == domain {
			for n, cap := range inner {
				if n.String() == id {
					return cap, true
				}
			}
		}
	}

	return DeviceCapacity{}, false
}

// Entries iterates over all capacities with fully qualified name as key,
// separated into domain and identifier (in this order)
func (m DeviceCapacities) Entries() iter.Seq2[FullyQualifiedName, DeviceCapacity] {
	return func(yield func(FullyQualifiedName, DeviceCapacity) bool) {
		for domain, inner := range m.Nested {
			for id, cap := range inner {
				if !yield(FullyQualifiedName{Domain: domain, Identifier: id}, cap) {
					return
				}
			}
		}
	}
}

// Len returns the total number of capacities across all domains.
func (m DeviceCapacities) Len() int {
	l := 0
	for _, inner := range m.Nested {
		l += len(inner)
	}
	return l
}
