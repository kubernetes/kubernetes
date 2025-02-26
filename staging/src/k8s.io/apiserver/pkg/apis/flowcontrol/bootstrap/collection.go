/*
Copyright 2025 The Kubernetes Authors.

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

package bootstrap

import (
	flowcontrolv1 "k8s.io/api/flowcontrol/v1"
	"k8s.io/apiserver/pkg/apis/flowcontrol/base"
	bootstrapold "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap-old"
	bootstrapv134 "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap-v134"
)

// This package provides access to the "default" configuration objects of
// API Priority and Fairness according to the settings of the Feature(s)
// that control which collection of defaults to use.
// Currently there is exactly one such Feature, APFv134Config.

// GetV1ConfigCollection is given the values of the features that define
// the default APF configuration objects to use, and returns them (in V1 form).
// The returned value is deeply immutable.
// v134 is the value of the APFv134Config feature.
func GetV1ConfigCollection(v134 bool) *base.V1ConfigCollection {
	if v134 {
		return &bootstrapv134.V1ConfigCollection
	} else {
		return &bootstrapold.V1ConfigCollection
	}
}

// GetPrioritylevelConfigurations returns the default PriorityLevelConfiguration objects,
// as a deeply immutable slice.
// The types are the latest external type (version).
// The arguments are the values of the features that control which config collection to use.
func GetPrioritylevelConfigurations(v134 bool) []*flowcontrolv1.PriorityLevelConfiguration {
	return PrioritylevelConfigurations[CollectionID{V134: v134}]
}

// PrioritylevelConfigurations maps CollectionId to a slice of all the default objects.
// Deeply immutable.
var PrioritylevelConfigurations = map[CollectionID][]*flowcontrolv1.PriorityLevelConfiguration{
	{false}: Concat(bootstrapold.V1ConfigCollection.Mandatory.PriorityLevelConfigurations,
		bootstrapold.V1ConfigCollection.Suggested.PriorityLevelConfigurations),
	{true}: Concat(bootstrapv134.V1ConfigCollection.Mandatory.PriorityLevelConfigurations,
		bootstrapv134.V1ConfigCollection.Suggested.PriorityLevelConfigurations),
}

// GetFlowSchemas returns the default FlowSchema objects,
// as a deeply immutable slice.
// The types are the latest external type (version).
// The arguments are the values of the features that control which config collection to use.
func GetFlowSchemas(v134 bool) []*flowcontrolv1.FlowSchema {
	return FlowSchemas[CollectionID{V134: v134}]
}

// FlowSchemas maps CollectionId to a slice of all the default objects.
// Deeply immutable.
var FlowSchemas = map[CollectionID][]*flowcontrolv1.FlowSchema{
	{false}: Concat(bootstrapold.V1ConfigCollection.Mandatory.FlowSchemas,
		bootstrapold.V1ConfigCollection.Suggested.FlowSchemas),
	{true}: Concat(bootstrapv134.V1ConfigCollection.Mandatory.FlowSchemas,
		bootstrapv134.V1ConfigCollection.Suggested.FlowSchemas),
}

// CollectionID identifies the collection of API Priority and Fairness config objects
// to use as the defaults.
type CollectionID struct {
	// V134=true means use the collection introduced for release 1.34.
	// V134=false means use the older collection.
	V134 bool
}

// Concat concatenates the given slices, without reusing any given backing array.
func Concat[T any](slices ...[]T) []T {
	var ans []T
	for _, slice := range slices {
		ans = append(ans, slice...)
	}
	return ans
}
