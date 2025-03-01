/*
Copyright 2019 The Kubernetes Authors.

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

package internalbootstrap

import (
	flowcontrolv1 "k8s.io/api/flowcontrol/v1"
	"k8s.io/apimachinery/pkg/runtime"
	bootstrapnew "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap-new"
	bootstrapold "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap-old"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"k8s.io/kubernetes/pkg/apis/flowcontrol/install"
)

// Concat concatenates the given slices, without reusing any given backing array.
func Concat[T any](slices ...[]T) []T {
	var ans []T
	for _, slice := range slices {
		ans = append(ans, slice...)
	}
	return ans
}

// PrioritylevelConfigurations maps `newConfig bool` to a slice of all the default objects
// Deeply immutable.
var PrioritylevelConfigurations = map[bool][]*flowcontrolv1.PriorityLevelConfiguration{
	false: Concat(bootstrapold.MandatoryPriorityLevelConfigurations, bootstrapold.SuggestedPriorityLevelConfigurations),
	true:  Concat(bootstrapnew.MandatoryPriorityLevelConfigurations, bootstrapnew.SuggestedPriorityLevelConfigurations),
}

// FlowSchemas maps `newConfig bool` to a slice of all the default objects
// Deeply immutable.
var FlowSchemas = map[bool][]*flowcontrolv1.FlowSchema{
	false: Concat(bootstrapold.MandatoryFlowSchemas, bootstrapold.SuggestedFlowSchemas),
	true:  Concat(bootstrapnew.MandatoryFlowSchemas, bootstrapnew.SuggestedFlowSchemas),
}

// MandatoryFlowSchemas holds the untyped renditions of the mandatory
// flow schemas.  In the outer map the key is `newConfig bool` and
// in the inner map the key is the schema's name and the
// value is the `*FlowSchema`.  Nobody should mutate anything
// reachable from this map.
var MandatoryFlowSchemas = map[bool]map[string]*flowcontrol.FlowSchema{
	false: internalizeFSes(bootstrapold.MandatoryFlowSchemas),
	true:  internalizeFSes(bootstrapnew.MandatoryFlowSchemas),
}

// MandatoryPriorityLevelConfigurations holds the untyped renditions of the
// mandatory priority level configuration objects.  In the outer map the key is `newConfig bool` and
// in the inner map the key is the object's name and the value is the
// `*PriorityLevelConfiguration`.  Nobody should mutate anything
// reachable from this map.
var MandatoryPriorityLevelConfigurations = map[bool]map[string]*flowcontrol.PriorityLevelConfiguration{
	false: internalizePLs(bootstrapold.MandatoryPriorityLevelConfigurations),
	true:  internalizePLs(bootstrapnew.MandatoryPriorityLevelConfigurations),
}

// NewAPFScheme constructs and returns a Scheme configured to handle
// the API object types that are used to configure API Priority and
// Fairness
func NewAPFScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	install.Install(scheme)
	return scheme
}

func internalizeFSes(exts []*flowcontrolv1.FlowSchema) map[string]*flowcontrol.FlowSchema {
	ans := make(map[string]*flowcontrol.FlowSchema, len(exts))
	scheme := NewAPFScheme()
	for _, ext := range exts {
		var untyped flowcontrol.FlowSchema
		if err := scheme.Convert(ext, &untyped, nil); err != nil {
			panic(err)
		}
		ans[ext.Name] = &untyped
	}
	return ans
}

func internalizePLs(exts []*flowcontrolv1.PriorityLevelConfiguration) map[string]*flowcontrol.PriorityLevelConfiguration {
	ans := make(map[string]*flowcontrol.PriorityLevelConfiguration, len(exts))
	scheme := NewAPFScheme()
	for _, ext := range exts {
		var untyped flowcontrol.PriorityLevelConfiguration
		if err := scheme.Convert(ext, &untyped, nil); err != nil {
			panic(err)
		}
		ans[ext.Name] = &untyped
	}
	return ans
}
