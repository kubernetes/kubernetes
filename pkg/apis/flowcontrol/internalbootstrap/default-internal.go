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
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"k8s.io/kubernetes/pkg/apis/flowcontrol/install"
)

// FlowSchemasMap is a collection of unversioned (internal) FlowSchema objects indexed by name.
type FlowSchemasMap = map[string]*flowcontrol.FlowSchema

// GetMandatoryFlowSchemasMap returns the unversioned (internal) mandatory FlowSchema objects,
// as a deeply immutable map.
// The arguments are the values of the features that control which config collection to use.
func GetMandatoryFlowSchemasMap(featureGate featuregate.FeatureGate) FlowSchemasMap {
	return MandatoryFlowSchemasMap[bootstrap.CollectionID{V134: featureGate.Enabled(features.APFv134Config)}]
}

var oldConfig = bootstrap.GetV1ConfigCollection(bootstrap.MakeGate(false))

// MandatoryFlowSchemasMap holds the unversioned (internal) renditions of the mandatory
// flow schemas.  In the outer map the key is CollectionId and
// in the inner map the key is the schema's name and the
// value is the `*FlowSchema`.  Nobody should mutate anything
// reachable from this map.
var MandatoryFlowSchemasMap = map[bootstrap.CollectionID]FlowSchemasMap{
	{V134: false}: internalizeFSes(oldConfig.Mandatory.FlowSchemas),
	{V134: true}:  internalizeFSes(bootstrap.Latest.Mandatory.FlowSchemas),
}

// PriorityLevelConfigurationsMap is a collection of unversioned (internal) PriorityLevelConfiguration objects, indexed by name.
type PriorityLevelConfigurationsMap = map[string]*flowcontrol.PriorityLevelConfiguration

// GetMandatoryPriorityLevelConfigurationsMap returns the mandatory PriorityLevelConfiguration objects,
// as a deeply immutable map.
// The arguments are the values of the features that control which config collection to use.
func GetMandatoryPriorityLevelConfigurationsMap(featureGate featuregate.FeatureGate) PriorityLevelConfigurationsMap {
	return MandatoryPriorityLevelConfigurationsMap[bootstrap.CollectionID{V134: featureGate.Enabled(features.APFv134Config)}]
}

// MandatoryPriorityLevelConfigurationsMap holds the untyped renditions of the
// mandatory priority level configuration objects.  In the outer map the key is `bootstrap.CollectionID` and
// in the inner map the key is the object's name and the value is the
// `*PriorityLevelConfiguration`.  Nobody should mutate anything
// reachable from this map.
var MandatoryPriorityLevelConfigurationsMap = map[bootstrap.CollectionID]PriorityLevelConfigurationsMap{
	{V134: false}: internalizePLs(oldConfig.Mandatory.PriorityLevelConfigurations),
	{V134: true}:  internalizePLs(bootstrap.Latest.Mandatory.PriorityLevelConfigurations),
}

func internalizeFSes(exts []*flowcontrolv1.FlowSchema) FlowSchemasMap {
	ans := make(FlowSchemasMap, len(exts))
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

func internalizePLs(exts []*flowcontrolv1.PriorityLevelConfiguration) PriorityLevelConfigurationsMap {
	ans := make(PriorityLevelConfigurationsMap, len(exts))
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

// NewAPFScheme constructs and returns a Scheme configured to handle
// the API object types that are used to configure API Priority and
// Fairness
func NewAPFScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	install.Install(scheme)
	return scheme
}
