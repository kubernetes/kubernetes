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
	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"k8s.io/kubernetes/pkg/apis/flowcontrol/install"
)

// MandatoryFlowSchemas holds the untyped renditions of the mandatory
// flow schemas.  In this map the key is the schema's name and the
// value is the `*FlowSchema`.  Nobody should mutate anything
// reachable from this map.
var MandatoryFlowSchemas = internalizeFSes(bootstrap.MandatoryFlowSchemas)

// MandatoryPriorityLevelConfigurations holds the untyped renditions of the
// mandatory priority level configuration objects.  In this map the
// key is the object's name and the value is the
// `*PriorityLevelConfiguration`.  Nobody should mutate anything
// reachable from this map.
var MandatoryPriorityLevelConfigurations = internalizePLs(bootstrap.MandatoryPriorityLevelConfigurations)

// NewAPFScheme constructs and returns a Scheme configured to handle
// the API object types that are used to configure API Priority and
// Fairness
func NewAPFScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	install.Install(scheme)
	return scheme
}

func internalizeFSes(exts []*fcv1a1.FlowSchema) map[string]*flowcontrol.FlowSchema {
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

func internalizePLs(exts []*fcv1a1.PriorityLevelConfiguration) map[string]*flowcontrol.PriorityLevelConfiguration {
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
