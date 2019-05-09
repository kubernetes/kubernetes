/*
Copyright 2018 The Kubernetes Authors.

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

package v1alpha1

import (
	flowregistrationv1alpha1 "k8s.io/api/flowregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
)

// TODO(aaron-prindle) actually implement defaulting

// DefaultThrottle is a default throttle config
func DefaultFlowSchema() *flowregistrationv1alpha1.FlowSchema {
	return &flowregistrationv1alpha1.FlowSchema{}
}

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_FlowSchema sets defaults if the flow sink isn't present
func SetDefaults_FlowSchema(obj *flowregistrationv1alpha1.FlowSchema) {
}
