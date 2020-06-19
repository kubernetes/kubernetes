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

package v1beta1

import (
	v1 "k8s.io/api/core/v1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_HTTPIngressPath(obj *networkingv1beta1.HTTPIngressPath) {
	var defaultPathType = networkingv1beta1.PathTypeImplementationSpecific
	if obj.PathType == nil {
		obj.PathType = &defaultPathType
	}
}

func SetDefaults_Ingress(obj *networkingv1beta1.Ingress) {
	if utilfeature.DefaultFeatureGate.Enabled(features.LoadBalancerIPMode) {
		ipMode := v1.LoadBalancerIPModeVIP

		for i, ing := range obj.Status.LoadBalancer.Ingress {
			if ing.IP != "" && ing.IPMode == nil {
				obj.Status.LoadBalancer.Ingress[i].IPMode = &ipMode
			}
		}
	}
}
