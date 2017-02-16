/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/v1"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	RegisterDefaults(scheme)
	return scheme.AddDefaultingFuncs(
		SetDefaults_DaemonSet,
		SetDefaults_Deployment,
		SetDefaults_ReplicaSet,
		SetDefaults_NetworkPolicy,
		SetDefaults_ThirdPartyResource,
	)
}

func convertToCamelCase(input string) string {
	result := ""
	toUpper := true
	for ix := range input {
		char := input[ix]
		if toUpper {
			result = result + string([]byte{(char - 32)})
			toUpper = false
		} else if char == '-' {
			toUpper = true
		} else {
			result = result + string([]byte{char})
		}
	}
	return result
}

// ExtractAPIGroupAndKind turns the object name in the form "foo-bar.acme.com" into a kind FooBar and group acme.com.
// It is *only* used to populate the kind and group in v1beta1 for backwards compatibility.
func ExtractAPIGroupAndKind(name string) (kind string, group string, err error) {
	parts := strings.Split(name, ".")
	if len(parts) < 3 {
		return "", "", fmt.Errorf("unexpectedly short resource name: %s, expected at least <resource>.<domain>.<tld>", name)
	}
	return convertToCamelCase(parts[0]), strings.Join(parts[1:], "."), nil
}

func SetDefaults_ThirdPartyResource(obj *ThirdPartyResource) {
	// Default a completely empty spec
	if reflect.DeepEqual(obj.Spec, ThirdPartyResourceSpec{}) {
		gvk := schema.GroupVersionKind{}
		if len(obj.Versions) > 0 {
			gvk.Version = obj.Versions[0].Name
		}
		gvk.Kind, gvk.Group, _ = ExtractAPIGroupAndKind(obj.Name)
		plural, singular := meta.KindToResource(gvk)

		obj.Spec.Group = gvk.Group
		obj.Spec.Kind = gvk.Kind
		obj.Spec.Version = gvk.Version
		obj.Spec.Resource = plural.Resource
		obj.Spec.ResourceSingular = singular.Resource
		t := true
		obj.Spec.Namespaced = &t
		obj.Spec.Description = obj.Description
	}

	if len(obj.Status.Conditions) == 0 {
		obj.Status.Conditions = []ThirdPartyResourceCondition{{
			Type:           ThirdPartyResourceActive,
			Status:         ThirdPartyResourceConditionStatusUnknown,
			LastUpdateTime: obj.CreationTimestamp,
		}}
	}
}

func SetDefaults_DaemonSet(obj *DaemonSet) {
	labels := obj.Spec.Template.Labels

	// TODO: support templates defined elsewhere when we support them in the API
	if labels != nil {
		if obj.Spec.Selector == nil {
			obj.Spec.Selector = &metav1.LabelSelector{
				MatchLabels: labels,
			}
		}
		if len(obj.Labels) == 0 {
			obj.Labels = labels
		}
	}
}

func SetDefaults_Deployment(obj *Deployment) {
	// Default labels and selector to labels from pod template spec.
	labels := obj.Spec.Template.Labels

	if labels != nil {
		if obj.Spec.Selector == nil {
			obj.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
		}
		if len(obj.Labels) == 0 {
			obj.Labels = labels
		}
	}
	// Set DeploymentSpec.Replicas to 1 if it is not set.
	if obj.Spec.Replicas == nil {
		obj.Spec.Replicas = new(int32)
		*obj.Spec.Replicas = 1
	}
	strategy := &obj.Spec.Strategy
	// Set default DeploymentStrategyType as RollingUpdate.
	if strategy.Type == "" {
		strategy.Type = RollingUpdateDeploymentStrategyType
	}
	if strategy.Type == RollingUpdateDeploymentStrategyType || strategy.RollingUpdate != nil {
		if strategy.RollingUpdate == nil {
			rollingUpdate := RollingUpdateDeployment{}
			strategy.RollingUpdate = &rollingUpdate
		}
		if strategy.RollingUpdate.MaxUnavailable == nil {
			// Set default MaxUnavailable as 1 by default.
			maxUnavailable := intstr.FromInt(1)
			strategy.RollingUpdate.MaxUnavailable = &maxUnavailable
		}
		if strategy.RollingUpdate.MaxSurge == nil {
			// Set default MaxSurge as 1 by default.
			maxSurge := intstr.FromInt(1)
			strategy.RollingUpdate.MaxSurge = &maxSurge
		}
	}
}

func SetDefaults_ReplicaSet(obj *ReplicaSet) {
	labels := obj.Spec.Template.Labels

	// TODO: support templates defined elsewhere when we support them in the API
	if labels != nil {
		if obj.Spec.Selector == nil {
			obj.Spec.Selector = &metav1.LabelSelector{
				MatchLabels: labels,
			}
		}
		if len(obj.Labels) == 0 {
			obj.Labels = labels
		}
	}
	if obj.Spec.Replicas == nil {
		obj.Spec.Replicas = new(int32)
		*obj.Spec.Replicas = 1
	}
}

func SetDefaults_NetworkPolicy(obj *NetworkPolicy) {
	// Default any undefined Protocol fields to TCP.
	for _, i := range obj.Spec.Ingress {
		// TODO: Update Ports to be a pointer to slice as soon as auto-generation supports it.
		for _, p := range i.Ports {
			if p.Protocol == nil {
				proto := v1.ProtocolTCP
				p.Protocol = &proto
			}
		}
	}
}
