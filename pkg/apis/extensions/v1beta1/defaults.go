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
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_DaemonSet(obj *extensionsv1beta1.DaemonSet) {
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
	updateStrategy := &obj.Spec.UpdateStrategy
	if updateStrategy.Type == "" {
		updateStrategy.Type = extensionsv1beta1.OnDeleteDaemonSetStrategyType
	}
	if updateStrategy.Type == extensionsv1beta1.RollingUpdateDaemonSetStrategyType {
		if updateStrategy.RollingUpdate == nil {
			rollingUpdate := extensionsv1beta1.RollingUpdateDaemonSet{}
			updateStrategy.RollingUpdate = &rollingUpdate
		}
		if updateStrategy.RollingUpdate.MaxUnavailable == nil {
			// Set default MaxUnavailable as 1 by default.
			maxUnavailable := intstr.FromInt(1)
			updateStrategy.RollingUpdate.MaxUnavailable = &maxUnavailable
		}
	}
	if obj.Spec.RevisionHistoryLimit == nil {
		obj.Spec.RevisionHistoryLimit = new(int32)
		*obj.Spec.RevisionHistoryLimit = 10
	}
}

func SetDefaults_Deployment(obj *extensionsv1beta1.Deployment) {
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
	// Set extensionsv1beta1.DeploymentSpec.Replicas to 1 if it is not set.
	if obj.Spec.Replicas == nil {
		obj.Spec.Replicas = new(int32)
		*obj.Spec.Replicas = 1
	}
	strategy := &obj.Spec.Strategy
	// Set default extensionsv1beta1.DeploymentStrategyType as RollingUpdate.
	if strategy.Type == "" {
		strategy.Type = extensionsv1beta1.RollingUpdateDeploymentStrategyType
	}
	if strategy.Type == extensionsv1beta1.RollingUpdateDeploymentStrategyType || strategy.RollingUpdate != nil {
		if strategy.RollingUpdate == nil {
			rollingUpdate := extensionsv1beta1.RollingUpdateDeployment{}
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

func SetDefaults_ReplicaSet(obj *extensionsv1beta1.ReplicaSet) {
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

func SetDefaults_NetworkPolicy(obj *extensionsv1beta1.NetworkPolicy) {
	// Default any undefined Protocol fields to TCP.
	for _, i := range obj.Spec.Ingress {
		for _, p := range i.Ports {
			if p.Protocol == nil {
				proto := v1.ProtocolTCP
				p.Protocol = &proto
			}
		}
	}

	if len(obj.Spec.PolicyTypes) == 0 {
		// Any policy that does not specify policyTypes implies at least "Ingress".
		obj.Spec.PolicyTypes = []extensionsv1beta1.PolicyType{extensionsv1beta1.PolicyTypeIngress}
		if len(obj.Spec.Egress) != 0 {
			obj.Spec.PolicyTypes = append(obj.Spec.PolicyTypes, extensionsv1beta1.PolicyTypeEgress)
		}
	}
}
