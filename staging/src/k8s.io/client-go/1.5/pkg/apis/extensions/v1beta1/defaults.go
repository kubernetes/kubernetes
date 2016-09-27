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
	"k8s.io/client-go/1.5/pkg/api/v1"
	"k8s.io/client-go/1.5/pkg/runtime"
	"k8s.io/client-go/1.5/pkg/util/intstr"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return scheme.AddDefaultingFuncs(
		SetDefaults_DaemonSet,
		SetDefaults_Deployment,
		SetDefaults_Job,
		SetDefaults_HorizontalPodAutoscaler,
		SetDefaults_ReplicaSet,
		SetDefaults_NetworkPolicy,
	)
}

func SetDefaults_DaemonSet(obj *DaemonSet) {
	labels := obj.Spec.Template.Labels

	// TODO: support templates defined elsewhere when we support them in the API
	if labels != nil {
		if obj.Spec.Selector == nil {
			obj.Spec.Selector = &LabelSelector{
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
			obj.Spec.Selector = &LabelSelector{MatchLabels: labels}
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
	if strategy.Type == RollingUpdateDeploymentStrategyType {
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

func SetDefaults_Job(obj *Job) {
	labels := obj.Spec.Template.Labels
	// TODO: support templates defined elsewhere when we support them in the API
	if labels != nil {
		// if an autoselector is requested, we'll build the selector later with controller-uid and job-name
		autoSelector := bool(obj.Spec.AutoSelector != nil && *obj.Spec.AutoSelector)

		// otherwise, we are using a manual selector
		manualSelector := !autoSelector

		// and default behavior for an unspecified manual selector is to use the pod template labels
		if manualSelector && obj.Spec.Selector == nil {
			obj.Spec.Selector = &LabelSelector{
				MatchLabels: labels,
			}
		}
		if len(obj.Labels) == 0 {
			obj.Labels = labels
		}
	}
	// For a non-parallel job, you can leave both `.spec.completions` and
	// `.spec.parallelism` unset.  When both are unset, both are defaulted to 1.
	if obj.Spec.Completions == nil && obj.Spec.Parallelism == nil {
		obj.Spec.Completions = new(int32)
		*obj.Spec.Completions = 1
		obj.Spec.Parallelism = new(int32)
		*obj.Spec.Parallelism = 1
	}
	if obj.Spec.Parallelism == nil {
		obj.Spec.Parallelism = new(int32)
		*obj.Spec.Parallelism = 1
	}
}

func SetDefaults_HorizontalPodAutoscaler(obj *HorizontalPodAutoscaler) {
	if obj.Spec.MinReplicas == nil {
		minReplicas := int32(1)
		obj.Spec.MinReplicas = &minReplicas
	}
	if obj.Spec.CPUUtilization == nil {
		obj.Spec.CPUUtilization = &CPUTargetUtilization{TargetPercentage: 80}
	}
}

func SetDefaults_ReplicaSet(obj *ReplicaSet) {
	labels := obj.Spec.Template.Labels

	// TODO: support templates defined elsewhere when we support them in the API
	if labels != nil {
		if obj.Spec.Selector == nil {
			obj.Spec.Selector = &LabelSelector{
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
