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

	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/networking"
)

func Convert_autoscaling_ScaleStatus_To_v1beta1_ScaleStatus(in *autoscaling.ScaleStatus, out *extensionsv1beta1.ScaleStatus, s conversion.Scope) error {
	out.Replicas = int32(in.Replicas)
	out.TargetSelector = in.Selector

	out.Selector = nil
	selector, err := metav1.ParseToLabelSelector(in.Selector)
	if err != nil {
		return fmt.Errorf("failed to parse selector: %v", err)
	}
	if len(selector.MatchExpressions) == 0 {
		out.Selector = selector.MatchLabels
	}

	return nil
}

func Convert_v1beta1_ScaleStatus_To_autoscaling_ScaleStatus(in *extensionsv1beta1.ScaleStatus, out *autoscaling.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas

	if in.TargetSelector != "" {
		out.Selector = in.TargetSelector
	} else if in.Selector != nil {
		set := labels.Set{}
		for key, val := range in.Selector {
			set[key] = val
		}
		out.Selector = labels.SelectorFromSet(set).String()
	} else {
		out.Selector = ""
	}
	return nil
}

func Convert_apps_DeploymentSpec_To_v1beta1_DeploymentSpec(in *apps.DeploymentSpec, out *extensionsv1beta1.DeploymentSpec, s conversion.Scope) error {
	out.Replicas = &in.Replicas
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_apps_DeploymentStrategy_To_v1beta1_DeploymentStrategy(&in.Strategy, &out.Strategy, s); err != nil {
		return err
	}
	if in.RevisionHistoryLimit != nil {
		out.RevisionHistoryLimit = new(int32)
		*out.RevisionHistoryLimit = int32(*in.RevisionHistoryLimit)
	}
	out.MinReadySeconds = int32(in.MinReadySeconds)
	out.Paused = in.Paused
	if in.RollbackTo != nil {
		out.RollbackTo = new(extensionsv1beta1.RollbackConfig)
		out.RollbackTo.Revision = int64(in.RollbackTo.Revision)
	} else {
		out.RollbackTo = nil
	}
	if in.ProgressDeadlineSeconds != nil {
		out.ProgressDeadlineSeconds = new(int32)
		*out.ProgressDeadlineSeconds = *in.ProgressDeadlineSeconds
	}
	return nil
}

func Convert_v1beta1_DeploymentSpec_To_apps_DeploymentSpec(in *extensionsv1beta1.DeploymentSpec, out *apps.DeploymentSpec, s conversion.Scope) error {
	if in.Replicas != nil {
		out.Replicas = *in.Replicas
	}
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_DeploymentStrategy_To_apps_DeploymentStrategy(&in.Strategy, &out.Strategy, s); err != nil {
		return err
	}
	out.RevisionHistoryLimit = in.RevisionHistoryLimit
	out.MinReadySeconds = in.MinReadySeconds
	out.Paused = in.Paused
	if in.RollbackTo != nil {
		out.RollbackTo = new(apps.RollbackConfig)
		out.RollbackTo.Revision = in.RollbackTo.Revision
	} else {
		out.RollbackTo = nil
	}
	if in.ProgressDeadlineSeconds != nil {
		out.ProgressDeadlineSeconds = new(int32)
		*out.ProgressDeadlineSeconds = *in.ProgressDeadlineSeconds
	}
	return nil
}

func Convert_apps_DeploymentStrategy_To_v1beta1_DeploymentStrategy(in *apps.DeploymentStrategy, out *extensionsv1beta1.DeploymentStrategy, s conversion.Scope) error {
	out.Type = extensionsv1beta1.DeploymentStrategyType(in.Type)
	if in.RollingUpdate != nil {
		out.RollingUpdate = new(extensionsv1beta1.RollingUpdateDeployment)
		if err := Convert_apps_RollingUpdateDeployment_To_v1beta1_RollingUpdateDeployment(in.RollingUpdate, out.RollingUpdate, s); err != nil {
			return err
		}
	} else {
		out.RollingUpdate = nil
	}
	return nil
}

func Convert_v1beta1_DeploymentStrategy_To_apps_DeploymentStrategy(in *extensionsv1beta1.DeploymentStrategy, out *apps.DeploymentStrategy, s conversion.Scope) error {
	out.Type = apps.DeploymentStrategyType(in.Type)
	if in.RollingUpdate != nil {
		out.RollingUpdate = new(apps.RollingUpdateDeployment)
		if err := Convert_v1beta1_RollingUpdateDeployment_To_apps_RollingUpdateDeployment(in.RollingUpdate, out.RollingUpdate, s); err != nil {
			return err
		}
	} else {
		out.RollingUpdate = nil
	}
	return nil
}

func Convert_apps_RollingUpdateDeployment_To_v1beta1_RollingUpdateDeployment(in *apps.RollingUpdateDeployment, out *extensionsv1beta1.RollingUpdateDeployment, s conversion.Scope) error {
	if out.MaxUnavailable == nil {
		out.MaxUnavailable = &intstr.IntOrString{}
	}
	if err := s.Convert(&in.MaxUnavailable, out.MaxUnavailable, 0); err != nil {
		return err
	}
	if out.MaxSurge == nil {
		out.MaxSurge = &intstr.IntOrString{}
	}
	if err := s.Convert(&in.MaxSurge, out.MaxSurge, 0); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_RollingUpdateDeployment_To_apps_RollingUpdateDeployment(in *extensionsv1beta1.RollingUpdateDeployment, out *apps.RollingUpdateDeployment, s conversion.Scope) error {
	if err := s.Convert(in.MaxUnavailable, &out.MaxUnavailable, 0); err != nil {
		return err
	}
	if err := s.Convert(in.MaxSurge, &out.MaxSurge, 0); err != nil {
		return err
	}
	return nil
}

func Convert_apps_RollingUpdateDaemonSet_To_v1beta1_RollingUpdateDaemonSet(in *apps.RollingUpdateDaemonSet, out *extensionsv1beta1.RollingUpdateDaemonSet, s conversion.Scope) error {
	if out.MaxUnavailable == nil {
		out.MaxUnavailable = &intstr.IntOrString{}
	}
	if err := s.Convert(&in.MaxUnavailable, out.MaxUnavailable, 0); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_RollingUpdateDaemonSet_To_apps_RollingUpdateDaemonSet(in *extensionsv1beta1.RollingUpdateDaemonSet, out *apps.RollingUpdateDaemonSet, s conversion.Scope) error {
	if err := s.Convert(in.MaxUnavailable, &out.MaxUnavailable, 0); err != nil {
		return err
	}
	return nil
}

func Convert_apps_ReplicaSetSpec_To_v1beta1_ReplicaSetSpec(in *apps.ReplicaSetSpec, out *extensionsv1beta1.ReplicaSetSpec, s conversion.Scope) error {
	out.Replicas = new(int32)
	*out.Replicas = int32(in.Replicas)
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_ReplicaSetSpec_To_apps_ReplicaSetSpec(in *extensionsv1beta1.ReplicaSetSpec, out *apps.ReplicaSetSpec, s conversion.Scope) error {
	if in.Replicas != nil {
		out.Replicas = *in.Replicas
	}
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_NetworkPolicy_To_networking_NetworkPolicy(in *extensionsv1beta1.NetworkPolicy, out *networking.NetworkPolicy, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	return Convert_v1beta1_NetworkPolicySpec_To_networking_NetworkPolicySpec(&in.Spec, &out.Spec, s)
}

func Convert_networking_NetworkPolicy_To_v1beta1_NetworkPolicy(in *networking.NetworkPolicy, out *extensionsv1beta1.NetworkPolicy, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	return Convert_networking_NetworkPolicySpec_To_v1beta1_NetworkPolicySpec(&in.Spec, &out.Spec, s)
}

func Convert_v1beta1_NetworkPolicySpec_To_networking_NetworkPolicySpec(in *extensionsv1beta1.NetworkPolicySpec, out *networking.NetworkPolicySpec, s conversion.Scope) error {
	if err := s.Convert(&in.PodSelector, &out.PodSelector, 0); err != nil {
		return err
	}
	out.Ingress = make([]networking.NetworkPolicyIngressRule, len(in.Ingress))
	for i := range in.Ingress {
		if err := Convert_v1beta1_NetworkPolicyIngressRule_To_networking_NetworkPolicyIngressRule(&in.Ingress[i], &out.Ingress[i], s); err != nil {
			return err
		}
	}
	out.Egress = make([]networking.NetworkPolicyEgressRule, len(in.Egress))
	for i := range in.Egress {
		if err := Convert_v1beta1_NetworkPolicyEgressRule_To_networking_NetworkPolicyEgressRule(&in.Egress[i], &out.Egress[i], s); err != nil {
			return err
		}
	}
	if in.PolicyTypes != nil {
		in, out := &in.PolicyTypes, &out.PolicyTypes
		*out = make([]networking.PolicyType, len(*in))
		for i := range *in {
			if err := s.Convert(&(*in)[i], &(*out)[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func Convert_networking_NetworkPolicySpec_To_v1beta1_NetworkPolicySpec(in *networking.NetworkPolicySpec, out *extensionsv1beta1.NetworkPolicySpec, s conversion.Scope) error {
	if err := s.Convert(&in.PodSelector, &out.PodSelector, 0); err != nil {
		return err
	}
	out.Ingress = make([]extensionsv1beta1.NetworkPolicyIngressRule, len(in.Ingress))
	for i := range in.Ingress {
		if err := Convert_networking_NetworkPolicyIngressRule_To_v1beta1_NetworkPolicyIngressRule(&in.Ingress[i], &out.Ingress[i], s); err != nil {
			return err
		}
	}
	out.Egress = make([]extensionsv1beta1.NetworkPolicyEgressRule, len(in.Egress))
	for i := range in.Egress {
		if err := Convert_networking_NetworkPolicyEgressRule_To_v1beta1_NetworkPolicyEgressRule(&in.Egress[i], &out.Egress[i], s); err != nil {
			return err
		}
	}
	if in.PolicyTypes != nil {
		in, out := &in.PolicyTypes, &out.PolicyTypes
		*out = make([]extensionsv1beta1.PolicyType, len(*in))
		for i := range *in {
			if err := s.Convert(&(*in)[i], &(*out)[i], 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func Convert_v1beta1_NetworkPolicyIngressRule_To_networking_NetworkPolicyIngressRule(in *extensionsv1beta1.NetworkPolicyIngressRule, out *networking.NetworkPolicyIngressRule, s conversion.Scope) error {
	out.Ports = make([]networking.NetworkPolicyPort, len(in.Ports))
	for i := range in.Ports {
		if err := Convert_v1beta1_NetworkPolicyPort_To_networking_NetworkPolicyPort(&in.Ports[i], &out.Ports[i], s); err != nil {
			return err
		}
	}
	if in.From != nil {
		out.From = make([]networking.NetworkPolicyPeer, len(in.From))
		for i := range in.From {
			if err := Convert_v1beta1_NetworkPolicyPeer_To_networking_NetworkPolicyPeer(&in.From[i], &out.From[i], s); err != nil {
				return err
			}
		}
	}
	return nil
}

func Convert_networking_NetworkPolicyIngressRule_To_v1beta1_NetworkPolicyIngressRule(in *networking.NetworkPolicyIngressRule, out *extensionsv1beta1.NetworkPolicyIngressRule, s conversion.Scope) error {
	out.Ports = make([]extensionsv1beta1.NetworkPolicyPort, len(in.Ports))
	for i := range in.Ports {
		if err := Convert_networking_NetworkPolicyPort_To_v1beta1_NetworkPolicyPort(&in.Ports[i], &out.Ports[i], s); err != nil {
			return err
		}
	}
	if in.From != nil {
		out.From = make([]extensionsv1beta1.NetworkPolicyPeer, len(in.From))
		for i := range in.From {
			if err := Convert_networking_NetworkPolicyPeer_To_v1beta1_NetworkPolicyPeer(&in.From[i], &out.From[i], s); err != nil {
				return err
			}
		}
	}
	return nil
}

func Convert_v1beta1_NetworkPolicyEgressRule_To_networking_NetworkPolicyEgressRule(in *extensionsv1beta1.NetworkPolicyEgressRule, out *networking.NetworkPolicyEgressRule, s conversion.Scope) error {
	out.Ports = make([]networking.NetworkPolicyPort, len(in.Ports))
	for i := range in.Ports {
		if err := Convert_v1beta1_NetworkPolicyPort_To_networking_NetworkPolicyPort(&in.Ports[i], &out.Ports[i], s); err != nil {
			return err
		}
	}
	out.To = make([]networking.NetworkPolicyPeer, len(in.To))
	for i := range in.To {
		if err := Convert_v1beta1_NetworkPolicyPeer_To_networking_NetworkPolicyPeer(&in.To[i], &out.To[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_networking_NetworkPolicyEgressRule_To_v1beta1_NetworkPolicyEgressRule(in *networking.NetworkPolicyEgressRule, out *extensionsv1beta1.NetworkPolicyEgressRule, s conversion.Scope) error {
	out.Ports = make([]extensionsv1beta1.NetworkPolicyPort, len(in.Ports))
	for i := range in.Ports {
		if err := Convert_networking_NetworkPolicyPort_To_v1beta1_NetworkPolicyPort(&in.Ports[i], &out.Ports[i], s); err != nil {
			return err
		}
	}
	out.To = make([]extensionsv1beta1.NetworkPolicyPeer, len(in.To))
	for i := range in.To {
		if err := Convert_networking_NetworkPolicyPeer_To_v1beta1_NetworkPolicyPeer(&in.To[i], &out.To[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_v1beta1_NetworkPolicyPeer_To_networking_NetworkPolicyPeer(in *extensionsv1beta1.NetworkPolicyPeer, out *networking.NetworkPolicyPeer, s conversion.Scope) error {
	if in.PodSelector != nil {
		out.PodSelector = new(metav1.LabelSelector)
		if err := s.Convert(in.PodSelector, out.PodSelector, 0); err != nil {
			return err
		}
	} else {
		out.PodSelector = nil
	}
	if in.NamespaceSelector != nil {
		out.NamespaceSelector = new(metav1.LabelSelector)
		if err := s.Convert(in.NamespaceSelector, out.NamespaceSelector, 0); err != nil {
			return err
		}
	} else {
		out.NamespaceSelector = nil
	}
	if in.IPBlock != nil {
		out.IPBlock = new(networking.IPBlock)
		if err := s.Convert(in.IPBlock, out.IPBlock, 0); err != nil {
			return err
		}
	} else {
		out.IPBlock = nil
	}
	return nil
}

func Convert_networking_NetworkPolicyPeer_To_v1beta1_NetworkPolicyPeer(in *networking.NetworkPolicyPeer, out *extensionsv1beta1.NetworkPolicyPeer, s conversion.Scope) error {
	if in.PodSelector != nil {
		out.PodSelector = new(metav1.LabelSelector)
		if err := s.Convert(in.PodSelector, out.PodSelector, 0); err != nil {
			return err
		}
	} else {
		out.PodSelector = nil
	}
	if in.NamespaceSelector != nil {
		out.NamespaceSelector = new(metav1.LabelSelector)
		if err := s.Convert(in.NamespaceSelector, out.NamespaceSelector, 0); err != nil {
			return err
		}
	} else {
		out.NamespaceSelector = nil
	}
	if in.IPBlock != nil {
		out.IPBlock = new(extensionsv1beta1.IPBlock)
		if err := s.Convert(in.IPBlock, out.IPBlock, 0); err != nil {
			return err
		}
	} else {
		out.IPBlock = nil
	}
	return nil
}

func Convert_v1beta1_IPBlock_To_networking_IPBlock(in *extensionsv1beta1.IPBlock, out *networking.IPBlock, s conversion.Scope) error {
	out.CIDR = in.CIDR

	out.Except = make([]string, len(in.Except))
	copy(out.Except, in.Except)
	return nil
}

func Convert_networking_IPBlock_To_v1beta1_IPBlock(in *networking.IPBlock, out *extensionsv1beta1.IPBlock, s conversion.Scope) error {
	out.CIDR = in.CIDR

	out.Except = make([]string, len(in.Except))
	copy(out.Except, in.Except)
	return nil
}

func Convert_v1beta1_NetworkPolicyPort_To_networking_NetworkPolicyPort(in *extensionsv1beta1.NetworkPolicyPort, out *networking.NetworkPolicyPort, s conversion.Scope) error {
	if in.Protocol != nil {
		out.Protocol = new(api.Protocol)
		*out.Protocol = api.Protocol(*in.Protocol)
	} else {
		out.Protocol = nil
	}
	out.Port = in.Port
	return nil
}

func Convert_networking_NetworkPolicyPort_To_v1beta1_NetworkPolicyPort(in *networking.NetworkPolicyPort, out *extensionsv1beta1.NetworkPolicyPort, s conversion.Scope) error {
	if in.Protocol != nil {
		out.Protocol = new(v1.Protocol)
		*out.Protocol = v1.Protocol(*in.Protocol)
	} else {
		out.Protocol = nil
	}
	out.Port = in.Port
	return nil
}

func Convert_v1beta1_NetworkPolicyList_To_networking_NetworkPolicyList(in *extensionsv1beta1.NetworkPolicyList, out *networking.NetworkPolicyList, s conversion.Scope) error {
	out.ListMeta = in.ListMeta
	out.Items = make([]networking.NetworkPolicy, len(in.Items))
	for i := range in.Items {
		if err := Convert_v1beta1_NetworkPolicy_To_networking_NetworkPolicy(&in.Items[i], &out.Items[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_networking_NetworkPolicyList_To_v1beta1_NetworkPolicyList(in *networking.NetworkPolicyList, out *extensionsv1beta1.NetworkPolicyList, s conversion.Scope) error {
	out.ListMeta = in.ListMeta
	out.Items = make([]extensionsv1beta1.NetworkPolicy, len(in.Items))
	for i := range in.Items {
		if err := Convert_networking_NetworkPolicy_To_v1beta1_NetworkPolicy(&in.Items[i], &out.Items[i], s); err != nil {
			return err
		}
	}
	return nil
}
