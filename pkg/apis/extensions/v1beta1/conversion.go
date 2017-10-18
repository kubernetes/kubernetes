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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/networking"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_extensions_ScaleStatus_To_v1beta1_ScaleStatus,
		Convert_v1beta1_ScaleStatus_To_extensions_ScaleStatus,
		Convert_extensions_DeploymentSpec_To_v1beta1_DeploymentSpec,
		Convert_v1beta1_DeploymentSpec_To_extensions_DeploymentSpec,
		Convert_extensions_DeploymentStrategy_To_v1beta1_DeploymentStrategy,
		Convert_v1beta1_DeploymentStrategy_To_extensions_DeploymentStrategy,
		Convert_extensions_RollingUpdateDeployment_To_v1beta1_RollingUpdateDeployment,
		Convert_v1beta1_RollingUpdateDeployment_To_extensions_RollingUpdateDeployment,
		Convert_extensions_RollingUpdateDaemonSet_To_v1beta1_RollingUpdateDaemonSet,
		Convert_v1beta1_RollingUpdateDaemonSet_To_extensions_RollingUpdateDaemonSet,
		Convert_extensions_ReplicaSetSpec_To_v1beta1_ReplicaSetSpec,
		Convert_v1beta1_ReplicaSetSpec_To_extensions_ReplicaSetSpec,
		Convert_v1beta1_NetworkPolicy_To_networking_NetworkPolicy,
		Convert_networking_NetworkPolicy_To_v1beta1_NetworkPolicy,
		Convert_v1beta1_NetworkPolicyIngressRule_To_networking_NetworkPolicyIngressRule,
		Convert_networking_NetworkPolicyIngressRule_To_v1beta1_NetworkPolicyIngressRule,
		Convert_v1beta1_NetworkPolicyList_To_networking_NetworkPolicyList,
		Convert_networking_NetworkPolicyList_To_v1beta1_NetworkPolicyList,
		Convert_v1beta1_NetworkPolicyPeer_To_networking_NetworkPolicyPeer,
		Convert_networking_NetworkPolicyPeer_To_v1beta1_NetworkPolicyPeer,
		Convert_v1beta1_NetworkPolicyPort_To_networking_NetworkPolicyPort,
		Convert_networking_NetworkPolicyPort_To_v1beta1_NetworkPolicyPort,
		Convert_v1beta1_NetworkPolicySpec_To_networking_NetworkPolicySpec,
		Convert_networking_NetworkPolicySpec_To_v1beta1_NetworkPolicySpec,
		Convert_extensions_PodSecurityPolicySpec_To_v1beta1_PodSecurityPolicySpec,
		Convert_v1beta1_IPBlock_To_networking_IPBlock,
		Convert_networking_IPBlock_To_v1beta1_IPBlock,
		Convert_networking_NetworkPolicyEgressRule_To_v1beta1_NetworkPolicyEgressRule,
		Convert_v1beta1_NetworkPolicyEgressRule_To_networking_NetworkPolicyEgressRule,
	)
	if err != nil {
		return err
	}

	return nil
}

func Convert_extensions_ScaleStatus_To_v1beta1_ScaleStatus(in *extensions.ScaleStatus, out *extensionsv1beta1.ScaleStatus, s conversion.Scope) error {
	out.Replicas = int32(in.Replicas)

	out.Selector = nil
	out.TargetSelector = ""
	if in.Selector != nil {
		if in.Selector.MatchExpressions == nil || len(in.Selector.MatchExpressions) == 0 {
			out.Selector = in.Selector.MatchLabels
		}

		selector, err := metav1.LabelSelectorAsSelector(in.Selector)
		if err != nil {
			return fmt.Errorf("invalid label selector: %v", err)
		}
		out.TargetSelector = selector.String()
	}
	return nil
}

func Convert_v1beta1_ScaleStatus_To_extensions_ScaleStatus(in *extensionsv1beta1.ScaleStatus, out *extensions.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas

	// Normally when 2 fields map to the same internal value we favor the old field, since
	// old clients can't be expected to know about new fields but clients that know about the
	// new field can be expected to know about the old field (though that's not quite true, due
	// to kubectl apply). However, these fields are readonly, so any non-nil value should work.
	if in.TargetSelector != "" {
		labelSelector, err := metav1.ParseToLabelSelector(in.TargetSelector)
		if err != nil {
			out.Selector = nil
			return fmt.Errorf("failed to parse target selector: %v", err)
		}
		out.Selector = labelSelector
	} else if in.Selector != nil {
		out.Selector = new(metav1.LabelSelector)
		selector := make(map[string]string)
		for key, val := range in.Selector {
			selector[key] = val
		}
		out.Selector.MatchLabels = selector
	} else {
		out.Selector = nil
	}
	return nil
}

func Convert_extensions_DeploymentSpec_To_v1beta1_DeploymentSpec(in *extensions.DeploymentSpec, out *extensionsv1beta1.DeploymentSpec, s conversion.Scope) error {
	out.Replicas = &in.Replicas
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_extensions_DeploymentStrategy_To_v1beta1_DeploymentStrategy(&in.Strategy, &out.Strategy, s); err != nil {
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

func Convert_v1beta1_DeploymentSpec_To_extensions_DeploymentSpec(in *extensionsv1beta1.DeploymentSpec, out *extensions.DeploymentSpec, s conversion.Scope) error {
	if in.Replicas != nil {
		out.Replicas = *in.Replicas
	}
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_DeploymentStrategy_To_extensions_DeploymentStrategy(&in.Strategy, &out.Strategy, s); err != nil {
		return err
	}
	out.RevisionHistoryLimit = in.RevisionHistoryLimit
	out.MinReadySeconds = in.MinReadySeconds
	out.Paused = in.Paused
	if in.RollbackTo != nil {
		out.RollbackTo = new(extensions.RollbackConfig)
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

func Convert_extensions_DeploymentStrategy_To_v1beta1_DeploymentStrategy(in *extensions.DeploymentStrategy, out *extensionsv1beta1.DeploymentStrategy, s conversion.Scope) error {
	out.Type = extensionsv1beta1.DeploymentStrategyType(in.Type)
	if in.RollingUpdate != nil {
		out.RollingUpdate = new(extensionsv1beta1.RollingUpdateDeployment)
		if err := Convert_extensions_RollingUpdateDeployment_To_v1beta1_RollingUpdateDeployment(in.RollingUpdate, out.RollingUpdate, s); err != nil {
			return err
		}
	} else {
		out.RollingUpdate = nil
	}
	return nil
}

func Convert_v1beta1_DeploymentStrategy_To_extensions_DeploymentStrategy(in *extensionsv1beta1.DeploymentStrategy, out *extensions.DeploymentStrategy, s conversion.Scope) error {
	out.Type = extensions.DeploymentStrategyType(in.Type)
	if in.RollingUpdate != nil {
		out.RollingUpdate = new(extensions.RollingUpdateDeployment)
		if err := Convert_v1beta1_RollingUpdateDeployment_To_extensions_RollingUpdateDeployment(in.RollingUpdate, out.RollingUpdate, s); err != nil {
			return err
		}
	} else {
		out.RollingUpdate = nil
	}
	return nil
}

func Convert_extensions_RollingUpdateDeployment_To_v1beta1_RollingUpdateDeployment(in *extensions.RollingUpdateDeployment, out *extensionsv1beta1.RollingUpdateDeployment, s conversion.Scope) error {
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

func Convert_v1beta1_RollingUpdateDeployment_To_extensions_RollingUpdateDeployment(in *extensionsv1beta1.RollingUpdateDeployment, out *extensions.RollingUpdateDeployment, s conversion.Scope) error {
	if err := s.Convert(in.MaxUnavailable, &out.MaxUnavailable, 0); err != nil {
		return err
	}
	if err := s.Convert(in.MaxSurge, &out.MaxSurge, 0); err != nil {
		return err
	}
	return nil
}

func Convert_extensions_RollingUpdateDaemonSet_To_v1beta1_RollingUpdateDaemonSet(in *extensions.RollingUpdateDaemonSet, out *extensionsv1beta1.RollingUpdateDaemonSet, s conversion.Scope) error {
	if out.MaxUnavailable == nil {
		out.MaxUnavailable = &intstr.IntOrString{}
	}
	if err := s.Convert(&in.MaxUnavailable, out.MaxUnavailable, 0); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_RollingUpdateDaemonSet_To_extensions_RollingUpdateDaemonSet(in *extensionsv1beta1.RollingUpdateDaemonSet, out *extensions.RollingUpdateDaemonSet, s conversion.Scope) error {
	if err := s.Convert(in.MaxUnavailable, &out.MaxUnavailable, 0); err != nil {
		return err
	}
	return nil
}

func Convert_extensions_ReplicaSetSpec_To_v1beta1_ReplicaSetSpec(in *extensions.ReplicaSetSpec, out *extensionsv1beta1.ReplicaSetSpec, s conversion.Scope) error {
	out.Replicas = new(int32)
	*out.Replicas = int32(in.Replicas)
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_ReplicaSetSpec_To_extensions_ReplicaSetSpec(in *extensionsv1beta1.ReplicaSetSpec, out *extensions.ReplicaSetSpec, s conversion.Scope) error {
	if in.Replicas != nil {
		out.Replicas = *in.Replicas
	}
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
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

func Convert_extensions_PodSecurityPolicySpec_To_v1beta1_PodSecurityPolicySpec(in *extensions.PodSecurityPolicySpec, out *extensionsv1beta1.PodSecurityPolicySpec, s conversion.Scope) error {
	return autoConvert_extensions_PodSecurityPolicySpec_To_v1beta1_PodSecurityPolicySpec(in, out, s)
}
