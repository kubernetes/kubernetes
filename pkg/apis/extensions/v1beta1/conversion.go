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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
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

func Convert_v1beta1_NetworkPolicySpec_To_networking_NetworkPolicySpec(in *extensionsv1beta1.NetworkPolicySpec, out *networking.NetworkPolicySpec, s conversion.Scope) error {
	if err := autoConvert_v1beta1_NetworkPolicySpec_To_networking_NetworkPolicySpec(in, out, s); err != nil {
		return err
	}
	if out.Ingress == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Ingress = make([]networking.NetworkPolicyIngressRule, 0)
	}
	if out.Egress == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Egress = make([]networking.NetworkPolicyEgressRule, 0)
	}
	return nil
}

func Convert_networking_NetworkPolicySpec_To_v1beta1_NetworkPolicySpec(in *networking.NetworkPolicySpec, out *extensionsv1beta1.NetworkPolicySpec, s conversion.Scope) error {
	if err := autoConvert_networking_NetworkPolicySpec_To_v1beta1_NetworkPolicySpec(in, out, s); err != nil {
		return err
	}
	if out.Ingress == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Ingress = make([]extensionsv1beta1.NetworkPolicyIngressRule, 0)
	}
	if out.Egress == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Egress = make([]extensionsv1beta1.NetworkPolicyEgressRule, 0)
	}
	return nil
}

func Convert_v1beta1_NetworkPolicyIngressRule_To_networking_NetworkPolicyIngressRule(in *extensionsv1beta1.NetworkPolicyIngressRule, out *networking.NetworkPolicyIngressRule, s conversion.Scope) error {
	if err := autoConvert_v1beta1_NetworkPolicyIngressRule_To_networking_NetworkPolicyIngressRule(in, out, s); err != nil {
		return err
	}
	if out.Ports == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Ports = make([]networking.NetworkPolicyPort, 0)
	}
	return nil
}

func Convert_networking_NetworkPolicyIngressRule_To_v1beta1_NetworkPolicyIngressRule(in *networking.NetworkPolicyIngressRule, out *extensionsv1beta1.NetworkPolicyIngressRule, s conversion.Scope) error {
	if err := autoConvert_networking_NetworkPolicyIngressRule_To_v1beta1_NetworkPolicyIngressRule(in, out, s); err != nil {
		return err
	}
	if out.Ports == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Ports = make([]extensionsv1beta1.NetworkPolicyPort, 0)
	}
	return nil
}

func Convert_v1beta1_NetworkPolicyEgressRule_To_networking_NetworkPolicyEgressRule(in *extensionsv1beta1.NetworkPolicyEgressRule, out *networking.NetworkPolicyEgressRule, s conversion.Scope) error {
	if err := autoConvert_v1beta1_NetworkPolicyEgressRule_To_networking_NetworkPolicyEgressRule(in, out, s); err != nil {
		return err
	}
	if out.Ports == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Ports = make([]networking.NetworkPolicyPort, 0)
	}
	if out.To == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.To = make([]networking.NetworkPolicyPeer, 0)
	}
	return nil
}

func Convert_networking_NetworkPolicyEgressRule_To_v1beta1_NetworkPolicyEgressRule(in *networking.NetworkPolicyEgressRule, out *extensionsv1beta1.NetworkPolicyEgressRule, s conversion.Scope) error {
	if err := autoConvert_networking_NetworkPolicyEgressRule_To_v1beta1_NetworkPolicyEgressRule(in, out, s); err != nil {
		return err
	}
	if out.Ports == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.Ports = make([]extensionsv1beta1.NetworkPolicyPort, 0)
	}
	if out.To == nil {
		// Produce a zero-length non-nil slice for compatibility with previous manual conversion.
		out.To = make([]extensionsv1beta1.NetworkPolicyPeer, 0)
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

func Convert_v1beta1_IngressBackend_To_networking_IngressBackend(in *extensionsv1beta1.IngressBackend, out *networking.IngressBackend, s conversion.Scope) error {
	if err := autoConvert_v1beta1_IngressBackend_To_networking_IngressBackend(in, out, s); err != nil {
		return err
	}
	if len(in.ServiceName) > 0 || in.ServicePort.IntVal != 0 || in.ServicePort.StrVal != "" || in.ServicePort.Type == intstr.String {
		out.Service = &networking.IngressServiceBackend{}
		out.Service.Name = in.ServiceName
		out.Service.Port.Name = in.ServicePort.StrVal
		out.Service.Port.Number = in.ServicePort.IntVal
	}
	return nil
}

func Convert_networking_IngressBackend_To_v1beta1_IngressBackend(in *networking.IngressBackend, out *extensionsv1beta1.IngressBackend, s conversion.Scope) error {
	if err := autoConvert_networking_IngressBackend_To_v1beta1_IngressBackend(in, out, s); err != nil {
		return err
	}
	if in.Service != nil {
		out.ServiceName = in.Service.Name
		if len(in.Service.Port.Name) > 0 {
			out.ServicePort = intstr.FromString(in.Service.Port.Name)
		} else {
			out.ServicePort = intstr.FromInt32(in.Service.Port.Number)
		}
	}
	return nil
}

func Convert_v1beta1_IngressSpec_To_networking_IngressSpec(in *extensionsv1beta1.IngressSpec, out *networking.IngressSpec, s conversion.Scope) error {
	if err := autoConvert_v1beta1_IngressSpec_To_networking_IngressSpec(in, out, s); err != nil {
		return err
	}
	if in.Backend != nil {
		out.DefaultBackend = &networking.IngressBackend{}
		if err := Convert_v1beta1_IngressBackend_To_networking_IngressBackend(in.Backend, out.DefaultBackend, s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_networking_IngressSpec_To_v1beta1_IngressSpec(in *networking.IngressSpec, out *extensionsv1beta1.IngressSpec, s conversion.Scope) error {
	if err := autoConvert_networking_IngressSpec_To_v1beta1_IngressSpec(in, out, s); err != nil {
		return err
	}
	if in.DefaultBackend != nil {
		out.Backend = &extensionsv1beta1.IngressBackend{}
		if err := Convert_networking_IngressBackend_To_v1beta1_IngressBackend(in.DefaultBackend, out.Backend, s); err != nil {
			return err
		}
	}
	return nil
}
