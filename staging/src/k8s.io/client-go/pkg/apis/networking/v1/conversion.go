/*
Copyright 2017 The Kubernetes Authors.

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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/pkg/apis/extensions"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	return scheme.AddConversionFuncs(
		Convert_v1_NetworkPolicy_To_extensions_NetworkPolicy,
		Convert_extensions_NetworkPolicy_To_v1_NetworkPolicy,
		Convert_v1_NetworkPolicyIngressRule_To_extensions_NetworkPolicyIngressRule,
		Convert_extensions_NetworkPolicyIngressRule_To_v1_NetworkPolicyIngressRule,
		Convert_v1_NetworkPolicyList_To_extensions_NetworkPolicyList,
		Convert_extensions_NetworkPolicyList_To_v1_NetworkPolicyList,
		Convert_v1_NetworkPolicyPeer_To_extensions_NetworkPolicyPeer,
		Convert_extensions_NetworkPolicyPeer_To_v1_NetworkPolicyPeer,
		Convert_v1_NetworkPolicyPort_To_extensions_NetworkPolicyPort,
		Convert_extensions_NetworkPolicyPort_To_v1_NetworkPolicyPort,
		Convert_v1_NetworkPolicySpec_To_extensions_NetworkPolicySpec,
		Convert_extensions_NetworkPolicySpec_To_v1_NetworkPolicySpec,
	)
}

func Convert_v1_NetworkPolicy_To_extensions_NetworkPolicy(in *NetworkPolicy, out *extensions.NetworkPolicy, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	return Convert_v1_NetworkPolicySpec_To_extensions_NetworkPolicySpec(&in.Spec, &out.Spec, s)
}

func Convert_extensions_NetworkPolicy_To_v1_NetworkPolicy(in *extensions.NetworkPolicy, out *NetworkPolicy, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	return Convert_extensions_NetworkPolicySpec_To_v1_NetworkPolicySpec(&in.Spec, &out.Spec, s)
}

func Convert_v1_NetworkPolicySpec_To_extensions_NetworkPolicySpec(in *NetworkPolicySpec, out *extensions.NetworkPolicySpec, s conversion.Scope) error {
	if err := s.Convert(&in.PodSelector, &out.PodSelector, 0); err != nil {
		return err
	}
	out.Ingress = make([]extensions.NetworkPolicyIngressRule, len(in.Ingress))
	for i := range in.Ingress {
		if err := Convert_v1_NetworkPolicyIngressRule_To_extensions_NetworkPolicyIngressRule(&in.Ingress[i], &out.Ingress[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_extensions_NetworkPolicySpec_To_v1_NetworkPolicySpec(in *extensions.NetworkPolicySpec, out *NetworkPolicySpec, s conversion.Scope) error {
	if err := s.Convert(&in.PodSelector, &out.PodSelector, 0); err != nil {
		return err
	}
	out.Ingress = make([]NetworkPolicyIngressRule, len(in.Ingress))
	for i := range in.Ingress {
		if err := Convert_extensions_NetworkPolicyIngressRule_To_v1_NetworkPolicyIngressRule(&in.Ingress[i], &out.Ingress[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_v1_NetworkPolicyIngressRule_To_extensions_NetworkPolicyIngressRule(in *NetworkPolicyIngressRule, out *extensions.NetworkPolicyIngressRule, s conversion.Scope) error {
	out.Ports = make([]extensions.NetworkPolicyPort, len(in.Ports))
	for i := range in.Ports {
		if err := Convert_v1_NetworkPolicyPort_To_extensions_NetworkPolicyPort(&in.Ports[i], &out.Ports[i], s); err != nil {
			return err
		}
	}
	out.From = make([]extensions.NetworkPolicyPeer, len(in.From))
	for i := range in.From {
		if err := Convert_v1_NetworkPolicyPeer_To_extensions_NetworkPolicyPeer(&in.From[i], &out.From[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_extensions_NetworkPolicyIngressRule_To_v1_NetworkPolicyIngressRule(in *extensions.NetworkPolicyIngressRule, out *NetworkPolicyIngressRule, s conversion.Scope) error {
	out.Ports = make([]NetworkPolicyPort, len(in.Ports))
	for i := range in.Ports {
		if err := Convert_extensions_NetworkPolicyPort_To_v1_NetworkPolicyPort(&in.Ports[i], &out.Ports[i], s); err != nil {
			return err
		}
	}
	out.From = make([]NetworkPolicyPeer, len(in.From))
	for i := range in.From {
		if err := Convert_extensions_NetworkPolicyPeer_To_v1_NetworkPolicyPeer(&in.From[i], &out.From[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_v1_NetworkPolicyPeer_To_extensions_NetworkPolicyPeer(in *NetworkPolicyPeer, out *extensions.NetworkPolicyPeer, s conversion.Scope) error {
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
	return nil
}

func Convert_extensions_NetworkPolicyPeer_To_v1_NetworkPolicyPeer(in *extensions.NetworkPolicyPeer, out *NetworkPolicyPeer, s conversion.Scope) error {
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
	return nil
}

func Convert_v1_NetworkPolicyPort_To_extensions_NetworkPolicyPort(in *NetworkPolicyPort, out *extensions.NetworkPolicyPort, s conversion.Scope) error {
	if in.Protocol != nil {
		out.Protocol = new(api.Protocol)
		*out.Protocol = api.Protocol(*in.Protocol)
	} else {
		out.Protocol = nil
	}
	out.Port = in.Port
	return nil
}

func Convert_extensions_NetworkPolicyPort_To_v1_NetworkPolicyPort(in *extensions.NetworkPolicyPort, out *NetworkPolicyPort, s conversion.Scope) error {
	if in.Protocol != nil {
		out.Protocol = new(v1.Protocol)
		*out.Protocol = v1.Protocol(*in.Protocol)
	} else {
		out.Protocol = nil
	}
	out.Port = in.Port
	return nil
}

func Convert_v1_NetworkPolicyList_To_extensions_NetworkPolicyList(in *NetworkPolicyList, out *extensions.NetworkPolicyList, s conversion.Scope) error {
	out.ListMeta = in.ListMeta
	out.Items = make([]extensions.NetworkPolicy, len(in.Items))
	for i := range in.Items {
		if err := Convert_v1_NetworkPolicy_To_extensions_NetworkPolicy(&in.Items[i], &out.Items[i], s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_extensions_NetworkPolicyList_To_v1_NetworkPolicyList(in *extensions.NetworkPolicyList, out *NetworkPolicyList, s conversion.Scope) error {
	out.ListMeta = in.ListMeta
	out.Items = make([]NetworkPolicy, len(in.Items))
	for i := range in.Items {
		if err := Convert_extensions_NetworkPolicy_To_v1_NetworkPolicy(&in.Items[i], &out.Items[i], s); err != nil {
			return err
		}
	}
	return nil
}
