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

package selfhosting

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestMutatePodSpec(t *testing.T) {
	var tests = []struct {
		component string
		podSpec   *v1.PodSpec
		expected  v1.PodSpec
	}{
		{
			component: kubeadmconstants.KubeAPIServer,
			podSpec:   &v1.PodSpec{},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
		{
			component: kubeadmconstants.KubeControllerManager,
			podSpec:   &v1.PodSpec{},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
		{
			component: kubeadmconstants.KubeScheduler,
			podSpec:   &v1.PodSpec{},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
	}

	cfg := &kubeadmapi.MasterConfiguration{}
	for _, rt := range tests {
		mutatePodSpec(cfg, rt.component, rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed mutatePodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestAddNodeSelectorToPodSpec(t *testing.T) {
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
			},
		},
		{
			podSpec: &v1.PodSpec{
				NodeSelector: map[string]string{
					"foo": "bar",
				},
			},
			expected: v1.PodSpec{
				NodeSelector: map[string]string{
					"foo": "bar",
					kubeadmconstants.LabelNodeRoleMaster: "",
				},
			},
		},
	}

	cfg := &kubeadmapi.MasterConfiguration{}
	for _, rt := range tests {
		addNodeSelectorToPodSpec(cfg, rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed addNodeSelectorToPodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetMasterTolerationOnPodSpec(t *testing.T) {
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{},
			expected: v1.PodSpec{
				Tolerations: []v1.Toleration{
					kubeadmconstants.MasterToleration,
				},
			},
		},
		{
			podSpec: &v1.PodSpec{
				Tolerations: []v1.Toleration{
					{Key: "foo", Value: "bar"},
				},
			},
			expected: v1.PodSpec{
				Tolerations: []v1.Toleration{
					{Key: "foo", Value: "bar"},
					kubeadmconstants.MasterToleration,
				},
			},
		},
	}

	cfg := &kubeadmapi.MasterConfiguration{}
	for _, rt := range tests {
		setMasterTolerationOnPodSpec(cfg, rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setMasterTolerationOnPodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}

func TestSetRightDNSPolicyOnPodSpec(t *testing.T) {
	var tests = []struct {
		podSpec  *v1.PodSpec
		expected v1.PodSpec
	}{
		{
			podSpec: &v1.PodSpec{},
			expected: v1.PodSpec{
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
		{
			podSpec: &v1.PodSpec{
				DNSPolicy: v1.DNSClusterFirst,
			},
			expected: v1.PodSpec{
				DNSPolicy: v1.DNSClusterFirstWithHostNet,
			},
		},
	}

	cfg := &kubeadmapi.MasterConfiguration{}
	for _, rt := range tests {
		setRightDNSPolicyOnPodSpec(cfg, rt.podSpec)

		if !reflect.DeepEqual(*rt.podSpec, rt.expected) {
			t.Errorf("failed setRightDNSPolicyOnPodSpec:\nexpected:\n%v\nsaw:\n%v", rt.expected, *rt.podSpec)
		}
	}
}
