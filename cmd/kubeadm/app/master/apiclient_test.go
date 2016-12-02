/*
Copyright 2016 The Kubernetes Authors.

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

package master

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
)

func TestCreateClientAndWaitForAPI(t *testing.T) {
	cfg := &kubeadmapi.MasterConfiguration{
		Networking: kubeadm.Networking{DNSDomain: "localhost"},
	}
	fmt.Println(cfg)

}

func TestStandardLabels(t *testing.T) {
	var tests = []struct {
		n        string
		expected string
	}{
		{
			n:        "foo",
			expected: "foo",
		},
		{
			n:        "bar",
			expected: "bar",
		},
	}

	for _, rt := range tests {
		actual := standardLabels(rt.n)
		if actual["component"] != rt.expected {
			t.Errorf(
				"failed standardLabels:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual["component"],
			)
		}
		if actual["name"] != rt.expected {
			t.Errorf(
				"failed standardLabels:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual["name"],
			)
		}
		if actual["k8s-app"] != rt.expected {
			t.Errorf(
				"failed standardLabels:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual["k8s-app"],
			)
		}
	}
}

func TestNewDaemonSet(t *testing.T) {
	var tests = []struct {
		dn       string
		expected string
	}{
		{
			dn:       "foo",
			expected: "foo",
		},
		{
			dn:       "bar",
			expected: "bar",
		},
	}

	for _, rt := range tests {
		p := apiv1.PodSpec{}
		actual := NewDaemonSet(rt.dn, p)
		if actual.Spec.Selector.MatchLabels["k8s-app"] != rt.expected {
			t.Errorf(
				"failed NewDaemonSet:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.Spec.Selector.MatchLabels["k8s-app"],
			)
		}
		if actual.Spec.Selector.MatchLabels["component"] != rt.expected {
			t.Errorf(
				"failed NewDaemonSet:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.Spec.Selector.MatchLabels["component"],
			)
		}
		if actual.Spec.Selector.MatchLabels["name"] != rt.expected {
			t.Errorf(
				"failed NewDaemonSet:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.Spec.Selector.MatchLabels["name"],
			)
		}
	}
}

func TestNewService(t *testing.T) {
	var tests = []struct {
		dn       string
		expected string
	}{
		{
			dn:       "foo",
			expected: "foo",
		},
		{
			dn:       "bar",
			expected: "bar",
		},
	}

	for _, rt := range tests {
		p := apiv1.ServiceSpec{}
		actual := NewService(rt.dn, p)
		if actual.ObjectMeta.Labels["k8s-app"] != rt.expected {
			t.Errorf(
				"failed NewService:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.ObjectMeta.Labels["k8s-app"],
			)
		}
		if actual.ObjectMeta.Labels["component"] != rt.expected {
			t.Errorf(
				"failed NewService:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.ObjectMeta.Labels["component"],
			)
		}
		if actual.ObjectMeta.Labels["name"] != rt.expected {
			t.Errorf(
				"failed NewService:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.ObjectMeta.Labels["name"],
			)
		}
	}
}

func TestNewDeployment(t *testing.T) {
	var tests = []struct {
		dn       string
		expected string
	}{
		{
			dn:       "foo",
			expected: "foo",
		},
		{
			dn:       "bar",
			expected: "bar",
		},
	}

	for _, rt := range tests {
		p := apiv1.PodSpec{}
		actual := NewDeployment(rt.dn, 1, p)
		if actual.Spec.Selector.MatchLabels["k8s-app"] != rt.expected {
			t.Errorf(
				"failed NewDeployment:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.Spec.Selector.MatchLabels["k8s-app"],
			)
		}
		if actual.Spec.Selector.MatchLabels["component"] != rt.expected {
			t.Errorf(
				"failed NewDeployment:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.Spec.Selector.MatchLabels["component"],
			)
		}
		if actual.Spec.Selector.MatchLabels["name"] != rt.expected {
			t.Errorf(
				"failed NewDeployment:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual.Spec.Selector.MatchLabels["name"],
			)
		}
	}
}
