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

package images

import (
	"fmt"
	"runtime"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

type getCoreImageTest struct {
	i string
	c *kubeadmapi.MasterConfiguration
	o string
}

const testversion = "1"

func TestGetCoreImage(t *testing.T) {
	var tokenTest = []struct {
		t        getCoreImageTest
		expected string
	}{
		{getCoreImageTest{o: "override"}, "override"},
		{getCoreImageTest{
			i: KubeEtcdImage,
			c: &kubeadmapi.MasterConfiguration{}},
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "etcd", runtime.GOARCH, etcdVersion),
		},
		{getCoreImageTest{
			i: KubeAPIServerImage,
			c: &kubeadmapi.MasterConfiguration{KubernetesVersion: testversion}},
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-apiserver", runtime.GOARCH, testversion),
		},
		{getCoreImageTest{
			i: KubeControllerManagerImage,
			c: &kubeadmapi.MasterConfiguration{KubernetesVersion: testversion}},
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-controller-manager", runtime.GOARCH, testversion),
		},
		{getCoreImageTest{
			i: KubeSchedulerImage,
			c: &kubeadmapi.MasterConfiguration{KubernetesVersion: testversion}},
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-scheduler", runtime.GOARCH, testversion),
		},
		{getCoreImageTest{
			i: KubeProxyImage,
			c: &kubeadmapi.MasterConfiguration{KubernetesVersion: testversion}},
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-proxy", runtime.GOARCH, testversion),
		},
	}
	for _, rt := range tokenTest {
		actual := GetCoreImage(rt.t.i, rt.t.c, rt.t.o)
		if actual != rt.expected {
			t.Errorf(
				"failed GetCoreImage:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestGetAddonImage(t *testing.T) {
	var tokenTest = []struct {
		t        string
		expected string
	}{
		{"matches nothing", ""},
		{
			KubeDNSImage,
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kubedns", runtime.GOARCH, kubeDNSVersion),
		},
		{
			KubeDNSmasqImage,
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-dnsmasq", runtime.GOARCH, dnsmasqVersion),
		},
		{
			KubeExechealthzImage,
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "exechealthz", runtime.GOARCH, exechealthzVersion),
		},
		{
			Pause,
			fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "pause", runtime.GOARCH, pauseVersion),
		},
	}
	for _, rt := range tokenTest {
		actual := GetAddonImage(rt.t)
		if actual != rt.expected {
			t.Errorf(
				"failed GetCoreImage:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}
