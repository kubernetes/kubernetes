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

package proxy

import (
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
)

func TestCreateServiceAccount(t *testing.T) {
	tests := []struct {
		name      string
		createErr error
		expectErr bool
	}{
		{
			"error-free case",
			nil,
			false,
		},
		{
			"duplication errors should be ignored",
			apierrors.NewAlreadyExists(api.Resource(""), ""),
			false,
		},
		{
			"unexpected errors should be returned",
			apierrors.NewUnauthorized(""),
			true,
		},
	}

	for _, tc := range tests {
		client := clientsetfake.NewSimpleClientset()
		if tc.createErr != nil {
			client.PrependReactor("create", "serviceaccounts", func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, tc.createErr
			})
		}

		err := CreateServiceAccount(client)
		if tc.expectErr {
			if err == nil {
				t.Errorf("CreateServiceAccounts(%s) wanted err, got nil", tc.name)
			}
			continue
		} else if !tc.expectErr && err != nil {
			t.Errorf("CreateServiceAccounts(%s) returned unexpected err: %v", tc.name, err)
		}

		wantResourcesCreated := 1
		if len(client.Actions()) != wantResourcesCreated {
			t.Errorf("CreateServiceAccounts(%s) should have made %d actions, but made %d", tc.name, wantResourcesCreated, len(client.Actions()))
		}

		for _, action := range client.Actions() {
			if action.GetVerb() != "create" || action.GetResource().Resource != "serviceaccounts" {
				t.Errorf("CreateServiceAccounts(%s) called [%v %v], but wanted [create serviceaccounts]",
					tc.name, action.GetVerb(), action.GetResource().Resource)
			}
		}

	}
}

func TestGetClusterCIDR(t *testing.T) {
	emptyClusterCIDR := getClusterCIDR("")
	if emptyClusterCIDR != "" {
		t.Errorf("Invalid format: %s", emptyClusterCIDR)
	}

	clusterCIDR := getClusterCIDR("10.244.0.0/16")
	if clusterCIDR != "- --cluster-cidr=10.244.0.0/16" {
		t.Errorf("Invalid format: %s", clusterCIDR)
	}

	clusterIPv6CIDR := getClusterCIDR("2001:db8::/64")
	if clusterIPv6CIDR != "- --cluster-cidr=2001:db8::/64" {
		t.Errorf("Invalid format: %s", clusterIPv6CIDR)
	}
}

func TestCompileManifests(t *testing.T) {
	var tests = []struct {
		manifest string
		data     interface{}
		expected bool
	}{
		{
			manifest: KubeProxyConfigMap,
			data: struct{ MasterEndpoint string }{
				MasterEndpoint: "foo",
			},
			expected: true,
		},
		{
			manifest: KubeProxyDaemonSet,
			data: struct{ ImageRepository, Arch, Version, ImageOverride, ClusterCIDR, MasterTaintKey, CloudTaintKey string }{
				ImageRepository: "foo",
				Arch:            "foo",
				Version:         "foo",
				ImageOverride:   "foo",
				ClusterCIDR:     "foo",
				MasterTaintKey:  "foo",
				CloudTaintKey:   "foo",
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		_, actual := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CompileManifests:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
