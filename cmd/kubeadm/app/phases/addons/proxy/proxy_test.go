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
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/utils/pointer"
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

func TestCompileManifests(t *testing.T) {
	var tests = []struct {
		manifest string
		data     interface{}
		expected bool
	}{
		{
			manifest: KubeProxyConfigMap19,
			data: struct {
				MasterEndpoint, ProxyConfig, ProxyConfigMap, ProxyConfigMapKey string
			}{
				MasterEndpoint:    "foo",
				ProxyConfig:       "  bindAddress: 0.0.0.0\n  clusterCIDR: 192.168.1.1\n  enableProfiling: false",
				ProxyConfigMap:    "bar",
				ProxyConfigMapKey: "baz",
			},
			expected: true,
		},
		{
			manifest: KubeProxyDaemonSet19,
			data: struct{ Image, ProxyConfigMap, ProxyConfigMapKey string }{
				Image:             "foo",
				ProxyConfigMap:    "bar",
				ProxyConfigMapKey: "baz",
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		_, actual := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed to compile %s manifest:\n\texpected: %t\n\t  actual: %t",
				rt.manifest,
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestEnsureProxyAddon(t *testing.T) {
	type SimulatedError int
	const (
		NoError SimulatedError = iota
		ServiceAccountError
		InvalidMasterEndpoint
		IPv6SetBindAddress
	)

	var testCases = []struct {
		name           string
		simError       SimulatedError
		expErrString   string
		expBindAddr    string
		expClusterCIDR string
	}{
		{
			name:           "Successful proxy addon",
			simError:       NoError,
			expErrString:   "",
			expBindAddr:    "0.0.0.0",
			expClusterCIDR: "5.6.7.8/24",
		}, {
			name:           "Simulated service account error",
			simError:       ServiceAccountError,
			expErrString:   "error when creating kube-proxy service account",
			expBindAddr:    "0.0.0.0",
			expClusterCIDR: "5.6.7.8/24",
		}, {
			name:           "IPv6 AdvertiseAddress address",
			simError:       IPv6SetBindAddress,
			expErrString:   "",
			expBindAddr:    "::",
			expClusterCIDR: "2001:101::/96",
		},
	}

	for _, tc := range testCases {

		// Create a fake client and set up default test configuration
		client := clientsetfake.NewSimpleClientset()
		// TODO: Consider using a YAML file instead for this that makes it possible to specify YAML documents for the ComponentConfigs
		masterConfig := &kubeadmapiv1beta1.InitConfiguration{
			LocalAPIEndpoint: kubeadmapiv1beta1.APIEndpoint{
				AdvertiseAddress: "1.2.3.4",
				BindPort:         1234,
			},
			ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
				Networking: kubeadmapiv1beta1.Networking{
					PodSubnet: "5.6.7.8/24",
				},
				ImageRepository:   "someRepo",
				KubernetesVersion: "v1.12.0",
			},
		}

		// Simulate an error if necessary
		switch tc.simError {
		case ServiceAccountError:
			client.PrependReactor("create", "serviceaccounts", func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewUnauthorized("")
			})
		case InvalidMasterEndpoint:
			masterConfig.LocalAPIEndpoint.AdvertiseAddress = "1.2.3"
		case IPv6SetBindAddress:
			masterConfig.LocalAPIEndpoint.AdvertiseAddress = "1:2::3:4"
			masterConfig.Networking.PodSubnet = "2001:101::/96"
		}

		intMaster, err := configutil.ConfigFileAndDefaultsToInternalConfig("", masterConfig)
		if err != nil {
			t.Errorf("test failed to convert external to internal version")
			break
		}
		intMaster.ComponentConfigs.KubeProxy = &kubeproxyconfig.KubeProxyConfiguration{
			BindAddress:        "",
			HealthzBindAddress: "0.0.0.0:10256",
			MetricsBindAddress: "127.0.0.1:10249",
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				Max:                   pointer.Int32Ptr(2),
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		}
		// Run dynamic defaulting again as we changed the internal cfg
		if err := configutil.SetInitDynamicDefaults(intMaster); err != nil {
			t.Errorf("test failed to set dynamic defaults: %v", err)
			break
		}
		err = EnsureProxyAddon(intMaster, client)

		// Compare actual to expected errors
		actErr := "No error"
		if err != nil {
			actErr = err.Error()
		}
		expErr := "No error"
		if tc.expErrString != "" {
			expErr = tc.expErrString
		}
		if !strings.Contains(actErr, expErr) {
			t.Errorf(
				"%s test failed, expected: %s, got: %s",
				tc.name,
				expErr,
				actErr)
		}
		if intMaster.ComponentConfigs.KubeProxy.BindAddress != tc.expBindAddr {
			t.Errorf("%s test failed, expected: %s, got: %s",
				tc.name,
				tc.expBindAddr,
				intMaster.ComponentConfigs.KubeProxy.BindAddress)
		}
		if intMaster.ComponentConfigs.KubeProxy.ClusterCIDR != tc.expClusterCIDR {
			t.Errorf("%s test failed, expected: %s, got: %s",
				tc.name,
				tc.expClusterCIDR,
				intMaster.ComponentConfigs.KubeProxy.ClusterCIDR)
		}
	}
}
