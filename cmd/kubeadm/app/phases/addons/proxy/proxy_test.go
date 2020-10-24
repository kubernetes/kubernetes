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

	apps "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	core "k8s.io/client-go/testing"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
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
			apierrors.NewAlreadyExists(schema.GroupResource{}, ""),
			false,
		},
		{
			"unexpected errors should be returned",
			apierrors.NewUnauthorized(""),
			true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
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
				return
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
		})
	}
}

func TestCompileManifests(t *testing.T) {
	var tests = []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "KubeProxyConfigMap19",
			manifest: KubeProxyConfigMap19,
			data: struct {
				ControlPlaneEndpoint, ProxyConfig, ProxyConfigMap, ProxyConfigMapKey string
			}{
				ControlPlaneEndpoint: "foo",
				ProxyConfig:          "  bindAddress: 0.0.0.0\n  clusterCIDR: 192.168.1.1\n  enableProfiling: false",
				ProxyConfigMap:       "bar",
				ProxyConfigMapKey:    "baz",
			},
		},
		{
			name:     "KubeProxyDaemonSet19",
			manifest: KubeProxyDaemonSet19,
			data: struct{ Image, ProxyConfigMap, ProxyConfigMapKey string }{
				Image:             "foo",
				ProxyConfigMap:    "bar",
				ProxyConfigMapKey: "baz",
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, err := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
			if err != nil {
				t.Errorf("unexpected ParseTemplate failure: %+v", err)
			}
		})
	}
}

func TestEnsureProxyAddon(t *testing.T) {
	type SimulatedError int
	const (
		NoError SimulatedError = iota
		ServiceAccountError
		InvalidControlPlaneEndpoint
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
		t.Run(tc.name, func(t *testing.T) {
			// Create a fake client and set up default test configuration
			client := clientsetfake.NewSimpleClientset()
			// TODO: Consider using a YAML file instead for this that makes it possible to specify YAML documents for the ComponentConfigs
			controlPlaneConfig := &kubeadmapiv1beta2.InitConfiguration{
				LocalAPIEndpoint: kubeadmapiv1beta2.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         1234,
				},
			}
			controlPlaneClusterConfig := &kubeadmapiv1beta2.ClusterConfiguration{
				Networking: kubeadmapiv1beta2.Networking{
					PodSubnet: "5.6.7.8/24",
				},
				ImageRepository:   "someRepo",
				KubernetesVersion: constants.MinimumControlPlaneVersion.String(),
			}

			// Simulate an error if necessary
			switch tc.simError {
			case ServiceAccountError:
				client.PrependReactor("create", "serviceaccounts", func(action core.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewUnauthorized("")
				})
			case InvalidControlPlaneEndpoint:
				controlPlaneConfig.LocalAPIEndpoint.AdvertiseAddress = "1.2.3"
			case IPv6SetBindAddress:
				controlPlaneConfig.LocalAPIEndpoint.AdvertiseAddress = "1:2::3:4"
				controlPlaneClusterConfig.Networking.PodSubnet = "2001:101::/48"
			}

			intControlPlane, err := configutil.DefaultedInitConfiguration(controlPlaneConfig, controlPlaneClusterConfig)
			if err != nil {
				t.Errorf("test failed to convert external to internal version: %v", err)
				return
			}
			err = EnsureProxyAddon(&intControlPlane.ClusterConfiguration, &intControlPlane.LocalAPIEndpoint, client)

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
		})
	}
}

func TestDaemonSetsHaveSystemNodeCriticalPriorityClassName(t *testing.T) {
	testCases := []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "KubeProxyDaemonSet19",
			manifest: KubeProxyDaemonSet19,
			data: struct{ Image, ProxyConfigMap, ProxyConfigMapKey string }{
				Image:             "foo",
				ProxyConfigMap:    "foo",
				ProxyConfigMapKey: "foo",
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			daemonSetBytes, _ := kubeadmutil.ParseTemplate(testCase.manifest, testCase.data)
			daemonSet := &apps.DaemonSet{}
			if err := runtime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), daemonSetBytes, daemonSet); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if daemonSet.Spec.Template.Spec.PriorityClassName != "system-node-critical" {
				t.Errorf("expected to see system-node-critical priority class name. Got %q instead", daemonSet.Spec.Template.Spec.PriorityClassName)
			}
		})
	}
}
