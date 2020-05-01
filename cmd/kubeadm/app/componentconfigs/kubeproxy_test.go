/*
Copyright 2019 The Kubernetes Authors.

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

package componentconfigs

import (
	"reflect"
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	componentbaseconfig "k8s.io/component-base/config/v1alpha1"
	kubeproxyconfig "k8s.io/kube-proxy/config/v1alpha1"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// kubeProxyMarshalCases holds common marshal test cases for both the marshal and unmarshal tests
var kubeProxyMarshalCases = []struct {
	name string
	obj  *kubeProxyConfig
	yaml string
}{
	{
		name: "Empty config",
		obj: &kubeProxyConfig{
			config: kubeproxyconfig.KubeProxyConfiguration{},
		},
		yaml: dedent.Dedent(`
			apiVersion: kubeproxy.config.k8s.io/v1alpha1
			bindAddress: ""
			bindAddressHardFail: false
			clientConnection:
			  acceptContentTypes: ""
			  burst: 0
			  contentType: ""
			  kubeconfig: ""
			  qps: 0
			clusterCIDR: ""
			configSyncPeriod: 0s
			conntrack:
			  maxPerCore: null
			  min: null
			  tcpCloseWaitTimeout: null
			  tcpEstablishedTimeout: null
			detectLocalMode: ""
			enableProfiling: false
			healthzBindAddress: ""
			hostnameOverride: ""
			iptables:
			  masqueradeAll: false
			  masqueradeBit: null
			  minSyncPeriod: 0s
			  syncPeriod: 0s
			ipvs:
			  excludeCIDRs: null
			  minSyncPeriod: 0s
			  scheduler: ""
			  strictARP: false
			  syncPeriod: 0s
			  tcpFinTimeout: 0s
			  tcpTimeout: 0s
			  udpTimeout: 0s
			kind: KubeProxyConfiguration
			metricsBindAddress: ""
			mode: ""
			nodePortAddresses: null
			oomScoreAdj: null
			portRange: ""
			showHiddenMetricsForVersion: ""
			udpIdleTimeout: 0s
			winkernel:
			  enableDSR: false
			  networkName: ""
			  sourceVip: ""
		`),
	},
	{
		name: "Non empty config",
		obj: &kubeProxyConfig{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:     "1.2.3.4",
				EnableProfiling: true,
			},
		},
		yaml: dedent.Dedent(`
			apiVersion: kubeproxy.config.k8s.io/v1alpha1
			bindAddress: 1.2.3.4
			bindAddressHardFail: false
			clientConnection:
			  acceptContentTypes: ""
			  burst: 0
			  contentType: ""
			  kubeconfig: ""
			  qps: 0
			clusterCIDR: ""
			configSyncPeriod: 0s
			conntrack:
			  maxPerCore: null
			  min: null
			  tcpCloseWaitTimeout: null
			  tcpEstablishedTimeout: null
			detectLocalMode: ""
			enableProfiling: true
			healthzBindAddress: ""
			hostnameOverride: ""
			iptables:
			  masqueradeAll: false
			  masqueradeBit: null
			  minSyncPeriod: 0s
			  syncPeriod: 0s
			ipvs:
			  excludeCIDRs: null
			  minSyncPeriod: 0s
			  scheduler: ""
			  strictARP: false
			  syncPeriod: 0s
			  tcpFinTimeout: 0s
			  tcpTimeout: 0s
			  udpTimeout: 0s
			kind: KubeProxyConfiguration
			metricsBindAddress: ""
			mode: ""
			nodePortAddresses: null
			oomScoreAdj: null
			portRange: ""
			showHiddenMetricsForVersion: ""
			udpIdleTimeout: 0s
			winkernel:
			  enableDSR: false
			  networkName: ""
			  sourceVip: ""
		`),
	},
}

func TestKubeProxyMarshal(t *testing.T) {
	for _, test := range kubeProxyMarshalCases {
		t.Run(test.name, func(t *testing.T) {
			b, err := test.obj.Marshal()
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			got := strings.TrimSpace(string(b))
			expected := strings.TrimSpace(test.yaml)
			if expected != string(got) {
				t.Fatalf("Missmatch between expected and got:\nExpected:\n%s\n---\nGot:\n%s", expected, string(got))
			}
		})
	}
}

func TestKubeProxyUnmarshal(t *testing.T) {
	for _, test := range kubeProxyMarshalCases {
		t.Run(test.name, func(t *testing.T) {
			gvkmap, err := kubeadmutil.SplitYAMLDocuments([]byte(test.yaml))
			if err != nil {
				t.Fatalf("unexpected failure of SplitYAMLDocuments: %v", err)
			}

			got := &kubeProxyConfig{}
			if err = got.Unmarshal(gvkmap); err != nil {
				t.Fatalf("unexpected failure of Unmarshal: %v", err)
			}

			expected := test.obj.DeepCopy().(*kubeProxyConfig)
			expected.config.APIVersion = kubeProxyHandler.GroupVersion.String()
			expected.config.Kind = "KubeProxyConfiguration"

			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("Missmatch between expected and got:\nExpected:\n%v\n---\nGot:\n%v", expected, got)
			}
		})
	}
}

func TestKubeProxyDefault(t *testing.T) {
	tests := []struct {
		name       string
		clusterCfg kubeadmapi.ClusterConfiguration
		endpoint   kubeadmapi.APIEndpoint
		expected   kubeProxyConfig
	}{
		{
			name:       "No specific defaulting works",
			clusterCfg: kubeadmapi.ClusterConfiguration{},
			endpoint:   kubeadmapi.APIEndpoint{},
			expected: kubeProxyConfig{
				config: kubeproxyconfig.KubeProxyConfiguration{
					FeatureGates: map[string]bool{},
					BindAddress:  kubeadmapiv1beta2.DefaultProxyBindAddressv6,
					ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
						Kubeconfig: kubeproxyKubeConfigFileName,
					},
				},
			},
		},
		{
			name:       "IPv4 bind address",
			clusterCfg: kubeadmapi.ClusterConfiguration{},
			endpoint: kubeadmapi.APIEndpoint{
				AdvertiseAddress: "1.2.3.4",
			},
			expected: kubeProxyConfig{
				config: kubeproxyconfig.KubeProxyConfiguration{
					FeatureGates: map[string]bool{},
					BindAddress:  kubeadmapiv1beta2.DefaultProxyBindAddressv4,
					ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
						Kubeconfig: kubeproxyKubeConfigFileName,
					},
				},
			},
		},
		{
			name: "ClusterCIDR is fetched from PodSubnet",
			clusterCfg: kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet: "192.168.0.0/16",
				},
			},
			endpoint: kubeadmapi.APIEndpoint{},
			expected: kubeProxyConfig{
				config: kubeproxyconfig.KubeProxyConfiguration{
					FeatureGates: map[string]bool{},
					BindAddress:  kubeadmapiv1beta2.DefaultProxyBindAddressv6,
					ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
						Kubeconfig: kubeproxyKubeConfigFileName,
					},
					ClusterCIDR: "192.168.0.0/16",
				},
			},
		},
		{
			name: "IPv6DualStack feature gate set to true",
			clusterCfg: kubeadmapi.ClusterConfiguration{
				FeatureGates: map[string]bool{
					features.IPv6DualStack: true,
				},
			},
			endpoint: kubeadmapi.APIEndpoint{},
			expected: kubeProxyConfig{
				config: kubeproxyconfig.KubeProxyConfiguration{
					FeatureGates: map[string]bool{
						features.IPv6DualStack: true,
					},
					BindAddress: kubeadmapiv1beta2.DefaultProxyBindAddressv6,
					ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
						Kubeconfig: kubeproxyKubeConfigFileName,
					},
				},
			},
		},
		{
			name: "IPv6DualStack feature gate set to false",
			clusterCfg: kubeadmapi.ClusterConfiguration{
				FeatureGates: map[string]bool{
					features.IPv6DualStack: false,
				},
			},
			endpoint: kubeadmapi.APIEndpoint{},
			expected: kubeProxyConfig{
				config: kubeproxyconfig.KubeProxyConfiguration{
					FeatureGates: map[string]bool{
						features.IPv6DualStack: false,
					},
					BindAddress: kubeadmapiv1beta2.DefaultProxyBindAddressv6,
					ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
						Kubeconfig: kubeproxyKubeConfigFileName,
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := &kubeProxyConfig{}
			got.Default(&test.clusterCfg, &test.endpoint, &kubeadmapi.NodeRegistrationOptions{})
			if !reflect.DeepEqual(got, &test.expected) {
				t.Fatalf("Missmatch between expected and got:\nExpected:\n%v\n---\nGot:\n%v", test.expected, got)
			}
		})
	}
}

// runKubeProxyFromTest holds common test case data and evaluation code for kubeProxyHandler.From* functions
func runKubeProxyFromTest(t *testing.T, perform func(t *testing.T, in string) (kubeadmapi.ComponentConfig, error)) {
	tests := []struct {
		name      string
		in        string
		out       *kubeProxyConfig
		expectErr bool
	}{
		{
			name: "Empty document map should return nothing successfully",
		},
		{
			name: "Non-empty non-kube-proxy document map returns nothing successfully",
			in: dedent.Dedent(`
				apiVersion: api.example.com/v1
				kind: Configuration
			`),
		},
		{
			name: "Old kube-proxy version returns an error",
			in: dedent.Dedent(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha0
				kind: KubeProxyConfiguration
			`),
			expectErr: true,
		},
		{
			name: "New kube-proxy version returns an error",
			in: dedent.Dedent(`
				apiVersion: kubeproxy.config.k8s.io/v1beta1
				kind: KubeProxyConfiguration
			`),
			expectErr: true,
		},
		{
			name: "Wrong kube-proxy kind returns an error",
			in: dedent.Dedent(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha1
				kind: Configuration
			`),
			expectErr: true,
		},
		{
			name: "Valid kube-proxy only config gets loaded",
			in: dedent.Dedent(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha1
				kind: KubeProxyConfiguration
				bindAddress: 1.2.3.4
				enableProfiling: true
			`),
			out: &kubeProxyConfig{
				config: kubeproxyconfig.KubeProxyConfiguration{
					TypeMeta: metav1.TypeMeta{
						APIVersion: kubeProxyHandler.GroupVersion.String(),
						Kind:       "KubeProxyConfiguration",
					},
					BindAddress:     "1.2.3.4",
					EnableProfiling: true,
				},
			},
		},
		{
			name: "Valid kube-proxy config gets loaded when coupled with an extra document",
			in: dedent.Dedent(`
				apiVersion: api.example.com/v1
				kind: Configuration
				---
				apiVersion: kubeproxy.config.k8s.io/v1alpha1
				kind: KubeProxyConfiguration
				bindAddress: 1.2.3.4
				enableProfiling: true
			`),
			out: &kubeProxyConfig{
				config: kubeproxyconfig.KubeProxyConfiguration{
					TypeMeta: metav1.TypeMeta{
						APIVersion: kubeProxyHandler.GroupVersion.String(),
						Kind:       "KubeProxyConfiguration",
					},
					BindAddress:     "1.2.3.4",
					EnableProfiling: true,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			componentCfg, err := perform(t, test.in)
			if err != nil {
				if !test.expectErr {
					t.Errorf("unexpected failure: %v", err)
				}
			} else {
				if test.expectErr {
					t.Error("unexpected success")
				} else {
					if componentCfg == nil {
						if test.out != nil {
							t.Error("unexpected nil result")
						}
					} else {
						if got, ok := componentCfg.(*kubeProxyConfig); !ok {
							t.Error("different result type")
						} else {
							if test.out == nil {
								t.Errorf("unexpected result: %v", got)
							} else if !reflect.DeepEqual(test.out, got) {
								t.Errorf("missmatch between expected and got:\nExpected:\n%v\n---\nGot:\n%v", test.out, got)
							}
						}
					}
				}
			}
		})
	}
}

func TestKubeProxyFromDocumentMap(t *testing.T) {
	runKubeProxyFromTest(t, func(t *testing.T, in string) (kubeadmapi.ComponentConfig, error) {
		gvkmap, err := kubeadmutil.SplitYAMLDocuments([]byte(in))
		if err != nil {
			t.Fatalf("unexpected failure of SplitYAMLDocuments: %v", err)
		}

		return kubeProxyHandler.FromDocumentMap(gvkmap)
	})
}

func TestKubeProxyFromCluster(t *testing.T) {
	runKubeProxyFromTest(t, func(t *testing.T, in string) (kubeadmapi.ComponentConfig, error) {
		client := clientsetfake.NewSimpleClientset(
			&v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      constants.KubeProxyConfigMap,
					Namespace: metav1.NamespaceSystem,
				},
				Data: map[string]string{
					constants.KubeProxyConfigMapKey: in,
				},
			},
		)

		return kubeProxyHandler.FromCluster(client, &kubeadmapi.ClusterConfiguration{})
	})
}
