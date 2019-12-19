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
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	kubeletconfig "k8s.io/kubelet/config/v1beta1"
	utilpointer "k8s.io/utils/pointer"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// kubeletMarshalCases holds common marshal test cases for both the marshal and unmarshal tests
var kubeletMarshalCases = []struct {
	name string
	obj  *kubeletConfig
	yaml string
}{
	{
		name: "Empty config",
		obj: &kubeletConfig{
			config: kubeletconfig.KubeletConfiguration{},
		},
		yaml: dedent.Dedent(`
			apiVersion: kubelet.config.k8s.io/v1beta1
			authentication:
			  anonymous: {}
			  webhook:
			    cacheTTL: 0s
			  x509: {}
			authorization:
			  webhook:
			    cacheAuthorizedTTL: 0s
			    cacheUnauthorizedTTL: 0s
			cpuManagerReconcilePeriod: 0s
			evictionPressureTransitionPeriod: 0s
			fileCheckFrequency: 0s
			httpCheckFrequency: 0s
			imageMinimumGCAge: 0s
			kind: KubeletConfiguration
			nodeStatusReportFrequency: 0s
			nodeStatusUpdateFrequency: 0s
			runtimeRequestTimeout: 0s
			streamingConnectionIdleTimeout: 0s
			syncFrequency: 0s
			volumeStatsAggPeriod: 0s
		`),
	},
	{
		name: "Non empty config",
		obj: &kubeletConfig{
			config: kubeletconfig.KubeletConfiguration{
				Address:            "1.2.3.4",
				Port:               12345,
				RotateCertificates: true,
			},
		},
		yaml: dedent.Dedent(`
			address: 1.2.3.4
			apiVersion: kubelet.config.k8s.io/v1beta1
			authentication:
			  anonymous: {}
			  webhook:
			    cacheTTL: 0s
			  x509: {}
			authorization:
			  webhook:
			    cacheAuthorizedTTL: 0s
			    cacheUnauthorizedTTL: 0s
			cpuManagerReconcilePeriod: 0s
			evictionPressureTransitionPeriod: 0s
			fileCheckFrequency: 0s
			httpCheckFrequency: 0s
			imageMinimumGCAge: 0s
			kind: KubeletConfiguration
			nodeStatusReportFrequency: 0s
			nodeStatusUpdateFrequency: 0s
			port: 12345
			rotateCertificates: true
			runtimeRequestTimeout: 0s
			streamingConnectionIdleTimeout: 0s
			syncFrequency: 0s
			volumeStatsAggPeriod: 0s
		`),
	},
}

func TestKubeletMarshal(t *testing.T) {
	for _, test := range kubeletMarshalCases {
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

func TestKubeletUnmarshal(t *testing.T) {
	for _, test := range kubeletMarshalCases {
		t.Run(test.name, func(t *testing.T) {
			gvkmap, err := kubeadmutil.SplitYAMLDocuments([]byte(test.yaml))
			if err != nil {
				t.Fatalf("unexpected failure of SplitYAMLDocuments: %v", err)
			}

			got := &kubeletConfig{}
			if err = got.Unmarshal(gvkmap); err != nil {
				t.Fatalf("unexpected failure of Unmarshal: %v", err)
			}

			expected := test.obj.DeepCopy().(*kubeletConfig)
			expected.config.APIVersion = kubeletHandler.GroupVersion.String()
			expected.config.Kind = "KubeletConfiguration"

			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("Missmatch between expected and got:\nExpected:\n%v\n---\nGot:\n%v", expected, got)
			}
		})
	}
}

func TestKubeletDefault(t *testing.T) {
	tests := []struct {
		name       string
		clusterCfg kubeadmapi.ClusterConfiguration
		expected   kubeletConfig
	}{
		{
			name:       "No specific defaulting works",
			clusterCfg: kubeadmapi.ClusterConfiguration{},
			expected: kubeletConfig{
				config: kubeletconfig.KubeletConfiguration{
					FeatureGates:  map[string]bool{},
					StaticPodPath: kubeadmapiv1beta2.DefaultManifestsDir,
					ClusterDNS:    []string{kubeadmapiv1beta2.DefaultClusterDNSIP},
					Authentication: kubeletconfig.KubeletAuthentication{
						X509: kubeletconfig.KubeletX509Authentication{
							ClientCAFile: constants.CACertName,
						},
						Anonymous: kubeletconfig.KubeletAnonymousAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationAnonymousEnabled),
						},
						Webhook: kubeletconfig.KubeletWebhookAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationWebhookEnabled),
						},
					},
					Authorization: kubeletconfig.KubeletAuthorization{
						Mode: kubeletconfig.KubeletAuthorizationModeWebhook,
					},
					HealthzBindAddress: kubeletHealthzBindAddress,
					HealthzPort:        utilpointer.Int32Ptr(constants.KubeletHealthzPort),
					RotateCertificates: kubeletRotateCertificates,
				},
			},
		},
		{
			name: "Service subnet, no dual stack defaulting works",
			clusterCfg: kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					ServiceSubnet: "192.168.0.0/16",
				},
			},
			expected: kubeletConfig{
				config: kubeletconfig.KubeletConfiguration{
					FeatureGates:  map[string]bool{},
					StaticPodPath: kubeadmapiv1beta2.DefaultManifestsDir,
					ClusterDNS:    []string{"192.168.0.10"},
					Authentication: kubeletconfig.KubeletAuthentication{
						X509: kubeletconfig.KubeletX509Authentication{
							ClientCAFile: constants.CACertName,
						},
						Anonymous: kubeletconfig.KubeletAnonymousAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationAnonymousEnabled),
						},
						Webhook: kubeletconfig.KubeletWebhookAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationWebhookEnabled),
						},
					},
					Authorization: kubeletconfig.KubeletAuthorization{
						Mode: kubeletconfig.KubeletAuthorizationModeWebhook,
					},
					HealthzBindAddress: kubeletHealthzBindAddress,
					HealthzPort:        utilpointer.Int32Ptr(constants.KubeletHealthzPort),
					RotateCertificates: kubeletRotateCertificates,
				},
			},
		},
		{
			name: "Service subnet, dual stack defaulting works",
			clusterCfg: kubeadmapi.ClusterConfiguration{
				FeatureGates: map[string]bool{
					features.IPv6DualStack: true,
				},
				Networking: kubeadmapi.Networking{
					ServiceSubnet: "192.168.0.0/16",
				},
			},
			expected: kubeletConfig{
				config: kubeletconfig.KubeletConfiguration{
					FeatureGates:  map[string]bool{},
					StaticPodPath: kubeadmapiv1beta2.DefaultManifestsDir,
					ClusterDNS:    []string{"192.168.0.10"},
					Authentication: kubeletconfig.KubeletAuthentication{
						X509: kubeletconfig.KubeletX509Authentication{
							ClientCAFile: constants.CACertName,
						},
						Anonymous: kubeletconfig.KubeletAnonymousAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationAnonymousEnabled),
						},
						Webhook: kubeletconfig.KubeletWebhookAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationWebhookEnabled),
						},
					},
					Authorization: kubeletconfig.KubeletAuthorization{
						Mode: kubeletconfig.KubeletAuthorizationModeWebhook,
					},
					HealthzBindAddress: kubeletHealthzBindAddress,
					HealthzPort:        utilpointer.Int32Ptr(constants.KubeletHealthzPort),
					RotateCertificates: kubeletRotateCertificates,
				},
			},
		},
		{
			name: "DNS domain defaulting works",
			clusterCfg: kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					DNSDomain: "example.com",
				},
			},
			expected: kubeletConfig{
				config: kubeletconfig.KubeletConfiguration{
					FeatureGates:  map[string]bool{},
					StaticPodPath: kubeadmapiv1beta2.DefaultManifestsDir,
					ClusterDNS:    []string{kubeadmapiv1beta2.DefaultClusterDNSIP},
					ClusterDomain: "example.com",
					Authentication: kubeletconfig.KubeletAuthentication{
						X509: kubeletconfig.KubeletX509Authentication{
							ClientCAFile: constants.CACertName,
						},
						Anonymous: kubeletconfig.KubeletAnonymousAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationAnonymousEnabled),
						},
						Webhook: kubeletconfig.KubeletWebhookAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationWebhookEnabled),
						},
					},
					Authorization: kubeletconfig.KubeletAuthorization{
						Mode: kubeletconfig.KubeletAuthorizationModeWebhook,
					},
					HealthzBindAddress: kubeletHealthzBindAddress,
					HealthzPort:        utilpointer.Int32Ptr(constants.KubeletHealthzPort),
					RotateCertificates: kubeletRotateCertificates,
				},
			},
		},
		{
			name: "CertificatesDir defaulting works",
			clusterCfg: kubeadmapi.ClusterConfiguration{
				CertificatesDir: "/path/to/certs",
			},
			expected: kubeletConfig{
				config: kubeletconfig.KubeletConfiguration{
					FeatureGates:  map[string]bool{},
					StaticPodPath: kubeadmapiv1beta2.DefaultManifestsDir,
					ClusterDNS:    []string{kubeadmapiv1beta2.DefaultClusterDNSIP},
					Authentication: kubeletconfig.KubeletAuthentication{
						X509: kubeletconfig.KubeletX509Authentication{
							ClientCAFile: filepath.Join("/path/to/certs", constants.CACertName),
						},
						Anonymous: kubeletconfig.KubeletAnonymousAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationAnonymousEnabled),
						},
						Webhook: kubeletconfig.KubeletWebhookAuthentication{
							Enabled: utilpointer.BoolPtr(kubeletAuthenticationWebhookEnabled),
						},
					},
					Authorization: kubeletconfig.KubeletAuthorization{
						Mode: kubeletconfig.KubeletAuthorizationModeWebhook,
					},
					HealthzBindAddress: kubeletHealthzBindAddress,
					HealthzPort:        utilpointer.Int32Ptr(constants.KubeletHealthzPort),
					RotateCertificates: kubeletRotateCertificates,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := &kubeletConfig{}
			got.Default(&test.clusterCfg, &kubeadmapi.APIEndpoint{})
			if !reflect.DeepEqual(got, &test.expected) {
				t.Fatalf("Missmatch between expected and got:\nExpected:\n%v\n---\nGot:\n%v", test.expected, got)
			}
		})
	}
}

// runKubeletFromTest holds common test case data and evaluation code for kubeletHandler.From* functions
func runKubeletFromTest(t *testing.T, perform func(t *testing.T, in string) (kubeadmapi.ComponentConfig, error)) {
	tests := []struct {
		name      string
		in        string
		out       *kubeletConfig
		expectErr bool
	}{
		{
			name: "Empty document map should return nothing successfully",
		},
		{
			name: "Non-empty non-kubelet document map returns nothing successfully",
			in: dedent.Dedent(`
				apiVersion: api.example.com/v1
				kind: Configuration
			`),
		},
		{
			name: "Old kubelet version returns an error",
			in: dedent.Dedent(`
				apiVersion: kubelet.config.k8s.io/v1alpha1
				kind: KubeletConfiguration
			`),
			expectErr: true,
		},
		{
			name: "New kubelet version returns an error",
			in: dedent.Dedent(`
				apiVersion: kubelet.config.k8s.io/v1
				kind: KubeletConfiguration
			`),
			expectErr: true,
		},
		{
			name: "Wrong kubelet kind returns an error",
			in: dedent.Dedent(`
				apiVersion: kubelet.config.k8s.io/v1beta1
				kind: Configuration
			`),
			expectErr: true,
		},
		{
			name: "Valid kubelet only config gets loaded",
			in: dedent.Dedent(`
				apiVersion: kubelet.config.k8s.io/v1beta1
				kind: KubeletConfiguration
				address: 1.2.3.4
				port: 12345
				rotateCertificates: true
			`),
			out: &kubeletConfig{
				config: kubeletconfig.KubeletConfiguration{
					TypeMeta: metav1.TypeMeta{
						APIVersion: kubeletHandler.GroupVersion.String(),
						Kind:       "KubeletConfiguration",
					},
					Address:            "1.2.3.4",
					Port:               12345,
					RotateCertificates: true,
				},
			},
		},
		{
			name: "Valid kubelet config gets loaded when coupled with an extra document",
			in: dedent.Dedent(`
				apiVersion: api.example.com/v1
				kind: Configuration
				---
				apiVersion: kubelet.config.k8s.io/v1beta1
				kind: KubeletConfiguration
				address: 1.2.3.4
				port: 12345
				rotateCertificates: true
			`),
			out: &kubeletConfig{
				config: kubeletconfig.KubeletConfiguration{
					TypeMeta: metav1.TypeMeta{
						APIVersion: kubeletHandler.GroupVersion.String(),
						Kind:       "KubeletConfiguration",
					},
					Address:            "1.2.3.4",
					Port:               12345,
					RotateCertificates: true,
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
						if got, ok := componentCfg.(*kubeletConfig); !ok {
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

func TestKubeletFromDocumentMap(t *testing.T) {
	runKubeletFromTest(t, func(t *testing.T, in string) (kubeadmapi.ComponentConfig, error) {
		gvkmap, err := kubeadmutil.SplitYAMLDocuments([]byte(in))
		if err != nil {
			t.Fatalf("unexpected failure of SplitYAMLDocuments: %v", err)
		}

		return kubeletHandler.FromDocumentMap(gvkmap)
	})
}

func TestKubeletFromCluster(t *testing.T) {
	runKubeletFromTest(t, func(t *testing.T, in string) (kubeadmapi.ComponentConfig, error) {
		clusterCfg := &kubeadmapi.ClusterConfiguration{
			KubernetesVersion: constants.CurrentKubernetesVersion.String(),
		}

		k8sVersion := version.MustParseGeneric(clusterCfg.KubernetesVersion)

		client := clientsetfake.NewSimpleClientset(
			&v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      constants.GetKubeletConfigMapName(k8sVersion),
					Namespace: metav1.NamespaceSystem,
				},
				Data: map[string]string{
					constants.KubeletBaseConfigurationConfigMapKey: in,
				},
			},
		)

		return kubeletHandler.FromCluster(client, clusterCfg)
	})
}
