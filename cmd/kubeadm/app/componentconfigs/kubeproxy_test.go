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
	"fmt"
	"reflect"
	"testing"

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	componentbaseconfig "k8s.io/component-base/config/v1alpha1"
	kubeproxyconfig "k8s.io/kube-proxy/config/v1alpha1"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func testKubeProxyConfigMap(contents string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constants.KubeProxyConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			constants.KubeProxyConfigMapKey: dedent.Dedent(contents),
		},
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
					BindAddress:  kubeadmapiv1.DefaultProxyBindAddressv6,
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
					BindAddress:  kubeadmapiv1.DefaultProxyBindAddressv4,
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
					BindAddress:  kubeadmapiv1.DefaultProxyBindAddressv6,
					ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
						Kubeconfig: kubeproxyKubeConfigFileName,
					},
					ClusterCIDR: "192.168.0.0/16",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// This is the same for all test cases so we set it here
			expected := test.expected
			expected.configBase.GroupVersion = kubeproxyconfig.SchemeGroupVersion

			got := &kubeProxyConfig{
				configBase: configBase{
					GroupVersion: kubeproxyconfig.SchemeGroupVersion,
				},
			}
			got.Default(&test.clusterCfg, &test.endpoint, &kubeadmapi.NodeRegistrationOptions{})
			if !reflect.DeepEqual(got, &expected) {
				t.Fatalf("Missmatch between expected and got:\nExpected:\n%v\n---\nGot:\n%v", expected, got)
			}
		})
	}
}

// runKubeProxyFromTest holds common test case data and evaluation code for kubeProxyHandler.From* functions
func runKubeProxyFromTest(t *testing.T, perform func(gvk schema.GroupVersionKind, yaml string) (kubeadmapi.ComponentConfig, error)) {
	const (
		kind        = "KubeProxyConfiguration"
		clusterCIDR = "1.2.3.4/16"
	)

	gvk := kubeProxyHandler.GroupVersion.WithKind(kind)
	yaml := fmt.Sprintf("apiVersion: %s\nkind: %s\nclusterCIDR: %s", kubeProxyHandler.GroupVersion, kind, clusterCIDR)

	cfg, err := perform(gvk, yaml)

	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	if cfg == nil {
		t.Fatal("no config loaded where it should have been")
	}
	if kubeproxyCfg, ok := cfg.(*kubeProxyConfig); !ok {
		t.Fatalf("found different object type than expected: %s", reflect.TypeOf(cfg))
	} else if kubeproxyCfg.config.ClusterCIDR != clusterCIDR {
		t.Fatalf("unexpected control value (clusterDomain):\n\tgot: %q\n\texpected: %q", kubeproxyCfg.config.ClusterCIDR, clusterCIDR)
	}
}

func TestKubeProxyFromDocumentMap(t *testing.T) {
	runKubeProxyFromTest(t, func(gvk schema.GroupVersionKind, yaml string) (kubeadmapi.ComponentConfig, error) {
		return kubeProxyHandler.FromDocumentMap(kubeadmapi.DocumentMap{
			gvk: []byte(yaml),
		})
	})
}

func TestKubeProxyFromCluster(t *testing.T) {
	runKubeProxyFromTest(t, func(_ schema.GroupVersionKind, yaml string) (kubeadmapi.ComponentConfig, error) {
		client := clientsetfake.NewSimpleClientset(
			testKubeProxyConfigMap(yaml),
		)

		return kubeProxyHandler.FromCluster(client, testClusterCfg())
	})
}
