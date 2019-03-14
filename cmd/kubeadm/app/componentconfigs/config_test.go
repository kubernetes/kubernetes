/*
Copyright 2018 The Kubernetes Authors.

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
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

var cfgFiles = map[string][]byte{
	"Kube-proxy_componentconfig": []byte(`
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
`),
	"Kubelet_componentconfig": []byte(`
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
`),
}

func TestGetFromConfigMap(t *testing.T) {
	k8sVersion := version.MustParseGeneric(kubeadmconstants.CurrentKubernetesVersion.String())

	var tests = []struct {
		name          string
		component     RegistrationKind
		configMap     *fakeConfigMap
		expectedError bool
	}{
		{
			name:      "valid kube-proxy",
			component: KubeProxyConfigurationKind,
			configMap: &fakeConfigMap{
				name: kubeadmconstants.KubeProxyConfigMap,
				data: map[string]string{
					kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
				},
			},
		},
		{
			name:          "invalid kube-proxy - missing ConfigMap",
			component:     KubeProxyConfigurationKind,
			configMap:     nil,
			expectedError: true,
		},
		{
			name:      "invalid kube-proxy - missing key",
			component: KubeProxyConfigurationKind,
			configMap: &fakeConfigMap{
				name: kubeadmconstants.KubeProxyConfigMap,
				data: map[string]string{},
			},
			expectedError: true,
		},
		{
			name:      "valid kubelet",
			component: KubeletConfigurationKind,
			configMap: &fakeConfigMap{
				name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion),
				data: map[string]string{
					kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(cfgFiles["Kubelet_componentconfig"]),
				},
			},
		},
		{
			name:          "invalid kubelet - missing ConfigMap",
			component:     KubeletConfigurationKind,
			configMap:     nil,
			expectedError: true,
		},
		{
			name:      "invalid kubelet - missing key",
			component: KubeletConfigurationKind,
			configMap: &fakeConfigMap{
				name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion),
				data: map[string]string{},
			},
			expectedError: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			client := clientsetfake.NewSimpleClientset()

			if rt.configMap != nil {
				err := rt.configMap.create(client)
				if err != nil {
					t.Errorf("unexpected create ConfigMap %s", rt.configMap.name)
					return
				}
			}

			registration := Known[rt.component]

			obj, err := registration.GetFromConfigMap(client, k8sVersion)
			if rt.expectedError != (err != nil) {
				t.Errorf("unexpected return err from GetFromConfigMap: %v", err)
				return
			}
			if rt.expectedError {
				return
			}

			if obj == nil {
				t.Error("unexpected nil return value")
			}
		})
	}
}

type fakeConfigMap struct {
	name string
	data map[string]string
}

func (c *fakeConfigMap) create(client clientset.Interface) error {
	return apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      c.name,
			Namespace: metav1.NamespaceSystem,
		},
		Data: c.data,
	})
}
