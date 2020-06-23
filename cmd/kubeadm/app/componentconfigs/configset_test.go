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
	"testing"

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	clientsetfake "k8s.io/client-go/kubernetes/fake"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestDefault(t *testing.T) {
	clusterCfg := &kubeadmapi.ClusterConfiguration{}
	localAPIEndpoint := &kubeadmapi.APIEndpoint{}
	nodeRegOps := &kubeadmapi.NodeRegistrationOptions{}

	Default(clusterCfg, localAPIEndpoint, nodeRegOps)

	if len(clusterCfg.ComponentConfigs) != len(known) {
		t.Errorf("missmatch between supported and defaulted type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(known))
	}
}

func TestFromCluster(t *testing.T) {
	clusterCfg := &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: constants.CurrentKubernetesVersion.String(),
	}

	k8sVersion := version.MustParseGeneric(clusterCfg.KubernetesVersion)

	objects := []runtime.Object{
		&v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constants.KubeProxyConfigMap,
				Namespace: metav1.NamespaceSystem,
			},
			Data: map[string]string{
				constants.KubeProxyConfigMapKey: dedent.Dedent(`
					apiVersion: kubeproxy.config.k8s.io/v1alpha1
					kind: KubeProxyConfiguration
				`),
			},
		},
		&v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constants.GetKubeletConfigMapName(k8sVersion),
				Namespace: metav1.NamespaceSystem,
			},
			Data: map[string]string{
				constants.KubeletBaseConfigurationConfigMapKey: dedent.Dedent(`
					apiVersion: kubelet.config.k8s.io/v1beta1
					kind: KubeletConfiguration
				`),
			},
		},
	}
	client := clientsetfake.NewSimpleClientset(objects...)

	if err := FetchFromCluster(clusterCfg, client); err != nil {
		t.Fatalf("FetchFromCluster failed: %v", err)
	}

	if len(clusterCfg.ComponentConfigs) != len(objects) {
		t.Fatalf("missmatch between supplied and loaded type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(objects))
	}
}

func TestFetchFromDocumentMap(t *testing.T) {
	test := dedent.Dedent(`
	apiVersion: kubeproxy.config.k8s.io/v1alpha1
	kind: KubeProxyConfiguration
	---
	apiVersion: kubelet.config.k8s.io/v1beta1
	kind: KubeletConfiguration
	`)
	gvkmap, err := kubeadmutil.SplitYAMLDocuments([]byte(test))
	if err != nil {
		t.Fatalf("unexpected failure of SplitYAMLDocuments: %v", err)
	}

	clusterCfg := &kubeadmapi.ClusterConfiguration{}
	if err = FetchFromDocumentMap(clusterCfg, gvkmap); err != nil {
		t.Fatalf("FetchFromDocumentMap failed: %v", err)
	}

	if len(clusterCfg.ComponentConfigs) != len(gvkmap) {
		t.Fatalf("missmatch between supplied and loaded type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(gvkmap))
	}
}

func kubeproxyConfigMap(contents string) *v1.ConfigMap {
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

func TestFetchFromClusterWithLocalUpgrades(t *testing.T) {
	cases := []struct {
		desc          string
		obj           runtime.Object
		config        string
		expectedValue string
		expectedErr   bool
	}{
		{
			desc: "reconginzed cluster object without overwrite is used",
			obj: kubeproxyConfigMap(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha1
				kind: KubeProxyConfiguration
				hostnameOverride: foo
			`),
			expectedValue: "foo",
		},
		{
			desc: "reconginzed cluster object with overwrite is not used",
			obj: kubeproxyConfigMap(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha1
				kind: KubeProxyConfiguration
				hostnameOverride: foo
			`),
			config: dedent.Dedent(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha1
				kind: KubeProxyConfiguration
				hostnameOverride: bar
			`),
			expectedValue: "bar",
		},
		{
			desc: "old config without overwrite returns an error",
			obj: kubeproxyConfigMap(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha0
				kind: KubeProxyConfiguration
				hostnameOverride: foo
			`),
			expectedErr: true,
		},
		{
			desc: "old config with recognized overwrite returns success",
			obj: kubeproxyConfigMap(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha0
				kind: KubeProxyConfiguration
				hostnameOverride: foo
			`),
			config: dedent.Dedent(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha1
				kind: KubeProxyConfiguration
				hostnameOverride: bar
			`),
			expectedValue: "bar",
		},
		{
			desc: "old config with old overwrite returns an error",
			obj: kubeproxyConfigMap(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha0
				kind: KubeProxyConfiguration
				hostnameOverride: foo
			`),
			config: dedent.Dedent(`
				apiVersion: kubeproxy.config.k8s.io/v1alpha0
				kind: KubeProxyConfiguration
				hostnameOverride: bar
			`),
			expectedErr: true,
		},
	}
	for _, test := range cases {
		t.Run(test.desc, func(t *testing.T) {
			clusterCfg := &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: constants.CurrentKubernetesVersion.String(),
			}

			k8sVersion := version.MustParseGeneric(clusterCfg.KubernetesVersion)

			client := clientsetfake.NewSimpleClientset(
				test.obj,
				&v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      constants.GetKubeletConfigMapName(k8sVersion),
						Namespace: metav1.NamespaceSystem,
					},
					Data: map[string]string{
						constants.KubeletBaseConfigurationConfigMapKey: dedent.Dedent(`
							apiVersion: kubelet.config.k8s.io/v1beta1
							kind: KubeletConfiguration
						`),
					},
				},
			)

			docmap, err := kubeadmutil.SplitYAMLDocuments([]byte(test.config))
			if err != nil {
				t.Fatalf("unexpected failure of SplitYAMLDocuments: %v", err)
			}

			err = FetchFromClusterWithLocalOverwrites(clusterCfg, client, docmap)
			if err != nil {
				if !test.expectedErr {
					t.Errorf("unexpected failure: %v", err)
				}
			} else {
				if test.expectedErr {
					t.Error("unexpected success")
				} else {
					kubeproxyCfg, ok := clusterCfg.ComponentConfigs[KubeProxyGroup]
					if !ok {
						t.Error("the config was reported as loaded, but was not in reality")
					} else {
						actualConfig, ok := kubeproxyCfg.(*kubeProxyConfig)
						if !ok {
							t.Error("the config is not of the expected type")
						} else if actualConfig.config.HostnameOverride != test.expectedValue {
							t.Errorf("unexpected value:\n\tgot: %q\n\texpected: %q", actualConfig.config.HostnameOverride, test.expectedValue)
						}
					}
				}
			}
		})
	}
}
