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

	Default(clusterCfg, localAPIEndpoint)

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
