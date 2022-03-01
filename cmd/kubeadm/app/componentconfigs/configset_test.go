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

	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// TODO: cleanup after UnversionedKubeletConfigMap goes GA:
// https://github.com/kubernetes/kubeadm/issues/1582
func testClusterCfg(legacyKubeletConfigMap bool) *kubeadmapi.ClusterConfiguration {
	if legacyKubeletConfigMap {
		return &kubeadmapi.ClusterConfiguration{
			KubernetesVersion: constants.CurrentKubernetesVersion.String(),
		}
	}
	return &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: constants.CurrentKubernetesVersion.String(),
		FeatureGates:      map[string]bool{features.UnversionedKubeletConfigMap: true},
	}
}

func TestDefault(t *testing.T) {
	legacyKubeletConfigMap := false
	clusterCfg := testClusterCfg(legacyKubeletConfigMap)
	localAPIEndpoint := &kubeadmapi.APIEndpoint{}
	nodeRegOps := &kubeadmapi.NodeRegistrationOptions{}

	Default(clusterCfg, localAPIEndpoint, nodeRegOps)

	if len(clusterCfg.ComponentConfigs) != len(known) {
		t.Errorf("mismatch between supported and defaulted type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(known))
	}
}

func TestFromCluster(t *testing.T) {
	objects := []runtime.Object{
		testKubeProxyConfigMap(`
			apiVersion: kubeproxy.config.k8s.io/v1alpha1
			kind: KubeProxyConfiguration
		`),
		testKubeletConfigMap(`
			apiVersion: kubelet.config.k8s.io/v1beta1
			kind: KubeletConfiguration
		`, false),
	}
	client := clientsetfake.NewSimpleClientset(objects...)
	legacyKubeletConfigMap := false
	clusterCfg := testClusterCfg(legacyKubeletConfigMap)

	if err := FetchFromCluster(clusterCfg, client); err != nil {
		t.Fatalf("FetchFromCluster failed: %v", err)
	}

	if len(clusterCfg.ComponentConfigs) != len(objects) {
		t.Fatalf("mismatch between supplied and loaded type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(objects))
	}

	// TODO: cleanup the legacy case below after UnversionedKubeletConfigMap goes GA:
	// https://github.com/kubernetes/kubeadm/issues/1582
	objectsLegacyKubelet := []runtime.Object{
		testKubeProxyConfigMap(`
			apiVersion: kubeproxy.config.k8s.io/v1alpha1
			kind: KubeProxyConfiguration
		`),
		testKubeletConfigMap(`
			apiVersion: kubelet.config.k8s.io/v1beta1
			kind: KubeletConfiguration
		`, true),
	}
	clientLegacyKubelet := clientsetfake.NewSimpleClientset(objectsLegacyKubelet...)
	legacyKubeletConfigMap = true
	clusterCfgLegacyKubelet := testClusterCfg(legacyKubeletConfigMap)

	if err := FetchFromCluster(clusterCfgLegacyKubelet, clientLegacyKubelet); err != nil {
		t.Fatalf("FetchFromCluster failed: %v", err)
	}

	if len(clusterCfgLegacyKubelet.ComponentConfigs) != len(objectsLegacyKubelet) {
		t.Fatalf("mismatch between supplied and loaded type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(objects))
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

	legacyKubeletConfigMap := false
	clusterCfg := testClusterCfg(legacyKubeletConfigMap)
	if err = FetchFromDocumentMap(clusterCfg, gvkmap); err != nil {
		t.Fatalf("FetchFromDocumentMap failed: %v", err)
	}

	if len(clusterCfg.ComponentConfigs) != len(gvkmap) {
		t.Fatalf("mismatch between supplied and loaded type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(gvkmap))
	}
}
