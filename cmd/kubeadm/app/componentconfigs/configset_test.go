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
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func testClusterCfg() *kubeadmapi.ClusterConfiguration {
	return &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: constants.CurrentKubernetesVersion.String(),
	}
}

func TestDefault(t *testing.T) {
	clusterCfg := testClusterCfg()
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
		`),
	}
	client := clientsetfake.NewSimpleClientset(objects...)
	clusterCfg := testClusterCfg()

	if err := FetchFromCluster(clusterCfg, client); err != nil {
		t.Fatalf("FetchFromCluster failed: %v", err)
	}

	if len(clusterCfg.ComponentConfigs) != len(objects) {
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
	gvkmap, err := kubeadmutil.SplitConfigDocuments([]byte(test))
	if err != nil {
		t.Fatalf("unexpected failure of SplitConfigDocuments: %v", err)
	}

	clusterCfg := testClusterCfg()
	if err = FetchFromDocumentMap(clusterCfg, gvkmap); err != nil {
		t.Fatalf("FetchFromDocumentMap failed: %v", err)
	}

	if len(clusterCfg.ComponentConfigs) != len(gvkmap) {
		t.Fatalf("mismatch between supplied and loaded type numbers:\n\tgot: %d\n\texpected: %d", len(clusterCfg.ComponentConfigs), len(gvkmap))
	}
}
