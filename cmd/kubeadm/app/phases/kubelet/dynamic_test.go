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

package kubelet

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

func TestEnableDynamicConfigForNode(t *testing.T) {
	nodeName := "fake-node"
	client := fake.NewSimpleClientset()
	client.PrependReactor("get", "nodes", func(action core.Action) (bool, runtime.Object, error) {
		return true, &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   nodeName,
				Labels: map[string]string{kubeletapis.LabelHostname: nodeName},
			},
			Spec: v1.NodeSpec{
				ConfigSource: &v1.NodeConfigSource{
					ConfigMap: &v1.ConfigMapNodeConfigSource{
						UID: "",
					},
				},
			},
		}, nil
	})
	client.PrependReactor("get", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		return true, &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "kubelet-config-1.11",
				Namespace: metav1.NamespaceSystem,
				UID:       "fake-uid",
			},
		}, nil
	})
	client.PrependReactor("patch", "nodes", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})

	if err := EnableDynamicConfigForNode(client, nodeName, version.MustParseSemantic("v1.11.0")); err != nil {
		t.Errorf("UpdateNodeWithConfigMap: unexpected error %v", err)
	}
}
