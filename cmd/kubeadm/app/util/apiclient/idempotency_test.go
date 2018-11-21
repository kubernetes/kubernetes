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

package apiclient_test

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

func TestPatchNodeNonErrorCases(t *testing.T) {
	testcases := []struct {
		name       string
		lookupName string
		node       v1.Node
		success    bool
	}{
		{
			name:       "simple update",
			lookupName: "testnode",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "testnode",
					Labels: map[string]string{kubeletapis.LabelHostname: ""},
				},
			},
			success: true,
		},
		{
			name:       "node does not exist",
			lookupName: "whale",
			success:    false,
		},
		{
			name:       "node not labelled yet",
			lookupName: "robin",
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "robin",
				},
			},
			success: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			_, err := client.Core().Nodes().Create(&tc.node)
			if err != nil {
				t.Fatalf("failed to create node to fake client: %v", err)
			}
			conditionFunction := apiclient.PatchNodeOnce(client, tc.lookupName, func(node *v1.Node) {
				node.Name = "testNewNode"
			})
			success, err := conditionFunction()
			if err != nil {
				t.Fatalf("did not expect error: %v", err)
			}
			if success != tc.success {
				t.Fatalf("expected %v got %v", tc.success, success)
			}
		})
	}
}
