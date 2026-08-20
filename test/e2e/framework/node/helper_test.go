/*
Copyright 2026 The Kubernetes Authors.

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

package node

import (
	"context"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

var _ = ginkgo.Describe("AddExtendedResource", func() {
	ginkgo.It("waits for node allocatable to reflect added extended resource", func(ctx context.Context) {
		nodeName := "test-worker"
		resourceName := v1.ResourceName("example.com/combined-resource")
		resourceQuantity := resource.MustParse("2")

		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: nodeName},
			Status: v1.NodeStatus{
				Capacity:    v1.ResourceList{},
				Allocatable: v1.ResourceList{},
			},
		}

		client := fake.NewSimpleClientset(node)

		// Simulate async propagation delay of the patched status in the backend
		client.PrependReactor("patch", "nodes", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
			go func() {
				time.Sleep(50 * time.Millisecond)
				n, _ := client.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
				if n.Status.Capacity == nil {
					n.Status.Capacity = v1.ResourceList{}
				}
				if n.Status.Allocatable == nil {
					n.Status.Allocatable = v1.ResourceList{}
				}
				n.Status.Capacity[resourceName] = resourceQuantity
				n.Status.Allocatable[resourceName] = resourceQuantity
				client.CoreV1().Nodes().UpdateStatus(context.Background(), n, metav1.UpdateOptions{})
			}()
			return true, node, nil
		})

		AddExtendedResource(ctx, client, nodeName, resourceName, resourceQuantity)

		// Immediate check upon return: Allocatable must reflect the added resource
		updatedNode, err := client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())

		q, ok := updatedNode.Status.Allocatable[resourceName]
		gomega.Expect(ok).To(gomega.BeTrue(), "expected allocatable to contain %v", resourceName)
		gomega.Expect(q.Cmp(resourceQuantity)).To(gomega.BeNumerically(">=", 0))
	})
})

func TestNode(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Node Suite")
}
