/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Node Lifecycle", func() {

	f := framework.NewDefaultFramework("fake-node")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should run through the lifecycle of a node", func(ctx context.Context) {
		// the scope of this test only covers the api-server

		nodeClient := f.ClientSet.CoreV1().Nodes()

		fakeNode := v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-fake-node-" + utilrand.String(5),
			},
			Status: v1.NodeStatus{
				Phase: v1.NodeRunning,
				Conditions: []v1.NodeCondition{
					{
						Status:  v1.ConditionTrue,
						Message: "Set from e2e test",
						Reason:  "E2E",
						Type:    v1.NodeReady,
					},
				},
			},
		}

		ginkgo.By(fmt.Sprintf("Create %q", fakeNode.Name))
		createdNode, err := nodeClient.Create(ctx, &fakeNode, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create node %q", fakeNode.Name)
		gomega.Expect(createdNode.Name).To(gomega.Equal(fakeNode.Name), "Checking that the node has been created")

		ginkgo.By(fmt.Sprintf("Getting %q", fakeNode.Name))
		retrievedNode, err := nodeClient.Get(ctx, fakeNode.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to retrieve Node %q", fakeNode.Name)
		gomega.Expect(retrievedNode.Name).To(gomega.Equal(fakeNode.Name), "Checking that the retrieved name has been found")

		ginkgo.By(fmt.Sprintf("Patching %q", fakeNode.Name))
		payload := "{\"metadata\":{\"labels\":{\"" + fakeNode.Name + "\":\"patched\"}}}"
		patchedNode, err := nodeClient.Patch(ctx, fakeNode.Name, types.StrategicMergePatchType, []byte(payload), metav1.PatchOptions{})
		framework.ExpectNoError(err, "Failed to patch %q", fakeNode.Name)
		gomega.Expect(patchedNode.Labels).To(gomega.HaveKeyWithValue(fakeNode.Name, "patched"), "Checking that patched label has been applied")
		patchedSelector := labels.Set{fakeNode.Name: "patched"}.AsSelector().String()

		ginkgo.By(fmt.Sprintf("Listing nodes with LabelSelector %q", patchedSelector))
		nodes, err := nodeClient.List(ctx, metav1.ListOptions{LabelSelector: patchedSelector})
		framework.ExpectNoError(err, "failed to list nodes")
		gomega.Expect(nodes.Items).To(gomega.HaveLen(1), "confirm that the patched node has been found")

		ginkgo.By(fmt.Sprintf("Updating %q", fakeNode.Name))
		var updatedNode *v1.Node

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			tmpNode, err := nodeClient.Get(ctx, fakeNode.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get %q", fakeNode.Name)
			tmpNode.Labels[fakeNode.Name] = "updated"
			updatedNode, err = nodeClient.Update(ctx, tmpNode, metav1.UpdateOptions{})

			return err
		})
		framework.ExpectNoError(err, "failed to update %q", fakeNode.Name)
		gomega.Expect(updatedNode.Labels).To(gomega.HaveKeyWithValue(fakeNode.Name, "updated"), "Checking that updated label has been applied")

		ginkgo.By(fmt.Sprintf("Delete %q", fakeNode.Name))
		err = nodeClient.Delete(ctx, fakeNode.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete node")

		ginkgo.By(fmt.Sprintf("Confirm deletion of %q", fakeNode.Name))
		gomega.Eventually(ctx, func(ctx context.Context) error {
			_, err := nodeClient.Get(ctx, fakeNode.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return nil
			}
			if err != nil {
				return fmt.Errorf("nodeClient.Get returned an unexpected error: %w", err)
			}
			return fmt.Errorf("node still exists: %s", fakeNode.Name)
		}, 3*time.Minute, 5*time.Second).Should(gomega.Succeed(), "Timeout while waiting to confirm Node deletion")
	})
})
