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

package storage

import (
	"context"
	"fmt"
	"time"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/client-go/util/retry"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = utils.SIGDescribe("CSINodes", func() {

	f := framework.NewDefaultFramework("csinodes")

	ginkgo.Describe("CSI Conformance", func() {

		/*
			Release: v1.32
			Testname: CSINode, lifecycle
			Description: Creating an initial CSINode MUST succeed. Reading a CSINode MUST
			succeed with required name retrieved. Patching a CSINode MUST succeed with its
			new label found. Listing CSINode with a labelSelector MUST succeed. Deleting a
			CSINode MUST succeed and it MUST be confirmed. Creating a replacement CSINode
			MUST succeed. Reading a CSINode MUST succeed with required name retrieved. Updating
			a CSINode MUST succeed with its new label found. Deleting the CSINode via deleteCollection
			MUST succeed and it MUST be confirmed.
		*/
		framework.ConformanceIt("should run through the lifecycle of a csinode", func(ctx context.Context) {

			csiNodeClient := f.ClientSet.StorageV1().CSINodes()

			initialCSINode := storagev1.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "e2e-csinode-" + utilrand.String(5),
				},
			}

			ginkgo.By(fmt.Sprintf("Creating initial csiNode %q", initialCSINode.Name))
			csiNode, err := csiNodeClient.Create(ctx, &initialCSINode, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create csiNode %q", initialCSINode.Name)
			gomega.Expect(csiNode).To(apimachineryutils.HaveValidResourceVersion())

			ginkgo.By(fmt.Sprintf("Getting initial csiNode %q", initialCSINode.Name))
			retrievedCSINode, err := csiNodeClient.Get(ctx, initialCSINode.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to retrieve csiNode %q", initialCSINode.Name)
			gomega.Expect(retrievedCSINode.Name).To(gomega.Equal(csiNode.Name), "Checking that the retrieved name has been found")

			ginkgo.By(fmt.Sprintf("Patching initial csiNode: %q", initialCSINode.Name))
			payload := "{\"metadata\":{\"labels\":{\"" + csiNode.Name + "\":\"patched\"}}}"
			patchedCSINode, err := csiNodeClient.Patch(ctx, csiNode.Name, types.StrategicMergePatchType, []byte(payload), metav1.PatchOptions{})
			framework.ExpectNoError(err, "Failed to patch csiNode %q", csiNode.Name)
			gomega.Expect(patchedCSINode.Labels).To(gomega.HaveKeyWithValue(csiNode.Name, "patched"), "Checking that patched label has been applied")
			gomega.Expect(resourceversion.CompareResourceVersion(csiNode.ResourceVersion, patchedCSINode.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

			patchedSelector := labels.Set{csiNode.Name: "patched"}.AsSelector().String()
			ginkgo.By(fmt.Sprintf("Listing csiNodes with LabelSelector %q", patchedSelector))
			csiNodeList, err := csiNodeClient.List(ctx, metav1.ListOptions{LabelSelector: patchedSelector})
			framework.ExpectNoError(err, "failed to list csiNodes")
			gomega.Expect(csiNodeList.Items).To(gomega.HaveLen(1))

			ginkgo.By(fmt.Sprintf("Delete initial csiNode: %q", initialCSINode.Name))
			err = csiNodeClient.Delete(ctx, csiNode.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete csiNode %q", initialCSINode.Name)

			ginkgo.By(fmt.Sprintf("Confirm deletion of csiNode %q", initialCSINode.Name))

			type state struct {
				CSINodes []storagev1.CSINode
			}

			err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*state, error) {
				csiNodeList, err := csiNodeClient.List(ctx, metav1.ListOptions{LabelSelector: patchedSelector})
				if err != nil {
					return nil, fmt.Errorf("failed to list CSINode: %w", err)
				}
				return &state{
					CSINodes: csiNodeList.Items,
				}, nil
			})).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(s *state) (func() string, error) {
				if len(s.CSINodes) == 0 {
					return nil, nil
				}
				return func() string {
					return fmt.Sprintf("Expected CSINode to be deleted, found %q", s.CSINodes[0].Name)
				}, nil
			}))
			framework.ExpectNoError(err, "Timeout while waiting to confirm CSINode %q deletion", initialCSINode.Name)

			replacementCSINode := storagev1.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "e2e-csinode-" + utilrand.String(5),
				},
			}

			ginkgo.By(fmt.Sprintf("Creating replacement csiNode %q", replacementCSINode.Name))
			secondCSINode, err := csiNodeClient.Create(ctx, &replacementCSINode, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create csiNode %q", replacementCSINode.Name)

			ginkgo.By(fmt.Sprintf("Getting replacement csiNode %q", replacementCSINode.Name))
			retrievedCSINode, err = csiNodeClient.Get(ctx, secondCSINode.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to retrieve CSINode %q", replacementCSINode.Name)
			gomega.Expect(retrievedCSINode.Name).To(gomega.Equal(secondCSINode.Name), "Checking that the retrieved name has been found")

			ginkgo.By(fmt.Sprintf("Updating replacement csiNode %q", retrievedCSINode.Name))
			var updatedCSINode *storagev1.CSINode

			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				tmpCSINode, err := csiNodeClient.Get(ctx, retrievedCSINode.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "Unable to get %q", replacementCSINode.Name)
				tmpCSINode.Labels = map[string]string{replacementCSINode.Name: "updated"}
				updatedCSINode, err = csiNodeClient.Update(ctx, tmpCSINode, metav1.UpdateOptions{})

				return err
			})
			framework.ExpectNoError(err, "failed to update %q", replacementCSINode.Name)
			gomega.Expect(updatedCSINode.Labels).To(gomega.HaveKeyWithValue(secondCSINode.Name, "updated"), "Checking that updated label has been applied")

			updatedSelector := labels.Set{retrievedCSINode.Name: "updated"}.AsSelector().String()
			ginkgo.By(fmt.Sprintf("DeleteCollection of CSINodes with %q label", updatedSelector))
			err = csiNodeClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: updatedSelector})
			framework.ExpectNoError(err, "failed to delete csiNode Colllection")

			ginkgo.By(fmt.Sprintf("Confirm deletion of replacement csiNode with LabelSelector %q", updatedSelector))

			err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*state, error) {
				csiNodeList, err := csiNodeClient.List(ctx, metav1.ListOptions{LabelSelector: updatedSelector})
				if err != nil {
					return nil, fmt.Errorf("failed to list CSINode: %w", err)
				}
				return &state{
					CSINodes: csiNodeList.Items,
				}, nil
			})).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(s *state) (func() string, error) {
				if len(s.CSINodes) == 0 {
					return nil, nil
				}
				return func() string {
					return fmt.Sprintf("Expected CSINode to be deleted, found %q", s.CSINodes[0].Name)
				}, nil
			}))
			framework.ExpectNoError(err, "Timeout while waiting to confirm CSINode %q deletion", replacementCSINode.Name)
		})
	})
})
