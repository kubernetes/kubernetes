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
	"math/rand"
	"time"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
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

var _ = utils.SIGDescribe("VolumeAttachment", func() {

	f := framework.NewDefaultFramework("volumeattachment")

	ginkgo.Describe("Conformance", func() {

		/*
			Release: v1.30
			Testname: VolumeAttachment, lifecycle
			Description: Creating an initial VolumeAttachment MUST succeed. Reading the VolumeAttachment
			MUST succeed with with required name retrieved. Patching a VolumeAttachment MUST
			succeed with its new label found. Listing VolumeAttachment with a labelSelector
			MUST succeed with a single item retrieved. Deleting a VolumeAttachment MUST succeed
			and it MUST be confirmed. Creating a second VolumeAttachment MUST succeed. Updating
			the second VolumentAttachment with a new label MUST succeed with its new label
			found. Creating a third VolumeAttachment MUST succeed. Updating the third VolumentAttachment
			with a new label MUST succeed with its new label found. Deleting both VolumeAttachments
			via deleteCollection MUST succeed and it MUST be confirmed.
		*/
		framework.ConformanceIt("should run through the lifecycle of a VolumeAttachment", func(ctx context.Context) {

			vaClient := f.ClientSet.StorageV1().VolumeAttachments()

			firstVA, vaNodeName := createVolumeAttachment(f, ctx)
			ginkgo.By(fmt.Sprintf("Get VolumeAttachment %q on node %q", firstVA, vaNodeName))
			retrievedVA, err := vaClient.Get(ctx, firstVA, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get VolumeAttachment %q", firstVA)
			gomega.Expect(retrievedVA.Name).To(gomega.Equal(firstVA), "Checking that retrieved VolumeAttachment has the correct name")
			gomega.Expect(retrievedVA).To(apimachineryutils.HaveValidResourceVersion())

			ginkgo.By(fmt.Sprintf("Patch VolumeAttachment %q on node %q", firstVA, vaNodeName))
			payload := "{\"metadata\":{\"labels\":{\"" + retrievedVA.Name + "\":\"patched\"}}}"
			patchedVA, err := vaClient.Patch(ctx, retrievedVA.Name, types.MergePatchType, []byte(payload), metav1.PatchOptions{})
			framework.ExpectNoError(err, "failed to patch PV %q", firstVA)
			gomega.Expect(patchedVA.Labels).To(gomega.HaveKeyWithValue(patchedVA.Name, "patched"), "Checking that patched label has been applied")
			gomega.Expect(resourceversion.CompareResourceVersion(retrievedVA.ResourceVersion, patchedVA.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

			patchedSelector := labels.Set{patchedVA.Name: "patched"}.AsSelector().String()
			ginkgo.By(fmt.Sprintf("List VolumeAttachments with %q label", patchedSelector))
			vaList, err := vaClient.List(ctx, metav1.ListOptions{LabelSelector: patchedSelector})
			framework.ExpectNoError(err, "failed to list VolumeAttachments")
			gomega.Expect(vaList.Items).To(gomega.HaveLen(1))

			ginkgo.By(fmt.Sprintf("Delete VolumeAttachment %q on node %q", firstVA, vaNodeName))
			err = vaClient.Delete(ctx, firstVA, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete VolumeAttachment %q", firstVA)

			ginkgo.By(fmt.Sprintf("Confirm deletion of VolumeAttachment %q on node %q", firstVA, vaNodeName))

			type state struct {
				VolumeAttachments []storagev1.VolumeAttachment
			}

			err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*state, error) {
				vaList, err := vaClient.List(ctx, metav1.ListOptions{LabelSelector: patchedSelector})
				if err != nil {
					return nil, fmt.Errorf("failed to list VolumeAttachment: %w", err)
				}
				return &state{
					VolumeAttachments: vaList.Items,
				}, nil
			})).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(s *state) (func() string, error) {
				if len(s.VolumeAttachments) == 0 {
					return nil, nil
				}
				return func() string {
					return fmt.Sprintf("Expected VolumeAttachment to be deleted, found %q", s.VolumeAttachments[0].Name)
				}, nil
			}))
			framework.ExpectNoError(err, "Timeout while waiting to confirm VolumeAttachment %q deletion", firstVA)

			secondVA, vaNodeName := createVolumeAttachment(f, ctx)
			updatedLabel := map[string]string{"va-e2e": "updated"}
			updatedSelector := labels.Set{"va-e2e": "updated"}.AsSelector().String()
			ginkgo.By(fmt.Sprintf("Update the VolumeAttachment %q on node %q with label %q", secondVA, vaNodeName, updatedSelector))
			var updatedVA *storagev1.VolumeAttachment

			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				currentVA, err := vaClient.Get(ctx, secondVA, metav1.GetOptions{})
				framework.ExpectNoError(err, "failed to get VolumeAttachment %q", patchedVA.Name)
				currentVA.Labels = updatedLabel
				updatedVA, err = vaClient.Update(ctx, currentVA, metav1.UpdateOptions{})

				return err
			})
			framework.ExpectNoError(err, "failed to update VolumeAttachment %q on node %q", secondVA, vaNodeName)
			gomega.Expect(updatedVA.Labels).To(gomega.HaveKeyWithValue("va-e2e", "updated"), "Checking that updated label has been applied")

			thirdVA, vaNodeName := createVolumeAttachment(f, ctx)
			ginkgo.By(fmt.Sprintf("Update the VolumeAttachment %q on node %q with label %q", thirdVA, vaNodeName, updatedSelector))

			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				currentVA, err := vaClient.Get(ctx, thirdVA, metav1.GetOptions{})
				framework.ExpectNoError(err, "failed to get VolumeAttachment %q", patchedVA.Name)
				currentVA.Labels = updatedLabel
				updatedVA, err = vaClient.Update(ctx, currentVA, metav1.UpdateOptions{})

				return err
			})
			framework.ExpectNoError(err, "failed to update VolumeAttachment %q on node %q", thirdVA, vaNodeName)
			gomega.Expect(updatedVA.Labels).To(gomega.HaveKeyWithValue("va-e2e", "updated"), "Checking that updated label has been applied")

			ginkgo.By(fmt.Sprintf("DeleteCollection of VolumeAttachments with %q label", updatedSelector))
			err = vaClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: updatedSelector})
			framework.ExpectNoError(err, "failed to delete VolumeAttachment collection")

			ginkgo.By(fmt.Sprintf("Confirm deleteCollection of VolumeAttachments with %q label", updatedSelector))

			err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*state, error) {
				vaList, err := vaClient.List(ctx, metav1.ListOptions{LabelSelector: updatedSelector})
				if err != nil {
					return nil, fmt.Errorf("failed to list VolumeAttachment: %w", err)
				}
				return &state{
					VolumeAttachments: vaList.Items,
				}, nil
			})).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(s *state) (func() string, error) {
				if len(s.VolumeAttachments) == 0 {
					return nil, nil
				}
				return func() string {
					list := []string{}
					for _, va := range s.VolumeAttachments {
						list = append(list, va.Name)
					}

					return fmt.Sprintf("Expected VolumeAttachment(s) to be deleted, found %v", list)
				}, nil
			}))
			framework.ExpectNoError(err, "Timeout while waiting to confirm deletion of all VolumeAttachments")
		})

		/*
			Release: v1.32
			Testname: VolumeAttachment, apply changes to a volumeattachment status
			Description: Creating an initial VolumeAttachment MUST succeed. Patching a VolumeAttachment
			MUST succeed with its new label found. Reading VolumeAttachment status MUST succeed
			with its attached status being false. Patching the VolumeAttachment status MUST
			succeed with its attached status being true. Updating the VolumeAttachment status
			MUST succeed with its attached status being false. Deleting a VolumeAttachment
			MUST succeed and it MUST be confirmed.
		*/
		framework.ConformanceIt("should apply changes to a volumeattachment status", func(ctx context.Context) {

			vaClient := f.ClientSet.StorageV1().VolumeAttachments()

			randUID := "e2e-" + utilrand.String(5)
			vaName := "va-" + randUID
			pvName := "pv-" + randUID

			nodes, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err, "failed to list nodes")
			randNode := rand.Intn(len(nodes.Items))
			vaNodeName := nodes.Items[randNode].Name
			vaAttachStatus := false

			ginkgo.By(fmt.Sprintf("Create VolumeAttachment %q on node %q", vaName, vaNodeName))
			initialVA := NewVolumeAttachment(vaName, pvName, vaNodeName, vaAttachStatus)

			createdVA, err := vaClient.Create(ctx, initialVA, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create VolumeAttachment %q", vaName)
			gomega.Expect(createdVA.Name).To(gomega.Equal(vaName), "Checking that the created VolumeAttachment has the correct name")

			ginkgo.By(fmt.Sprintf("Patch VolumeAttachment %q on node %q", vaName, vaNodeName))
			payload := "{\"metadata\":{\"labels\":{\"" + createdVA.Name + "\":\"patched\"}}}"
			patchedVA, err := vaClient.Patch(ctx, createdVA.Name, types.MergePatchType, []byte(payload), metav1.PatchOptions{})
			framework.ExpectNoError(err, "failed to patch PV %q", vaName)
			gomega.Expect(patchedVA.Labels).To(gomega.HaveKeyWithValue(patchedVA.Name, "patched"), "Checking that patched label has been applied")
			patchedSelector := labels.Set{patchedVA.Name: "patched"}.AsSelector().String()

			ginkgo.By(fmt.Sprintf("Reading %q Status", patchedVA.Name))
			vaResource := schema.GroupVersionResource{Group: "storage.k8s.io", Version: "v1", Resource: "volumeattachments"}
			vaUnstructured, err := f.DynamicClient.Resource(vaResource).Get(ctx, patchedVA.Name, metav1.GetOptions{}, "status")
			framework.ExpectNoError(err, "Failed to fetch the status of %q VolumeAttachment", patchedVA.Name)

			retrievedVA := &storagev1.VolumeAttachment{}
			err = runtime.DefaultUnstructuredConverter.FromUnstructured(vaUnstructured.UnstructuredContent(), &retrievedVA)
			framework.ExpectNoError(err, "Failed to retrieve %q status.", patchedVA.Name)
			gomega.Expect(retrievedVA.Status.Attached).To(gomega.BeFalseBecause("Checking that the VolumeAttachment status has been read"))

			ginkgo.By(fmt.Sprintf("Patching %q Status", patchedVA.Name))
			statusPayload := []byte(`{"status":{"attached":true}}`)

			vaPatched, err := vaClient.Patch(ctx, patchedVA.Name, types.MergePatchType, statusPayload, metav1.PatchOptions{}, "status")
			framework.ExpectNoError(err, "Failed to patch status.")
			gomega.Expect(vaPatched.Status.Attached).To(gomega.BeTrueBecause("Checking that the VolumeAttachment status has been patched"))
			framework.Logf("%q Status.Attached: %v", vaPatched.Name, vaPatched.Status.Attached)

			ginkgo.By(fmt.Sprintf("Updating %q Status", vaPatched.Name))
			var statusToUpdate, updatedVA *storagev1.VolumeAttachment

			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				statusToUpdate, err = vaClient.Get(ctx, vaPatched.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "Unable to retrieve VolumeAttachment %q", vaPatched.Name)

				statusToUpdate.Status.Attached = false

				updatedVA, err = vaClient.UpdateStatus(ctx, statusToUpdate, metav1.UpdateOptions{})
				return err
			})
			framework.ExpectNoError(err, "Failed to update status.")
			gomega.Expect(updatedVA.Status.Attached).To(gomega.BeFalseBecause("Checking that the VolumeAttachment status has been updated"))
			framework.Logf("%q Status.Attached: %v", updatedVA.Name, updatedVA.Status.Attached)

			ginkgo.By(fmt.Sprintf("Delete VolumeAttachment %q on node %q", vaName, vaNodeName))
			err = vaClient.Delete(ctx, vaName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete VolumeAttachment %q", vaName)

			ginkgo.By(fmt.Sprintf("Confirm deletion of VolumeAttachment %q on node %q", vaName, vaNodeName))

			type state struct {
				VolumeAttachments []storagev1.VolumeAttachment
			}

			err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*state, error) {
				vaList, err := vaClient.List(ctx, metav1.ListOptions{LabelSelector: patchedSelector})
				if err != nil {
					return nil, fmt.Errorf("failed to list VolumeAttachment: %w", err)
				}
				return &state{
					VolumeAttachments: vaList.Items,
				}, nil
			})).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(s *state) (func() string, error) {
				if len(s.VolumeAttachments) == 0 {
					return nil, nil
				}
				return func() string {
					return fmt.Sprintf("Expected VolumeAttachment to be deleted, found %q", s.VolumeAttachments[0].Name)
				}, nil
			}))
			framework.ExpectNoError(err, "Timeout while waiting to confirm VolumeAttachment %q deletion", vaName)
		})

	})
})

func NewVolumeAttachment(vaName, pvName, nodeName string, status bool) *storagev1.VolumeAttachment {
	return &storagev1.VolumeAttachment{

		ObjectMeta: metav1.ObjectMeta{
			UID:  types.UID(vaName),
			Name: vaName,
		},
		Spec: storagev1.VolumeAttachmentSpec{
			Attacher: "e2e-test.storage.k8s.io",
			NodeName: nodeName,
			Source: storagev1.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
		},
		Status: storagev1.VolumeAttachmentStatus{
			Attached: status,
		},
	}
}

func createVolumeAttachment(f *framework.Framework, ctx context.Context) (string, string) {

	randUID := "e2e-" + utilrand.String(5)
	vaName := "va-" + randUID
	pvName := "pv-" + randUID

	nodes, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "failed to list nodes")
	randNode := rand.Intn(len(nodes.Items))
	vaNodeName := nodes.Items[randNode].Name
	vaAttachStatus := false

	ginkgo.By(fmt.Sprintf("Create VolumeAttachment %q on node %q", vaName, vaNodeName))
	va := NewVolumeAttachment(vaName, pvName, vaNodeName, vaAttachStatus)

	createdVA, err := f.ClientSet.StorageV1().VolumeAttachments().Create(ctx, va, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create VolumeAttachment %q", vaName)
	gomega.Expect(createdVA.Name).To(gomega.Equal(vaName), "Checking that the created VolumeAttachment has the correct name")

	return createdVA.Name, vaNodeName
}
