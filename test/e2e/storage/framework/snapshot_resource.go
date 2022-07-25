/*
Copyright 2020 The Kubernetes Authors.

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

package framework

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// SnapshotResource represents a snapshot class, a snapshot and its bound snapshot contents for a specific test case
type SnapshotResource struct {
	Config  *PerTestConfig
	Pattern TestPattern

	Vs        *unstructured.Unstructured
	Vscontent *unstructured.Unstructured
	Vsclass   *unstructured.Unstructured
}

// CreateSnapshot creates a VolumeSnapshotClass with given SnapshotDeletionPolicy and a VolumeSnapshot
// from the VolumeSnapshotClass using a dynamic client.
// Returns the unstructured VolumeSnapshotClass and VolumeSnapshot objects.
func CreateSnapshot(sDriver SnapshottableTestDriver, config *PerTestConfig, pattern TestPattern, pvcName string, pvcNamespace string, timeouts *framework.TimeoutContext, parameters map[string]string) (*unstructured.Unstructured, *unstructured.Unstructured) {
	defer ginkgo.GinkgoRecover()
	var err error
	if pattern.SnapshotType != DynamicCreatedSnapshot && pattern.SnapshotType != PreprovisionedCreatedSnapshot {
		err = fmt.Errorf("SnapshotType must be set to either DynamicCreatedSnapshot or PreprovisionedCreatedSnapshot")
		framework.ExpectNoError(err)
	}
	dc := config.Framework.DynamicClient

	ginkgo.By("creating a SnapshotClass")
	sclass := sDriver.GetSnapshotClass(config, parameters)
	if sclass == nil {
		framework.Failf("Failed to get snapshot class based on test config")
	}
	sclass.Object["deletionPolicy"] = pattern.SnapshotDeletionPolicy.String()

	sclass, err = dc.Resource(utils.SnapshotClassGVR).Create(context.TODO(), sclass, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	sclass, err = dc.Resource(utils.SnapshotClassGVR).Get(context.TODO(), sclass.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("creating a dynamic VolumeSnapshot")
	// prepare a dynamically provisioned volume snapshot with certain data
	snapshot := getSnapshot(pvcName, pvcNamespace, sclass.GetName())

	snapshot, err = dc.Resource(utils.SnapshotGVR).Namespace(snapshot.GetNamespace()).Create(context.TODO(), snapshot, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	return sclass, snapshot
}

// CreateSnapshotResource creates a snapshot resource for the current test. It knows how to deal with
// different test pattern snapshot provisioning and deletion policy
func CreateSnapshotResource(sDriver SnapshottableTestDriver, config *PerTestConfig, pattern TestPattern, pvcName string, pvcNamespace string, timeouts *framework.TimeoutContext, parameters map[string]string) *SnapshotResource {
	var err error
	r := SnapshotResource{
		Config:  config,
		Pattern: pattern,
	}
	r.Vsclass, r.Vs = CreateSnapshot(sDriver, config, pattern, pvcName, pvcNamespace, timeouts, parameters)

	dc := r.Config.Framework.DynamicClient

	r.Vscontent = utils.GetSnapshotContentFromSnapshot(dc, r.Vs)

	if pattern.SnapshotType == PreprovisionedCreatedSnapshot {
		// prepare a pre-provisioned VolumeSnapshotContent with certain data
		// Because this could be run with an external CSI driver, we have no way
		// to pre-provision the snapshot as we normally would using their API.
		// We instead dynamically take a snapshot (above step), delete the old snapshot,
		// and create another snapshot using the first snapshot's snapshot handle.

		ginkgo.By("updating the snapshot content deletion policy to retain")
		r.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Retain"

		r.Vscontent, err = dc.Resource(utils.SnapshotContentGVR).Update(context.TODO(), r.Vscontent, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("recording properties of the preprovisioned snapshot")
		snapshotHandle := r.Vscontent.Object["status"].(map[string]interface{})["snapshotHandle"].(string)
		framework.Logf("Recording snapshot content handle: %s", snapshotHandle)
		snapshotContentAnnotations := r.Vscontent.GetAnnotations()
		framework.Logf("Recording snapshot content annotations: %v", snapshotContentAnnotations)
		csiDriverName := r.Vsclass.Object["driver"].(string)
		framework.Logf("Recording snapshot driver: %s", csiDriverName)

		// If the deletion policy is retain on vscontent:
		// when vs is deleted vscontent will not be deleted
		// when the vscontent is manually deleted then the underlying snapshot resource will not be deleted.
		// We exploit this to create a snapshot resource from which we can create a preprovisioned snapshot
		ginkgo.By("deleting the snapshot and snapshot content")
		err = dc.Resource(utils.SnapshotGVR).Namespace(r.Vs.GetNamespace()).Delete(context.TODO(), r.Vs.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			err = nil
		}
		framework.ExpectNoError(err)

		ginkgo.By("checking the Snapshot has been deleted")
		err = utils.WaitForNamespacedGVRDeletion(dc, utils.SnapshotGVR, r.Vs.GetName(), r.Vs.GetNamespace(), framework.Poll, timeouts.SnapshotDelete)
		framework.ExpectNoError(err)

		err = dc.Resource(utils.SnapshotContentGVR).Delete(context.TODO(), r.Vscontent.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			err = nil
		}
		framework.ExpectNoError(err)

		ginkgo.By("checking the Snapshot content has been deleted")
		err = utils.WaitForGVRDeletion(dc, utils.SnapshotContentGVR, r.Vscontent.GetName(), framework.Poll, timeouts.SnapshotDelete)
		framework.ExpectNoError(err)

		ginkgo.By("creating a snapshot content with the snapshot handle")
		uuid := uuid.NewUUID()

		snapName := getPreProvisionedSnapshotName(uuid)
		snapcontentName := getPreProvisionedSnapshotContentName(uuid)

		r.Vscontent = getPreProvisionedSnapshotContent(snapcontentName, snapshotContentAnnotations, snapName, pvcNamespace, snapshotHandle, pattern.SnapshotDeletionPolicy.String(), csiDriverName)
		r.Vscontent, err = dc.Resource(utils.SnapshotContentGVR).Create(context.TODO(), r.Vscontent, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating a snapshot with that snapshot content")
		r.Vs = getPreProvisionedSnapshot(snapName, pvcNamespace, snapcontentName)
		r.Vs, err = dc.Resource(utils.SnapshotGVR).Namespace(r.Vs.GetNamespace()).Create(context.TODO(), r.Vs, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = utils.WaitForSnapshotReady(dc, r.Vs.GetNamespace(), r.Vs.GetName(), framework.Poll, timeouts.SnapshotCreate)
		framework.ExpectNoError(err)

		ginkgo.By("getting the snapshot and snapshot content")
		r.Vs, err = dc.Resource(utils.SnapshotGVR).Namespace(r.Vs.GetNamespace()).Get(context.TODO(), r.Vs.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)

		r.Vscontent, err = dc.Resource(utils.SnapshotContentGVR).Get(context.TODO(), r.Vscontent.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)
	}
	return &r
}

// CleanupResource cleans up the snapshot resource and ignores not found errors
func (sr *SnapshotResource) CleanupResource(timeouts *framework.TimeoutContext) error {
	var err error
	var cleanupErrs []error

	dc := sr.Config.Framework.DynamicClient

	if sr.Vs != nil {
		framework.Logf("deleting snapshot %q/%q", sr.Vs.GetNamespace(), sr.Vs.GetName())

		sr.Vs, err = dc.Resource(utils.SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Get(context.TODO(), sr.Vs.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			snapshotStatus := sr.Vs.Object["status"].(map[string]interface{})
			snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
			framework.Logf("received snapshotStatus %v", snapshotStatus)
			framework.Logf("snapshotContentName %s", snapshotContentName)

			boundVsContent, err := dc.Resource(utils.SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
			switch {
			case err == nil:
				if boundVsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
					// The purpose of this block is to prevent physical snapshotContent leaks.
					// We must update the SnapshotContent to have Delete Deletion policy,
					// or else the physical snapshot content will be leaked.
					boundVsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
					boundVsContent, err = dc.Resource(utils.SnapshotContentGVR).Update(context.TODO(), boundVsContent, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = dc.Resource(utils.SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Delete(context.TODO(), sr.Vs.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForGVRDeletion(dc, utils.SnapshotContentGVR, boundVsContent.GetName(), framework.Poll, timeouts.SnapshotDelete)
				framework.ExpectNoError(err)

			case apierrors.IsNotFound(err):
				// the volume snapshot is not bound to snapshot content yet
				err = dc.Resource(utils.SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Delete(context.TODO(), sr.Vs.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForNamespacedGVRDeletion(dc, utils.SnapshotGVR, sr.Vs.GetName(), sr.Vs.GetNamespace(), framework.Poll, timeouts.SnapshotDelete)
				framework.ExpectNoError(err)
			default:
				cleanupErrs = append(cleanupErrs, err)
			}
		case apierrors.IsNotFound(err):
			// Hope that the underlying snapshot content and resource is gone already
		default:
			cleanupErrs = append(cleanupErrs, err)
		}
	}
	if sr.Vscontent != nil {
		framework.Logf("deleting snapshot content %q", sr.Vscontent.GetName())

		sr.Vscontent, err = dc.Resource(utils.SnapshotContentGVR).Get(context.TODO(), sr.Vscontent.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			if sr.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
				// The purpose of this block is to prevent physical snapshotContent leaks.
				// We must update the SnapshotContent to have Delete Deletion policy,
				// or else the physical snapshot content will be leaked.
				sr.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
				sr.Vscontent, err = dc.Resource(utils.SnapshotContentGVR).Update(context.TODO(), sr.Vscontent, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = dc.Resource(utils.SnapshotContentGVR).Delete(context.TODO(), sr.Vscontent.GetName(), metav1.DeleteOptions{})
			if apierrors.IsNotFound(err) {
				err = nil
			}
			framework.ExpectNoError(err)

			err = utils.WaitForGVRDeletion(dc, utils.SnapshotContentGVR, sr.Vscontent.GetName(), framework.Poll, timeouts.SnapshotDelete)
			framework.ExpectNoError(err)
		case apierrors.IsNotFound(err):
			// Hope the underlying physical snapshot resource has been deleted already
		default:
			cleanupErrs = append(cleanupErrs, err)
		}
	}
	if sr.Vsclass != nil {
		framework.Logf("deleting snapshot class %q", sr.Vsclass.GetName())
		// typically this snapshot class has already been deleted
		err = dc.Resource(utils.SnapshotClassGVR).Delete(context.TODO(), sr.Vsclass.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting snapshot class %q. Error: %v", sr.Vsclass.GetName(), err)
		}
		err = utils.WaitForGVRDeletion(dc, utils.SnapshotClassGVR, sr.Vsclass.GetName(), framework.Poll, timeouts.SnapshotDelete)
		framework.ExpectNoError(err)
	}
	return utilerrors.NewAggregate(cleanupErrs)
}

func getSnapshot(claimName string, ns, snapshotClassName string) *unstructured.Unstructured {
	snapshot := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshot",
			"apiVersion": utils.SnapshotAPIVersion,
			"metadata": map[string]interface{}{
				"generateName": "snapshot-",
				"namespace":    ns,
			},
			"spec": map[string]interface{}{
				"volumeSnapshotClassName": snapshotClassName,
				"source": map[string]interface{}{
					"persistentVolumeClaimName": claimName,
				},
			},
		},
	}

	return snapshot
}
func getPreProvisionedSnapshot(snapName, ns, snapshotContentName string) *unstructured.Unstructured {
	snapshot := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshot",
			"apiVersion": utils.SnapshotAPIVersion,
			"metadata": map[string]interface{}{
				"name":      snapName,
				"namespace": ns,
			},
			"spec": map[string]interface{}{
				"source": map[string]interface{}{
					"volumeSnapshotContentName": snapshotContentName,
				},
			},
		},
	}

	return snapshot
}
func getPreProvisionedSnapshotContent(snapcontentName string, snapshotContentAnnotations map[string]string, snapshotName, snapshotNamespace, snapshotHandle, deletionPolicy, csiDriverName string) *unstructured.Unstructured {
	snapshotContent := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshotContent",
			"apiVersion": utils.SnapshotAPIVersion,
			"metadata": map[string]interface{}{
				"name":        snapcontentName,
				"annotations": snapshotContentAnnotations,
			},
			"spec": map[string]interface{}{
				"source": map[string]interface{}{
					"snapshotHandle": snapshotHandle,
				},
				"volumeSnapshotRef": map[string]interface{}{
					"name":      snapshotName,
					"namespace": snapshotNamespace,
				},
				"driver":         csiDriverName,
				"deletionPolicy": deletionPolicy,
			},
		},
	}

	return snapshotContent
}

func getPreProvisionedSnapshotContentName(uuid types.UID) string {
	return fmt.Sprintf("pre-provisioned-snapcontent-%s", string(uuid))
}

func getPreProvisionedSnapshotName(uuid types.UID) string {
	return fmt.Sprintf("pre-provisioned-snapshot-%s", string(uuid))
}
