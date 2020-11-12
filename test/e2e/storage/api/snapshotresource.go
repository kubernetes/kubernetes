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

package api

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// snapshot CRD api group
const snapshotGroup = "snapshot.storage.k8s.io"

// snapshot CRD api version
const snapshotAPIVersion = "snapshot.storage.k8s.io/v1"

var (
	// SnapshotGVR is GroupVersionResource for volumesnapshots
	SnapshotGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1", Resource: "volumesnapshots"}
	// SnapshotClassGVR is GroupVersionResource for volumesnapshotclasses
	SnapshotClassGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1", Resource: "volumesnapshotclasses"}
	// SnapshotContentGVR is GroupVersionResource for volumesnapshotcontents
	SnapshotContentGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1", Resource: "volumesnapshotcontents"}
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
func CreateSnapshot(sDriver SnapshottableTestDriver, config *PerTestConfig, pattern TestPattern, pvcName string, pvcNamespace string) (*unstructured.Unstructured, *unstructured.Unstructured) {
	defer ginkgo.GinkgoRecover()
	var err error
	if pattern.SnapshotType != DynamicCreatedSnapshot && pattern.SnapshotType != PreprovisionedCreatedSnapshot {
		err = fmt.Errorf("SnapshotType must be set to either DynamicCreatedSnapshot or PreprovisionedCreatedSnapshot")
		framework.ExpectNoError(err)
	}
	dc := config.Framework.DynamicClient

	ginkgo.By("creating a SnapshotClass")
	sclass := sDriver.GetSnapshotClass(config)
	if sclass == nil {
		framework.Failf("Failed to get snapshot class based on test config")
	}
	sclass.Object["deletionPolicy"] = pattern.SnapshotDeletionPolicy.String()

	sclass, err = dc.Resource(SnapshotClassGVR).Create(context.TODO(), sclass, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	sclass, err = dc.Resource(SnapshotClassGVR).Get(context.TODO(), sclass.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("creating a dynamic VolumeSnapshot")
	// prepare a dynamically provisioned volume snapshot with certain data
	snapshot := getSnapshot(pvcName, pvcNamespace, sclass.GetName())

	snapshot, err = dc.Resource(SnapshotGVR).Namespace(snapshot.GetNamespace()).Create(context.TODO(), snapshot, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	return sclass, snapshot
}

// GetSnapshotContentFromSnapshot returns the VolumeSnapshotContent object Bound to a
// given VolumeSnapshot
func GetSnapshotContentFromSnapshot(dc dynamic.Interface, snapshot *unstructured.Unstructured) *unstructured.Unstructured {
	defer ginkgo.GinkgoRecover()
	err := WaitForSnapshotReady(dc, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	framework.ExpectNoError(err)

	vs, err := dc.Resource(SnapshotGVR).Namespace(snapshot.GetNamespace()).Get(context.TODO(), snapshot.GetName(), metav1.GetOptions{})

	snapshotStatus := vs.Object["status"].(map[string]interface{})
	snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
	framework.Logf("received snapshotStatus %v", snapshotStatus)
	framework.Logf("snapshotContentName %s", snapshotContentName)
	framework.ExpectNoError(err)

	vscontent, err := dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	return vscontent

}

// CreateSnapshotResource creates a snapshot resource for the current test. It knows how to deal with
// different test pattern snapshot provisioning and deletion policy
func CreateSnapshotResource(sDriver SnapshottableTestDriver, config *PerTestConfig, pattern TestPattern, pvcName string, pvcNamespace string) *SnapshotResource {
	var err error
	r := SnapshotResource{
		Config:  config,
		Pattern: pattern,
	}
	r.Vsclass, r.Vs = CreateSnapshot(sDriver, config, pattern, pvcName, pvcNamespace)

	dc := r.Config.Framework.DynamicClient

	r.Vscontent = GetSnapshotContentFromSnapshot(dc, r.Vs)

	if pattern.SnapshotType == PreprovisionedCreatedSnapshot {
		// prepare a pre-provisioned VolumeSnapshotContent with certain data
		// Because this could be run with an external CSI driver, we have no way
		// to pre-provision the snapshot as we normally would using their API.
		// We instead dynamically take a snapshot (above step), delete the old snapshot,
		// and create another snapshot using the first snapshot's snapshot handle.

		ginkgo.By("updating the snapshot content deletion policy to retain")
		r.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Retain"

		r.Vscontent, err = dc.Resource(SnapshotContentGVR).Update(context.TODO(), r.Vscontent, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("recording the volume handle and snapshotHandle")
		snapshotHandle := r.Vscontent.Object["status"].(map[string]interface{})["snapshotHandle"].(string)
		framework.Logf("Recording snapshot handle: %s", snapshotHandle)
		csiDriverName := r.Vsclass.Object["driver"].(string)

		// If the deletion policy is retain on vscontent:
		// when vs is deleted vscontent will not be deleted
		// when the vscontent is manually deleted then the underlying snapshot resource will not be deleted.
		// We exploit this to create a snapshot resource from which we can create a preprovisioned snapshot
		ginkgo.By("deleting the snapshot and snapshot content")
		err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Delete(context.TODO(), r.Vs.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			err = nil
		}
		framework.ExpectNoError(err)

		ginkgo.By("checking the Snapshot has been deleted")
		err = utils.WaitForNamespacedGVRDeletion(dc, SnapshotGVR, r.Vs.GetName(), r.Vs.GetNamespace(), framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)

		err = dc.Resource(SnapshotContentGVR).Delete(context.TODO(), r.Vscontent.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			err = nil
		}
		framework.ExpectNoError(err)

		ginkgo.By("checking the Snapshot content has been deleted")
		err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, r.Vscontent.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("creating a snapshot content with the snapshot handle")
		uuid := uuid.NewUUID()

		snapName := getPreProvisionedSnapshotName(uuid)
		snapcontentName := getPreProvisionedSnapshotContentName(uuid)

		r.Vscontent = getPreProvisionedSnapshotContent(snapcontentName, snapName, pvcNamespace, snapshotHandle, pattern.SnapshotDeletionPolicy.String(), csiDriverName)
		r.Vscontent, err = dc.Resource(SnapshotContentGVR).Create(context.TODO(), r.Vscontent, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating a snapshot with that snapshot content")
		r.Vs = getPreProvisionedSnapshot(snapName, pvcNamespace, snapcontentName)
		r.Vs, err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Create(context.TODO(), r.Vs, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = WaitForSnapshotReady(dc, r.Vs.GetNamespace(), r.Vs.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("getting the snapshot and snapshot content")
		r.Vs, err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Get(context.TODO(), r.Vs.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)

		r.Vscontent, err = dc.Resource(SnapshotContentGVR).Get(context.TODO(), r.Vscontent.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)
	}
	return &r
}

// CleanupResource cleans up the snapshot resource and ignores not found errors
func (sr *SnapshotResource) CleanupResource() error {
	var err error
	var cleanupErrs []error

	dc := sr.Config.Framework.DynamicClient

	if sr.Vs != nil {
		framework.Logf("deleting snapshot %q/%q", sr.Vs.GetNamespace(), sr.Vs.GetName())

		sr.Vs, err = dc.Resource(SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Get(context.TODO(), sr.Vs.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			snapshotStatus := sr.Vs.Object["status"].(map[string]interface{})
			snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
			framework.Logf("received snapshotStatus %v", snapshotStatus)
			framework.Logf("snapshotContentName %s", snapshotContentName)

			boundVsContent, err := dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
			switch {
			case err == nil:
				if boundVsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
					// The purpose of this block is to prevent physical snapshotContent leaks.
					// We must update the SnapshotContent to have Delete Deletion policy,
					// or else the physical snapshot content will be leaked.
					boundVsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
					boundVsContent, err = dc.Resource(SnapshotContentGVR).Update(context.TODO(), boundVsContent, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = dc.Resource(SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Delete(context.TODO(), sr.Vs.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, boundVsContent.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
				framework.ExpectNoError(err)

			case apierrors.IsNotFound(err):
				// the volume snapshot is not bound to snapshot content yet
				err = dc.Resource(SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Delete(context.TODO(), sr.Vs.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForNamespacedGVRDeletion(dc, SnapshotGVR, sr.Vs.GetName(), sr.Vs.GetNamespace(), framework.Poll, framework.SnapshotDeleteTimeout)
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

		sr.Vscontent, err = dc.Resource(SnapshotContentGVR).Get(context.TODO(), sr.Vscontent.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			if sr.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
				// The purpose of this block is to prevent physical snapshotContent leaks.
				// We must update the SnapshotContent to have Delete Deletion policy,
				// or else the physical snapshot content will be leaked.
				sr.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
				sr.Vscontent, err = dc.Resource(SnapshotContentGVR).Update(context.TODO(), sr.Vscontent, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = dc.Resource(SnapshotContentGVR).Delete(context.TODO(), sr.Vscontent.GetName(), metav1.DeleteOptions{})
			if apierrors.IsNotFound(err) {
				err = nil
			}
			framework.ExpectNoError(err)

			err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, sr.Vscontent.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
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
		err = dc.Resource(SnapshotClassGVR).Delete(context.TODO(), sr.Vsclass.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting snapshot class %q. Error: %v", sr.Vsclass.GetName(), err)
		}
		err = utils.WaitForGVRDeletion(dc, SnapshotClassGVR, sr.Vsclass.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)
	}
	return utilerrors.NewAggregate(cleanupErrs)
}

// DeleteAndWaitSnapshot deletes a VolumeSnapshot and waits for it to be deleted or until timeout occurs, whichever comes first
func DeleteAndWaitSnapshot(dc dynamic.Interface, ns string, snapshotName string, poll, timeout time.Duration) error {
	var err error
	ginkgo.By("deleting the snapshot")
	err = dc.Resource(SnapshotGVR).Namespace(ns).Delete(context.TODO(), snapshotName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}

	ginkgo.By("checking the Snapshot has been deleted")
	err = utils.WaitForNamespacedGVRDeletion(dc, SnapshotGVR, ns, snapshotName, poll, timeout)

	return err
}

func getSnapshot(claimName string, ns, snapshotClassName string) *unstructured.Unstructured {
	snapshot := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshot",
			"apiVersion": snapshotAPIVersion,
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
			"apiVersion": snapshotAPIVersion,
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
func getPreProvisionedSnapshotContent(snapcontentName, snapshotName, snapshotNamespace, snapshotHandle, deletionPolicy, csiDriverName string) *unstructured.Unstructured {
	snapshotContent := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshotContent",
			"apiVersion": snapshotAPIVersion,
			"metadata": map[string]interface{}{
				"name": snapcontentName,
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

// WaitForSnapshotReady waits for a VolumeSnapshot to be ready to use or until timeout occurs, whichever comes first.
func WaitForSnapshotReady(c dynamic.Interface, ns string, snapshotName string, poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for VolumeSnapshot %s to become ready", timeout, snapshotName)

	if successful := utils.WaitUntil(poll, timeout, func() bool {
		snapshot, err := c.Resource(SnapshotGVR).Namespace(ns).Get(context.TODO(), snapshotName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed to get snapshot %q, retrying in %v. Error: %v", snapshotName, poll, err)
			return false
		}

		status := snapshot.Object["status"]
		if status == nil {
			framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
			return false
		}
		value := status.(map[string]interface{})
		if value["readyToUse"] == true {
			framework.Logf("VolumeSnapshot %s found and is ready", snapshotName)
			return true
		}

		framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
		return false
	}); successful {
		return nil
	}

	return fmt.Errorf("VolumeSnapshot %s is not ready within %v", snapshotName, timeout)
}
