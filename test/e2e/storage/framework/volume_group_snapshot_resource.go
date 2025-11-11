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

package framework

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

func getVolumeGroupSnapshot(labels map[string]interface{}, ns, snapshotClassName string) *unstructured.Unstructured {
	snapshot := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeGroupSnapshot",
			"apiVersion": utils.VolumeGroupSnapshotAPIVersion,
			"metadata": map[string]interface{}{
				"generateName": "group-snapshot-",
				"namespace":    ns,
			},
			"spec": map[string]interface{}{
				"volumeGroupSnapshotClassName": snapshotClassName,
				"source": map[string]interface{}{
					"selector": map[string]interface{}{
						"matchLabels": labels,
					},
				},
			},
		},
	}

	return snapshot
}

// VolumeGroupSnapshotResource represents a volumegroupsnapshot class, a volumegroupsnapshot and its bound contents for a specific test case
type VolumeGroupSnapshotResource struct {
	Config  *PerTestConfig
	Pattern TestPattern

	VGS        *unstructured.Unstructured
	VGSContent *unstructured.Unstructured
	VGSClass   *unstructured.Unstructured
}

// CreateVolumeGroupSnapshot creates a VolumeGroupSnapshotClass with given SnapshotDeletionPolicy and a VolumeGroupSnapshot
// from the VolumeGroupSnapshotClass using a dynamic client.
// Returns the unstructured VolumeGroupSnapshotClass and VolumeGroupSnapshot objects.
func CreateVolumeGroupSnapshot(ctx context.Context, sDriver VolumeGroupSnapshottableTestDriver, config *PerTestConfig, pattern TestPattern, groupName string, pvcNamespace string, timeouts *framework.TimeoutContext, parameters map[string]string) (*unstructured.Unstructured, *unstructured.Unstructured, *unstructured.Unstructured) {
	defer ginkgo.GinkgoRecover()
	var err error
	if pattern.SnapshotType != VolumeGroupSnapshot {
		err = fmt.Errorf("SnapshotType must be set to VolumeGroupSnapshot")
		framework.ExpectNoError(err, "SnapshotType is set to VolumeGroupSnapshot")
	}
	dc := config.Framework.DynamicClient

	ginkgo.By("creating a VolumeGroupSnapshotClass")
	gsclass := sDriver.GetVolumeGroupSnapshotClass(ctx, config, parameters)
	if gsclass == nil {
		framework.Failf("Failed to get volume group snapshot class based on test config")
	}
	gsclass.Object["deletionPolicy"] = pattern.SnapshotDeletionPolicy.String()

	gsclass, err = dc.Resource(utils.VolumeGroupSnapshotClassGVR).Create(ctx, gsclass, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create volume group snapshot class")
	gsclass, err = dc.Resource(utils.VolumeGroupSnapshotClassGVR).Get(ctx, gsclass.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err, "Failed to get volume group snapshot class")

	ginkgo.By("creating a dynamic VolumeGroupSnapshot")
	// Prepare a dynamically provisioned group volume snapshot with certain data
	volumeGroupSnapshot := getVolumeGroupSnapshot(map[string]interface{}{
		"group": groupName,
	}, pvcNamespace, gsclass.GetName())

	volumeGroupSnapshot, err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(volumeGroupSnapshot.GetNamespace()).Create(ctx, volumeGroupSnapshot, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create volume group snapshot")
	ginkgo.By("Waiting for group snapshot to be ready")
	err = utils.WaitForVolumeGroupSnapshotReady(ctx, dc, volumeGroupSnapshot.GetNamespace(), volumeGroupSnapshot.GetName(), framework.Poll, timeouts.SnapshotCreate*10)
	framework.ExpectNoError(err, "Group snapshot is not ready to use within the timeout")
	ginkgo.By("Getting group snapshot and content")
	volumeGroupSnapshot, err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(volumeGroupSnapshot.GetNamespace()).Get(ctx, volumeGroupSnapshot.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err, "Failed to get volume group snapshot after creation")
	status := volumeGroupSnapshot.Object["status"]
	err = framework.Gomega().Expect(status).NotTo(gomega.BeNil())
	framework.ExpectNoError(err, "Failed to get status of volume group snapshot")
	vgscName := status.(map[string]interface{})["boundVolumeGroupSnapshotContentName"].(string)
	err = framework.Gomega().Expect(vgscName).NotTo(gomega.BeNil())
	framework.ExpectNoError(err, "Failed to get content name of volume group snapshot")
	vgsc, err := dc.Resource(utils.VolumeGroupSnapshotContentGVR).Get(ctx, vgscName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get content of group snapshot")
	return gsclass, volumeGroupSnapshot, vgsc
}

// CleanupResource deletes the VolumeGroupSnapshotClass and VolumeGroupSnapshot objects using a dynamic client,
// and ignores not found errors.
func (r *VolumeGroupSnapshotResource) CleanupResource(ctx context.Context, timeouts *framework.TimeoutContext) error {
	var err error
	var cleanupErrs []error

	dc := r.Config.Framework.DynamicClient
	if r.VGS != nil {
		framework.Logf("deleting groupSnapshot %q/%q", r.VGS.GetNamespace(), r.VGS.GetName())

		r.VGS, err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(r.VGS.GetNamespace()).Get(ctx, r.VGS.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			groupSnapshotStatus := r.VGS.Object["status"].(map[string]interface{})
			groupSnapshotContentName := groupSnapshotStatus["boundVolumeGroupSnapshotContentName"].(string)
			framework.Logf("received snapshotStatus %v", groupSnapshotStatus)
			framework.Logf("groupSnapshotContentName %s", groupSnapshotContentName)

			boundVgsContent, err := dc.Resource(utils.VolumeGroupSnapshotContentGVR).Get(ctx, groupSnapshotContentName, metav1.GetOptions{})
			switch {
			case err == nil:
				if boundVgsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
					// The purpose of this block is to prevent physical groupSnapshotContent leaks.
					// We must update the groupSnapshotContent to have Delete Deletion policy,
					// or else the physical group snapshot content will be leaked.
					boundVgsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
					boundVgsContent, err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Update(ctx, boundVgsContent, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(r.VGS.GetNamespace()).Delete(ctx, r.VGS.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotGVR, boundVgsContent.GetName(), framework.Poll, timeouts.SnapshotDelete)
				framework.ExpectNoError(err)

			case apierrors.IsNotFound(err):
				// the group snapshot is not bound to group snapshot content yet
				err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(r.VGS.GetNamespace()).Delete(ctx, r.VGS.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForNamespacedGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotGVR, r.VGS.GetName(), r.VGS.GetNamespace(), framework.Poll, timeouts.SnapshotDelete)
				framework.ExpectNoError(err)
			default:
				cleanupErrs = append(cleanupErrs, err)
			}
		case apierrors.IsNotFound(err):
			// Hope that the underlying snapshot contents and resources are gone already
		default:
			cleanupErrs = append(cleanupErrs, err)
		}
	}

	if r.VGSContent != nil {
		framework.Logf("deleting group snapshot content %q", r.VGSContent.GetName())

		r.VGSContent, err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Get(ctx, r.VGSContent.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			if r.VGSContent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
				// The purpose of this block is to prevent physical groupSnapshotContent leaks.
				// We must update the groupSnapshotContent to have Delete Deletion policy,
				// or else the physical group snapshot content will be leaked.
				r.VGSContent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
				r.VGSContent, err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Update(ctx, r.VGSContent, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Delete(ctx, r.VGSContent.GetName(), metav1.DeleteOptions{})
			if apierrors.IsNotFound(err) {
				err = nil
			}
			framework.ExpectNoError(err)

			err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotContentGVR, r.VGSContent.GetName(), framework.Poll, timeouts.SnapshotDelete)
			framework.ExpectNoError(err)
		case apierrors.IsNotFound(err):
			// Hope the underlying physical snapshot resource has been deleted already
		default:
			cleanupErrs = append(cleanupErrs, err)
		}
	}

	if r.VGSClass != nil {
		framework.Logf("deleting group snapshot class %q", r.VGSClass.GetName())
		// typically this snapshot class has already been deleted
		err = dc.Resource(utils.VolumeGroupSnapshotClassGVR).Delete(ctx, r.VGSClass.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting group snapshot class %q. Error: %v", r.VGSClass.GetName(), err)
		}
		err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotClassGVR, r.VGSClass.GetName(), framework.Poll, timeouts.SnapshotDelete)
		framework.ExpectNoError(err)
	}
	return utilerrors.NewAggregate(cleanupErrs)
}

// CreateVolumeGroupSnapshotResource creates a VolumeGroupSnapshotResource object with the given parameters.
func CreateVolumeGroupSnapshotResource(ctx context.Context, sDriver VolumeGroupSnapshottableTestDriver, config *PerTestConfig, pattern TestPattern, pvcName string, pvcNamespace string, timeouts *framework.TimeoutContext, parameters map[string]string) *VolumeGroupSnapshotResource {
	vgsClass, snapshot, vgsc := CreateVolumeGroupSnapshot(ctx, sDriver, config, pattern, pvcName, pvcNamespace, timeouts, parameters)
	vgs := &VolumeGroupSnapshotResource{
		Config:     config,
		Pattern:    pattern,
		VGS:        snapshot,
		VGSClass:   vgsClass,
		VGSContent: vgsc,
	}
	return vgs
}
