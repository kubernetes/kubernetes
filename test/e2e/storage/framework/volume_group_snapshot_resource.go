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
	"k8s.io/apimachinery/pkg/util/sets"
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

// CleanupVGS deletes the VolumeGroupSnapshot and ensures the bound VolumeGroupSnapshotContent has Delete policy.
// It waits for the VolumeGroupSnapshot, its owned VolumeSnapshots, VolumeSnapshotContents, and the bound
// VolumeGroupSnapshotContent to be fully deleted to prevent resource leaks.
func (r *VolumeGroupSnapshotResource) CleanupVGS(ctx context.Context, timeouts *framework.TimeoutContext) error {
	if r.VGS == nil {
		return nil
	}

	var cleanupErrs []error
	dc := r.Config.Framework.DynamicClient
	vgsNamespace := r.VGS.GetNamespace()
	vgsName := r.VGS.GetName()
	vgsUID := r.VGS.GetUID()
	framework.Logf("deleting groupSnapshot %q/%q and ensuring content has Delete policy", vgsNamespace, vgsName)

	vgs, err := dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(vgsNamespace).Get(ctx, vgsName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Hope that the underlying snapshot contents and resources are gone already
			return nil
		}
		return fmt.Errorf("failed to get VGS %q: %w", vgsName, err)
	}
	r.VGS = vgs

	// Filter snapshots owned by this VGS
	vss, err := dc.Resource(utils.SnapshotGVR).Namespace(vgsNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}

	contentNamesSet := sets.NewString()
	for _, vs := range vss.Items {
		for _, owner := range vs.GetOwnerReferences() {
			if owner.Kind == "VolumeGroupSnapshot" && owner.UID == vgsUID {
				if status, ok := vs.Object["status"].(map[string]interface{}); ok {
					if cName, ok := status["boundVolumeSnapshotContentName"].(string); ok && cName != "" {
						contentNamesSet.Insert(cName)
					}
				}
			}
		}
	}

	groupSnapshotStatus := r.VGS.Object["status"].(map[string]interface{})
	groupSnapshotContentName := groupSnapshotStatus["boundVolumeGroupSnapshotContentName"].(string)
	framework.Logf("received groupSnapshotStatus %v", groupSnapshotStatus)
	framework.Logf("groupSnapshotContentName %q", groupSnapshotContentName)

	boundVGSContent, err := dc.Resource(utils.VolumeGroupSnapshotContentGVR).Get(ctx, groupSnapshotContentName, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to get bound VGSContent %q: %w", boundVGSContent.GetName(), err)
	}

	// Ensure deletion policy is set to Delete to prevent leaks
	if boundVGSContent != nil {
		spec := boundVGSContent.Object["spec"].(map[string]interface{})
		if spec["deletionPolicy"] != "Delete" {
			spec["deletionPolicy"] = "Delete"
			boundVGSContent, err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Update(ctx, boundVGSContent, metav1.UpdateOptions{})
			if err != nil {
				return fmt.Errorf("failed to update VGSContent %q: %w", boundVGSContent.GetName(), err)
			}
		}
	}
	r.VGSContent = boundVGSContent

	// Delete the VolumeGroupSnapshot
	if err := dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(vgsNamespace).Delete(ctx, vgsName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to delete VGS %q: %w", vgsName, err)
	}

	// Wait for VolumeGroupSnapshot deleted
	if err := utils.WaitForNamespacedGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotGVR, vgsNamespace, vgsName, framework.Poll, timeouts.SnapshotDelete); err != nil {
		return fmt.Errorf("failed waiting for VGS %q deletion: %w", vgsName, err)
	}

	// Wait for VolumeSnapshots owned by this group to be gone
	if err := utils.WaitForOwnedResourcesDeleted(ctx, dc, utils.SnapshotGVR, vgsNamespace, vgsUID, framework.Poll, timeouts.SnapshotDelete); err != nil {
		return fmt.Errorf("failed waiting for owned snapshots deletion of VGS %q: %w", vgsName, err)
	}

	// Wait for all VolumeSnapshotsContents owned by this group to be gone
	for _, contentName := range contentNamesSet.List() {
		if err := utils.WaitForGVRDeletion(ctx, dc, utils.SnapshotContentGVR, contentName, framework.Poll, timeouts.SnapshotDelete); err != nil {
			cleanupErrs = append(cleanupErrs, fmt.Errorf("failed waiting for VSC %q deletion: %w", contentName, err))
		}
	}
	if len(cleanupErrs) > 0 {
		return utilerrors.NewAggregate(cleanupErrs)
	}

	// Wait for VolumeGroupSnapshotContent deleted
	if boundVGSContent != nil {
		err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotContentGVR, boundVGSContent.GetName(), framework.Poll, timeouts.SnapshotDelete)
		if err == nil {
			r.VGSContent = nil
		} else {
			return fmt.Errorf("failed waiting for VGSContent %q deletion: %w", vgsName, err)
		}
	}
	return nil
}

// CleanupVGSClass deletes the VolumeGroupSnapshotClass.
func (r *VolumeGroupSnapshotResource) CleanupVGSClass(ctx context.Context, timeouts *framework.TimeoutContext) error {
	if r.VGSClass == nil {
		return nil
	}

	dc := r.Config.Framework.DynamicClient
	vgsClassName := r.VGSClass.GetName()
	framework.Logf("deleting groupSnapshotClass %q", vgsClassName)

	err := dc.Resource(utils.VolumeGroupSnapshotClassGVR).Delete(ctx, vgsClassName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to delete groupSnapshotClass %q: %w", vgsClassName, err)
	}

	if err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotClassGVR, vgsClassName, framework.Poll, timeouts.SnapshotDelete); err != nil {
		return fmt.Errorf("failed waiting for groupSnapshotClass %q deletion: %w", vgsClassName, err)
	}

	framework.Logf("successfully deleted groupSnapshotClass %q", vgsClassName)
	return nil
}

// CleanupResource deletes the VolumeGroupSnapshotClass and VolumeGroupSnapshot objects using a dynamic client,
// and ignores not found errors.
func (r *VolumeGroupSnapshotResource) CleanupResource(ctx context.Context, timeouts *framework.TimeoutContext) error {
	var cleanupErrs []error

	if err := r.CleanupVGS(ctx, timeouts); err != nil {
		cleanupErrs = append(cleanupErrs, err)
	}

	if err := r.CleanupVGSClass(ctx, timeouts); err != nil {
		cleanupErrs = append(cleanupErrs, err)
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
