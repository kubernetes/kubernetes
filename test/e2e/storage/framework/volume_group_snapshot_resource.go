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
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// GetVolumeGroupSnapshot constructs a VolumeGroupSnapshot object with a label selector.
// vgsclassName is optional; if empty, the cluster's default VolumeGroupSnapshotClass
// will be used by the controller.
func GetVolumeGroupSnapshot(ns string, matchLabels map[string]interface{}, vgsclassName string) *unstructured.Unstructured {
	spec := map[string]interface{}{
		"source": map[string]interface{}{
			"selector": map[string]interface{}{
				"matchLabels": matchLabels,
			},
		},
	}
	if vgsclassName != "" {
		spec["volumeGroupSnapshotClassName"] = vgsclassName
	}
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeGroupSnapshot",
			"apiVersion": utils.VolumeGroupSnapshotAPIVersion,
			"metadata": map[string]interface{}{
				"generateName": "group-snapshot-",
				"namespace":    ns,
			},
			"spec": spec,
		},
	}
}

// VolumeGroupSnapshotResource represents a volumegroupsnapshot class, a volumegroupsnapshot and its bound contents for a specific test case
type VolumeGroupSnapshotResource struct {
	Config  *PerTestConfig
	Pattern TestPattern

	VGS        *unstructured.Unstructured
	VGSContent *unstructured.Unstructured
	VGSClass   *unstructured.Unstructured

	// SnapshotHandleToPVCName maps snapshot handle to source PVC name
	// This is populated during conversion to pre-provisioned snapshots
	// to preserve the PVC name information that is lost in pre-provisioned resources
	SnapshotHandleToPVCName map[string]string
}

// CreateVolumeGroupSnapshot creates a VolumeGroupSnapshotClass with given SnapshotDeletionPolicy and a VolumeGroupSnapshot
// from the VolumeGroupSnapshotClass using a dynamic client.
// Returns the unstructured VolumeGroupSnapshotClass and VolumeGroupSnapshot objects.
func CreateVolumeGroupSnapshot(ctx context.Context, sDriver VolumeGroupSnapshottableTestDriver, config *PerTestConfig, pattern TestPattern, groupName string, pvcNamespace string, timeouts *framework.TimeoutContext, parameters map[string]string) (*unstructured.Unstructured, *unstructured.Unstructured, *unstructured.Unstructured) {
	defer ginkgo.GinkgoRecover()
	var err error
	if pattern.SnapshotType != VolumeGroupSnapshot && pattern.SnapshotType != PreprovisionedCreatedVolumeGroupSnapshot {
		err = fmt.Errorf("SnapshotType must be set to either VolumeGroupSnapshot or PreprovisionedCreatedVolumeGroupSnapshot")
		framework.ExpectNoError(err, "SnapshotType is set to VolumeGroupSnapshot or PreprovisionedCreatedVolumeGroupSnapshot")
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
	volumeGroupSnapshot := GetVolumeGroupSnapshot(pvcNamespace, map[string]interface{}{
		"group": groupName,
	}, gsclass.GetName())

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

// patchVGSCDeletionPolicy patches the deletion policy of a single VolumeGroupSnapshotContent
func patchVGSCDeletionPolicy(ctx context.Context, dc dynamic.Interface, vgscName string, deletionPolicy string) error {
	patchData := fmt.Appendf(nil, `{"spec":{"deletionPolicy":"%s"}}`, deletionPolicy)
	_, err := dc.Resource(utils.VolumeGroupSnapshotContentGVR).Patch(ctx, vgscName, types.MergePatchType, patchData, metav1.PatchOptions{})
	if err != nil {
		return fmt.Errorf("failed to patch VGSC %q deletion policy to %s: %w", vgscName, deletionPolicy, err)
	}
	return nil
}

// patchVSCDeletionPolicy patches the deletion policy of a single VolumeSnapshotContent
func patchVSCDeletionPolicy(ctx context.Context, dc dynamic.Interface, vscName string, deletionPolicy string) error {
	patchData := fmt.Appendf(nil, `{"spec":{"deletionPolicy":"%s"}}`, deletionPolicy)
	_, err := dc.Resource(utils.SnapshotContentGVR).Patch(ctx, vscName, types.MergePatchType, patchData, metav1.PatchOptions{})
	if err != nil {
		return fmt.Errorf("failed to patch VSC %q deletion policy to %s: %w", vscName, deletionPolicy, err)
	}
	return nil
}

// patchVSCsDeletionPolicy patches deletion policy for all VolumeSnapshotContents owned by the given VGS
func (r *VolumeGroupSnapshotResource) patchVSCsDeletionPolicy(ctx context.Context, dc dynamic.Interface, vgsNamespace string, vgsUID types.UID, deletionPolicy string) error {
	// Get all VolumeSnapshotContent names owned by this VGS
	contentNamesSet, err := r.getVolumeSnapshotContentNames(ctx, dc, vgsNamespace, vgsUID)
	if err != nil {
		return fmt.Errorf("failed to get VolumeSnapshotContent names: %w", err)
	}

	// Patch all VolumeSnapshotContents deletion policy
	for _, contentName := range contentNamesSet.UnsortedList() {
		if err := patchVSCDeletionPolicy(ctx, dc, contentName, deletionPolicy); err != nil {
			return err
		}
	}

	return nil
}

// patchVGSAndVSCsDeletionPolicy patches the deletion policy of the VGSC and all its associated VSCs
func (r *VolumeGroupSnapshotResource) patchVGSAndVSCsDeletionPolicy(ctx context.Context, dc dynamic.Interface, vgsc *unstructured.Unstructured, vgsNamespace string, vgsUID types.UID, deletionPolicy string) (*unstructured.Unstructured, error) {
	// Patch VGSC deletion policy
	if err := patchVGSCDeletionPolicy(ctx, dc, vgsc.GetName(), deletionPolicy); err != nil {
		return nil, err
	}

	// Get the updated VGSC
	updatedVGSC, err := dc.Resource(utils.VolumeGroupSnapshotContentGVR).Get(ctx, vgsc.GetName(), metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get updated VGSC %q: %w", vgsc.GetName(), err)
	}

	// Patch all VolumeSnapshotContents deletion policy
	if err := r.patchVSCsDeletionPolicy(ctx, dc, vgsNamespace, vgsUID, deletionPolicy); err != nil {
		return nil, err
	}

	return updatedVGSC, nil
}

// getVolumeSnapshotContentNames returns the set of VolumeSnapshotContent names bound to VolumeSnapshots owned by the given VGS
func (r *VolumeGroupSnapshotResource) getVolumeSnapshotContentNames(ctx context.Context, dc dynamic.Interface, vgsNamespace string, vgsUID types.UID) (sets.Set[string], error) {
	vss, err := dc.Resource(utils.SnapshotGVR).Namespace(vgsNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	// Filter to VolumeSnapshots owned by this VGS
	ownedSnapshots := utils.FilterResourcesByOwner(vss.Items, "VolumeGroupSnapshot", vgsUID)

	// Extract VolumeSnapshotContent names from owned snapshots
	contentNamesSet := sets.New[string]()
	for _, vs := range ownedSnapshots {
		if status, ok := vs.Object["status"].(map[string]interface{}); ok {
			if cName, ok := status["boundVolumeSnapshotContentName"].(string); ok && cName != "" {
				contentNamesSet.Insert(cName)
			}
		}
	}
	return contentNamesSet, nil
}

// cleanupRetainedVolumeSnapshotContents forcibly deletes VolumeSnapshotContents with Retain policy
func (r *VolumeGroupSnapshotResource) cleanupRetainedVolumeSnapshotContents(ctx context.Context, dc dynamic.Interface, contentNames sets.Set[string], timeouts *framework.TimeoutContext) {
	// Update deletion policy and delete each VolumeSnapshotContent
	for _, contentName := range contentNames.UnsortedList() {
		// Patch deletion policy to Delete
		if err := patchVSCDeletionPolicy(ctx, dc, contentName, "Delete"); err != nil {
			framework.Failf("Failed to patch VolumeSnapshotContent %q deletion policy: %v", contentName, err)
			return
		}

		// Delete VolumeSnapshotContent
		err := dc.Resource(utils.SnapshotContentGVR).Delete(ctx, contentName, metav1.DeleteOptions{})
		if err != nil {
			framework.Failf("Failed to delete VolumeSnapshotContent %q: %v", contentName, err)
			return
		}
	}

	// Wait for all VolumeSnapshotContents to be deleted
	framework.Logf("Waiting for VolumeSnapshotContents to be deleted")
	for _, contentName := range contentNames.UnsortedList() {
		if err := utils.WaitForGVRDeletion(ctx, dc, utils.SnapshotContentGVR, contentName, framework.Poll, timeouts.SnapshotDelete); err != nil {
			framework.Failf("VolumeSnapshotContent %q was not fully deleted: %v", contentName, err)
			return
		}
	}
}

// cleanupRetainedVGSContent forcibly deletes VolumeGroupSnapshotContent with Retain policy
func (r *VolumeGroupSnapshotResource) cleanupRetainedVGSContent(ctx context.Context, dc dynamic.Interface, vgscName string, timeouts *framework.TimeoutContext) {
	framework.Logf("Deleting VGSContent %q", vgscName)

	// Patch deletion policy to Delete
	if err := patchVGSCDeletionPolicy(ctx, dc, vgscName, "Delete"); err != nil {
		framework.Failf("Failed to patch VGSContent %q deletion policy: %v", vgscName, err)
		return
	}

	// Delete VGSContent
	err := dc.Resource(utils.VolumeGroupSnapshotContentGVR).Delete(ctx, vgscName, metav1.DeleteOptions{})
	if err != nil {
		framework.Failf("Failed to delete VGSContent %q: %v", vgscName, err)
		return
	}

	// Wait for VGSContent to be deleted
	framework.Logf("Waiting for VGSContent %q to be deleted", vgscName)
	err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotContentGVR, vgscName, framework.Poll, timeouts.SnapshotDelete)
	if err != nil {
		framework.Failf("VGSContent %q was not fully deleted: %v", vgscName, err)
		return
	}
}

// verifyVolumeSnapshotContentsDeletion verifies that all VolumeSnapshotContents have been deleted
func (r *VolumeGroupSnapshotResource) verifyVolumeSnapshotContentsDeletion(ctx context.Context, dc dynamic.Interface, contentNames sets.Set[string], timeouts *framework.TimeoutContext) {
	ginkgo.By("Verifying all VolumeSnapshotContents have been deleted")
	var verifyErrs []error
	for _, contentName := range contentNames.UnsortedList() {
		err := utils.EnsureGVRDeletion(ctx, dc, utils.SnapshotContentGVR, contentName, framework.Poll, timeouts.SnapshotDelete, "")
		if err != nil {
			verifyErrs = append(verifyErrs, fmt.Errorf("VolumeSnapshotContent %q should be deleted: %w", contentName, err))
		}
	}
	if len(verifyErrs) > 0 {
		framework.ExpectNoError(utilerrors.NewAggregate(verifyErrs), "Failed to verify VolumeSnapshotContents deletion")
	}
}

// CleanupVGS deletes the VolumeGroupSnapshot and ensures the bound VolumeGroupSnapshotContent has Delete policy.
// It waits for the VolumeGroupSnapshot, its owned VolumeSnapshots, VolumeSnapshotContents, and the bound
// VolumeGroupSnapshotContent to be fully deleted to prevent resource leaks.
func (r *VolumeGroupSnapshotResource) CleanupVGS(ctx context.Context, timeouts *framework.TimeoutContext) error {
	if r.VGS == nil {
		return nil
	}

	dc := r.Config.Framework.DynamicClient
	vgsNamespace := r.VGS.GetNamespace()
	vgsName := r.VGS.GetName()
	vgsUID := r.VGS.GetUID()
	framework.Logf("deleting groupSnapshot %q/%q and ensuring content has Delete policy", vgsNamespace, vgsName)

	// Get the VGS to ensure it exists
	vgs, err := dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(vgsNamespace).Get(ctx, vgsName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get VGS %q: %w", vgsName, err)
	}
	r.VGS = vgs

	// Get all VolumeSnapshotContent names owned by this VGS
	contentNamesSet, err := r.getVolumeSnapshotContentNames(ctx, dc, vgsNamespace, vgsUID)
	if err != nil {
		return err
	}

	// Get the bound VolumeGroupSnapshotContent
	groupSnapshotStatus := r.VGS.Object["status"].(map[string]interface{})
	groupSnapshotContentName := groupSnapshotStatus["boundVolumeGroupSnapshotContentName"].(string)
	framework.Logf("received groupSnapshotStatus %v", groupSnapshotStatus)
	framework.Logf("groupSnapshotContentName %q", groupSnapshotContentName)

	boundVGSContent, err := dc.Resource(utils.VolumeGroupSnapshotContentGVR).Get(ctx, groupSnapshotContentName, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to get bound VGSContent %q: %w", groupSnapshotContentName, err)
	}

	// For Delete policy, ensure deletion policy is set to Delete to prevent leaks
	// For Retain policy, we'll update it after verifying retention behavior
	if boundVGSContent != nil && r.Pattern.SnapshotDeletionPolicy == DeleteSnapshot {
		spec := boundVGSContent.Object["spec"].(map[string]interface{})
		if spec["deletionPolicy"] != "Delete" {
			boundVGSContent, err = r.patchVGSAndVSCsDeletionPolicy(ctx, dc, boundVGSContent, vgsNamespace, vgsUID, "Delete")
			if err != nil {
				return fmt.Errorf("failed to update deletion policy: %w", err)
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

	// Wait for VolumeSnapshots owned by this VGS to be cascade-deleted
	// This happens regardless of deletion policy because VGS owns VolumeSnapshots via owner references
	if vgsUID != "" {
		if err := utils.WaitForOwnedResourcesDeleted(ctx, dc, utils.SnapshotGVR, vgsNamespace, vgsUID, framework.Poll, timeouts.SnapshotDelete); err != nil {
			return fmt.Errorf("failed waiting for owned snapshots deletion of VGS %q: %w", vgsName, err)
		}
	}

	// Verify deletion policy behavior for VolumeGroupSnapshotContent
	if boundVGSContent != nil {
		vgscName := boundVGSContent.GetName()

		switch r.Pattern.SnapshotDeletionPolicy {
		case DeleteSnapshot:
			ginkgo.By(fmt.Sprintf("Verifying VolumeGroupSnapshotContent %q has been deleted per Delete policy", vgscName))
			err = utils.EnsureGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotContentGVR, vgscName, framework.Poll, timeouts.SnapshotDelete, "")
			framework.ExpectNoError(err, "VolumeGroupSnapshotContent should be deleted with Delete policy")
			r.VGSContent = nil
		case RetainSnapshot:
			ginkgo.By(fmt.Sprintf("Verifying VolumeGroupSnapshotContent %q has been retained per Retain policy", vgscName))
			err = utils.EnsureNoGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotContentGVR, vgscName, framework.Poll, 20*framework.Poll, "")
			framework.ExpectNoError(err, "VolumeGroupSnapshotContent should be retained with Retain policy")

			// Force cleanup to prevent leaks
			// Cleanup order:
			// 1. VolumeSnapshotContents (update policy + delete + wait)
			// 2. VGSContent (update policy + delete + wait)
			framework.Logf("Retain policy verified, now forcing cleanup to prevent leaks")
			r.cleanupRetainedVolumeSnapshotContents(ctx, dc, contentNamesSet, timeouts)
			r.cleanupRetainedVGSContent(ctx, dc, vgscName, timeouts)
			r.VGSContent = nil
		}
	}

	// Verify all VolumeSnapshotContents have been deleted
	r.verifyVolumeSnapshotContentsDeletion(ctx, dc, contentNamesSet, timeouts)

	return nil
}

// CleanupVGSClass deletes the VolumeGroupSnapshotClass.
func (r *VolumeGroupSnapshotResource) CleanupVGSClass(ctx context.Context, timeouts *framework.TimeoutContext) error {
	if r.VGSClass == nil {
		framework.Logf("VGSClass is nil, skipping cleanup")
		return nil
	}

	dc := r.Config.Framework.DynamicClient
	vgsClassName := r.VGSClass.GetName()
	framework.Logf("deleting VolumeGroupSnapshotClass %q", vgsClassName)

	err := dc.Resource(utils.VolumeGroupSnapshotClassGVR).Delete(ctx, vgsClassName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to delete groupSnapshotClass %q: %w", vgsClassName, err)
	}

	if err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotClassGVR, vgsClassName, framework.Poll, timeouts.SnapshotDelete); err != nil {
		return fmt.Errorf("failed waiting for VolumeGroupSnapshotClass %q deletion: %w", vgsClassName, err)
	}

	framework.Logf("successfully deleted VolumeGroupSnapshotClass %q", vgsClassName)
	r.VGSClass = nil
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
	r := &VolumeGroupSnapshotResource{
		Config:     config,
		Pattern:    pattern,
		VGS:        snapshot,
		VGSClass:   vgsClass,
		VGSContent: vgsc,
	}

	dc := config.Framework.DynamicClient

	if pattern.SnapshotType == PreprovisionedCreatedVolumeGroupSnapshot {
		r.convertToPreprovisioned(ctx, dc, snapshot, vgsc, vgsClass, pvcNamespace, timeouts)
	}

	return r
}

// convertToPreprovisioned converts a dynamic VolumeGroupSnapshot to a pre-provisioned one.
// This is done by:
// 1. Recording the snapshot handles from the dynamic snapshot
// 2. Deleting the dynamic snapshot resources (with Retain policy to keep backend snapshots)
// 3. Creating new pre-provisioned resources that reference the same backend snapshots
func (r *VolumeGroupSnapshotResource) convertToPreprovisioned(ctx context.Context, dc dynamic.Interface, snapshot, vgsc, vgsClass *unstructured.Unstructured, pvcNamespace string, timeouts *framework.TimeoutContext) {
	var err error
	vgsUID := snapshot.GetUID()

	// Only update to Retain if the pattern specifies Delete policy
	// This preserves the backend snapshots when we delete the dynamic resources
	if r.Pattern.SnapshotDeletionPolicy == DeleteSnapshot {
		ginkgo.By("updating VGSC and VSC deletion policies to Retain")
		vgsc, err = r.patchVGSAndVSCsDeletionPolicy(ctx, dc, vgsc, pvcNamespace, vgsUID, "Retain")
		framework.ExpectNoError(err, "failed to update VGSC and VSC deletion policies to Retain")
	}

	ginkgo.By("recording properties of the pre-provisioned group snapshot")
	vgscStatus := vgsc.Object["status"].(map[string]interface{})
	groupSnapshotHandle := vgscStatus["volumeGroupSnapshotHandle"].(string)
	framework.Logf("Recording group snapshot content handle: %s", groupSnapshotHandle)

	// Extract individual volume snapshot handles from volumeSnapshotInfoList
	volumeSnapshotInfoList := vgscStatus["volumeSnapshotInfoList"].([]interface{})
	volumeSnapshotHandles := make([]string, 0, len(volumeSnapshotInfoList))
	for _, info := range volumeSnapshotInfoList {
		infoMap := info.(map[string]interface{})
		snapshotHandle := infoMap["snapshotHandle"].(string)
		volumeSnapshotHandles = append(volumeSnapshotHandles, snapshotHandle)
	}
	framework.Logf("Recording %d volume snapshot handles", len(volumeSnapshotHandles))

	csiDriverName := vgsClass.Object["driver"].(string)
	framework.Logf("Recording snapshot driver: %s", csiDriverName)

	// Get all VolumeSnapshotContent names owned by this VGS before deletion
	contentNamesSet, err := r.getVolumeSnapshotContentNames(ctx, dc, pvcNamespace, vgsUID)
	framework.ExpectNoError(err, "failed to get VolumeSnapshotContent names owned by VGS %s", vgsUID)

	// Capture the mapping of snapshot handle to source PVC name before deleting dynamic snapshots
	// This is needed because pre-provisioned snapshots don't have spec.source.persistentVolumeClaimName
	ginkgo.By("capturing source PVC names from dynamic VolumeSnapshots")
	vsList, err := dc.Resource(utils.SnapshotGVR).Namespace(pvcNamespace).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "failed to list VolumeSnapshots")

	// Filter to VolumeSnapshots owned by this VGS
	ownedSnapshots := utils.FilterResourcesByOwner(vsList.Items, "VolumeGroupSnapshot", vgsUID)

	// Map snapshotHandle to source PVC name and store in the resource struct
	r.SnapshotHandleToPVCName = make(map[string]string)
	for _, vs := range ownedSnapshots {
		// Get the source PVC name from the dynamic VolumeSnapshot
		// Since we waited for VGS to be ready, the VS should be bound and have source PVC name
		spec := vs.Object["spec"].(map[string]interface{})
		source := spec["source"].(map[string]interface{})
		sourcePVCName := source["persistentVolumeClaimName"].(string)

		// Get the VSC name from status (VS is bound since VGS is ready)
		status := vs.Object["status"].(map[string]interface{})
		vscName := status["boundVolumeSnapshotContentName"].(string)

		// Get the VSC to find the snapshot handle (should exist since VS is bound)
		vsc, err := dc.Resource(utils.SnapshotContentGVR).Get(ctx, vscName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get VolumeSnapshotContent %s for bound VolumeSnapshot %s", vscName, vs.GetName())

		vscStatus := vsc.Object["status"].(map[string]interface{})
		snapshotHandle := vscStatus["snapshotHandle"].(string)
		r.SnapshotHandleToPVCName[snapshotHandle] = sourcePVCName
		framework.Logf("Mapped snapshot handle %s to PVC %s", snapshotHandle, sourcePVCName)
	}

	// If the deletion policy is retain on vgsc:
	// when vgs is deleted vgsc will not be deleted
	// when the vgsc is manually deleted then the underlying group snapshot resource will not be deleted.
	// We exploit this to create a group snapshot resource from which we can create a preprovisioned snapshot
	ginkgo.By("deleting the group snapshot")
	err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(snapshot.GetNamespace()).Delete(ctx, snapshot.GetName(), metav1.DeleteOptions{})
	framework.ExpectNoError(err, "failed to delete VolumeGroupSnapshot %s/%s", snapshot.GetNamespace(), snapshot.GetName())

	ginkgo.By("checking the VolumeGroupSnapshot has been deleted")
	err = utils.WaitForNamespacedGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotGVR, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, timeouts.SnapshotDelete)
	framework.ExpectNoError(err, "VolumeGroupSnapshot %s/%s was not deleted within timeout", snapshot.GetNamespace(), snapshot.GetName())

	// Wait for VolumeSnapshots owned by this VGS to be cascade-deleted
	ginkgo.By("waiting for owned VolumeSnapshots to be deleted")
	err = utils.WaitForOwnedResourcesDeleted(ctx, dc, utils.SnapshotGVR, pvcNamespace, vgsUID, framework.Poll, timeouts.SnapshotDelete)
	framework.ExpectNoError(err, "VolumeSnapshots owned by VGS %s were not deleted within timeout", vgsUID)

	// Delete the VolumeSnapshotContents (with Retain policy, so physical snapshots remain)
	ginkgo.By("deleting VolumeSnapshotContents")
	for _, contentName := range contentNamesSet.UnsortedList() {
		err := dc.Resource(utils.SnapshotContentGVR).Delete(ctx, contentName, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "failed to delete VolumeSnapshotContent %s", contentName)
		}
	}

	// Wait for VolumeSnapshotContents to be deleted
	for _, contentName := range contentNamesSet.UnsortedList() {
		err := utils.WaitForGVRDeletion(ctx, dc, utils.SnapshotContentGVR, contentName, framework.Poll, timeouts.SnapshotDelete)
		framework.ExpectNoError(err, "VolumeSnapshotContent %s was not deleted within timeout", contentName)
	}

	// Delete the VolumeGroupSnapshotContent (with Retain policy, so physical group snapshot remains)
	err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Delete(ctx, vgsc.GetName(), metav1.DeleteOptions{})
	framework.ExpectNoError(err, "failed to delete VolumeGroupSnapshotContent %s", vgsc.GetName())

	ginkgo.By("checking the VolumeGroupSnapshotContent has been deleted")
	err = utils.WaitForGVRDeletion(ctx, dc, utils.VolumeGroupSnapshotContentGVR, vgsc.GetName(), framework.Poll, timeouts.SnapshotDelete)
	framework.ExpectNoError(err, "VolumeGroupSnapshotContent %s was not deleted within timeout", vgsc.GetName())

	// Create new pre-provisioned VGSContent with the extracted group snapshot handle
	ginkgo.By("creating a new pre-provisioned VolumeGroupSnapshotContent with the group snapshot handle")
	vgsUUID := uuid.NewUUID()

	vgsName := getPreProvisionedVolumeGroupSnapshotName(vgsUUID)
	vgscName := getPreProvisionedVolumeGroupSnapshotContentName(vgsUUID)

	r.VGSContent = getPreProvisionedVolumeGroupSnapshotContent(vgscName, vgsName, pvcNamespace, groupSnapshotHandle, volumeSnapshotHandles, r.Pattern.SnapshotDeletionPolicy.String(), csiDriverName)
	r.VGSContent, err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Create(ctx, r.VGSContent, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pre-provisioned VolumeGroupSnapshotContent %s", vgscName)

	ginkgo.By("creating a pre-provisioned VGS with that VGSContent")
	r.VGS = getPreProvisionedVolumeGroupSnapshot(vgsName, pvcNamespace, vgscName)
	r.VGS, err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(r.VGS.GetNamespace()).Create(ctx, r.VGS, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pre-provisioned VolumeGroupSnapshot %s/%s", pvcNamespace, vgsName)

	err = utils.WaitForVolumeGroupSnapshotReady(ctx, dc, r.VGS.GetNamespace(), r.VGS.GetName(), framework.Poll, timeouts.SnapshotCreate*10)
	framework.ExpectNoError(err, "pre-provisioned VolumeGroupSnapshot %s/%s did not become ready within timeout", r.VGS.GetNamespace(), r.VGS.GetName())

	// Get the new VGS UID for owner references
	r.VGS, err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(r.VGS.GetNamespace()).Get(ctx, r.VGS.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get VolumeGroupSnapshot %s/%s for UID", r.VGS.GetNamespace(), r.VGS.GetName())
	newVgsUID := r.VGS.GetUID()
	vgsAPIVersion := r.VGS.GetAPIVersion()
	vgsKind := r.VGS.GetKind()

	// Create new pre-provisioned VolumeSnapshotContents and VolumeSnapshots
	// The VolumeSnapshots need to be owned by the VGS so the controller can discover them
	ginkgo.By("creating pre-provisioned VolumeSnapshotContents and VolumeSnapshots owned by the VGS")

	for i, snapshotHandle := range volumeSnapshotHandles {
		// Generate unique names for the pre-provisioned VolumeSnapshotContent and VolumeSnapshot
		// Using VGS UUID + index to make names predictable and associated with parent VGS
		snapName := fmt.Sprintf("pre-provisioned-vs-%s-%d", string(vgsUUID), i)
		snapContentName := fmt.Sprintf("pre-provisioned-vsc-%s-%d", string(vgsUUID), i)

		// Create pre-provisioned VolumeSnapshotContent first
		// Note: volumeSnapshotClassName is not required for pre-provisioned VolumeSnapshotContent
		vsc := getPreProvisionedVolumeSnapshotContent(snapContentName, snapName, pvcNamespace, snapshotHandle, r.Pattern.SnapshotDeletionPolicy.String(), csiDriverName)
		_, err := dc.Resource(utils.SnapshotContentGVR).Create(ctx, vsc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pre-provisioned VolumeSnapshotContent %s", snapContentName)

		// Get the source PVC name mapping for this snapshot handle
		sourcePVCName := r.SnapshotHandleToPVCName[snapshotHandle]
		gomega.Expect(sourcePVCName).NotTo(gomega.BeEmpty(), "no source PVC name mapping found for snapshot handle %s", snapshotHandle)
		framework.Logf("Creating pre-provisioned snapshot %s for PVC %s (handle: %s)", snapName, sourcePVCName, snapshotHandle)

		// Create pre-provisioned VolumeSnapshot
		preProvisionedVS := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"kind":       "VolumeSnapshot",
				"apiVersion": utils.SnapshotAPIVersion,
				"metadata": map[string]interface{}{
					"name":      snapName,
					"namespace": pvcNamespace,
					"ownerReferences": []interface{}{
						map[string]interface{}{
							"apiVersion": vgsAPIVersion,
							"kind":       vgsKind,
							"name":       r.VGS.GetName(),
							"uid":        newVgsUID,
						},
					},
				},
				"spec": map[string]interface{}{
					"source": map[string]interface{}{
						"volumeSnapshotContentName": snapContentName,
					},
				},
			},
		}

		_, err = dc.Resource(utils.SnapshotGVR).Namespace(pvcNamespace).Create(ctx, preProvisionedVS, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pre-provisioned VolumeSnapshot %s/%s", pvcNamespace, snapName)

		// Wait for the VolumeSnapshot to be ready
		err = utils.WaitForSnapshotReady(ctx, dc, pvcNamespace, snapName, framework.Poll, timeouts.SnapshotCreate)
		framework.ExpectNoError(err, "pre-provisioned VolumeSnapshot %s/%s did not become ready within timeout", pvcNamespace, snapName)

		framework.Logf("Created pre-provisioned VolumeSnapshotContent %s and VolumeSnapshot %s (%d/%d)", snapContentName, snapName, i+1, len(volumeSnapshotHandles))
	}

	ginkgo.By("getting the group snapshot and group snapshot content")
	r.VGS, err = dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(r.VGS.GetNamespace()).Get(ctx, r.VGS.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get final VolumeGroupSnapshot %s/%s", r.VGS.GetNamespace(), r.VGS.GetName())

	r.VGSContent, err = dc.Resource(utils.VolumeGroupSnapshotContentGVR).Get(ctx, r.VGSContent.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get final VolumeGroupSnapshotContent %s", r.VGSContent.GetName())
}

func getPreProvisionedVolumeGroupSnapshot(vgsName, ns, vgscName string) *unstructured.Unstructured {
	snapshot := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeGroupSnapshot",
			"apiVersion": utils.VolumeGroupSnapshotAPIVersion,
			"metadata": map[string]interface{}{
				"name":      vgsName,
				"namespace": ns,
			},
			"spec": map[string]interface{}{
				"source": map[string]interface{}{
					"volumeGroupSnapshotContentName": vgscName,
				},
			},
		},
	}

	return snapshot
}

func getPreProvisionedVolumeGroupSnapshotName(uuid types.UID) string {
	return fmt.Sprintf("pre-provisioned-vgs-%s", string(uuid))
}

func getPreProvisionedVolumeGroupSnapshotContentName(uuid types.UID) string {
	return fmt.Sprintf("pre-provisioned-vgsc-%s", string(uuid))
}

func getPreProvisionedVolumeGroupSnapshotContent(vgscName, vgsName, vgsNamespace, groupSnapshotHandle string, volumeSnapshotHandles []string, deletionPolicy, csiDriverName string) *unstructured.Unstructured {
	snapshotContent := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeGroupSnapshotContent",
			"apiVersion": utils.VolumeGroupSnapshotAPIVersion,
			"metadata": map[string]interface{}{
				"name": vgscName,
			},
			"spec": map[string]interface{}{
				"source": map[string]interface{}{
					"groupSnapshotHandles": map[string]interface{}{
						"volumeGroupSnapshotHandle": groupSnapshotHandle,
						"volumeSnapshotHandles":     volumeSnapshotHandles,
					},
				},
				"volumeGroupSnapshotRef": map[string]interface{}{
					"name":      vgsName,
					"namespace": vgsNamespace,
				},
				"driver":         csiDriverName,
				"deletionPolicy": deletionPolicy,
			},
		},
	}

	return snapshotContent
}

func getPreProvisionedVolumeSnapshotContent(snapcontentName, snapshotName, snapshotNamespace, snapshotHandle, deletionPolicy, csiDriverName string) *unstructured.Unstructured {
	snapshotContent := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshotContent",
			"apiVersion": utils.SnapshotAPIVersion,
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
