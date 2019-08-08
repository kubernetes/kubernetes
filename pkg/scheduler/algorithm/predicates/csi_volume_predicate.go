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

package predicates

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apimachinery/pkg/util/rand"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	csilib "k8s.io/csi-translation-lib"
	"k8s.io/klog"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// CSIMaxVolumeLimitChecker defines predicate needed for counting CSI volumes
type CSIMaxVolumeLimitChecker struct {
	csiNodeInfo CSINodeInfo
	pvInfo      PersistentVolumeInfo
	pvcInfo     PersistentVolumeClaimInfo
	scInfo      StorageClassInfo

	randomVolumeIDPrefix string
}

// NewCSIMaxVolumeLimitPredicate returns a predicate for counting CSI volumes
func NewCSIMaxVolumeLimitPredicate(
	csiNodeInfo CSINodeInfo, pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo, scInfo StorageClassInfo) FitPredicate {
	c := &CSIMaxVolumeLimitChecker{
		csiNodeInfo:          csiNodeInfo,
		pvInfo:               pvInfo,
		pvcInfo:              pvcInfo,
		scInfo:               scInfo,
		randomVolumeIDPrefix: rand.String(32),
	}
	return c.attachableLimitPredicate
}

func (c *CSIMaxVolumeLimitChecker) getVolumeLimits(nodeInfo *schedulernodeinfo.NodeInfo, csiNode *storagev1beta1.CSINode) map[v1.ResourceName]int64 {
	// TODO: stop getting values from Node object in v1.18
	nodeVolumeLimits := nodeInfo.VolumeLimits()
	if csiNode != nil {
		for i := range csiNode.Spec.Drivers {
			d := csiNode.Spec.Drivers[i]
			if d.Allocatable != nil && d.Allocatable.Count != nil {
				// TODO: drop GetCSIAttachLimitKey once we don't get values from Node object (v1.18)
				k := v1.ResourceName(volumeutil.GetCSIAttachLimitKey(d.Name))
				nodeVolumeLimits[k] = int64(*d.Allocatable.Count)
			}
		}
	}
	return nodeVolumeLimits
}

func (c *CSIMaxVolumeLimitChecker) attachableLimitPredicate(
	pod *v1.Pod, meta PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	// If the new pod doesn't have any volume attached to it, the predicate will always be true
	if len(pod.Spec.Volumes) == 0 {
		return true, nil, nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		return true, nil, nil
	}

	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	// If CSINode doesn't exist, the predicate may read the limits from Node object
	csiNode, err := c.csiNodeInfo.GetCSINodeInfo(node.Name)
	if err != nil {
		// TODO: return the error once CSINode is created by default (2 releases)
		klog.V(5).Infof("Could not get a CSINode object for the node: %v", err)
	}

	newVolumes := make(map[string]string)
	if err := c.filterAttachableVolumes(csiNode, pod.Spec.Volumes, pod.Namespace, newVolumes); err != nil {
		return false, nil, err
	}

	// If the pod doesn't have any new CSI volumes, the predicate will always be true
	if len(newVolumes) == 0 {
		return true, nil, nil
	}

	// If the node doesn't have volume limits, the predicate will always be true
	nodeVolumeLimits := c.getVolumeLimits(nodeInfo, csiNode)
	if len(nodeVolumeLimits) == 0 {
		return true, nil, nil
	}

	attachedVolumes := make(map[string]string)
	for _, existingPod := range nodeInfo.Pods() {
		if err := c.filterAttachableVolumes(csiNode, existingPod.Spec.Volumes, existingPod.Namespace, attachedVolumes); err != nil {
			return false, nil, err
		}
	}

	attachedVolumeCount := map[string]int{}
	for volumeUniqueName, volumeLimitKey := range attachedVolumes {
		if _, ok := newVolumes[volumeUniqueName]; ok {
			// Don't count single volume used in multiple pods more than once
			delete(newVolumes, volumeUniqueName)
		}
		attachedVolumeCount[volumeLimitKey]++
	}

	newVolumeCount := map[string]int{}
	for _, volumeLimitKey := range newVolumes {
		newVolumeCount[volumeLimitKey]++
	}

	for volumeLimitKey, count := range newVolumeCount {
		maxVolumeLimit, ok := nodeVolumeLimits[v1.ResourceName(volumeLimitKey)]
		if ok {
			currentVolumeCount := attachedVolumeCount[volumeLimitKey]
			if currentVolumeCount+count > int(maxVolumeLimit) {
				return false, []PredicateFailureReason{ErrMaxVolumeCountExceeded}, nil
			}
		}
	}

	return true, nil, nil
}

func (c *CSIMaxVolumeLimitChecker) filterAttachableVolumes(
	csiNode *storagev1beta1.CSINode, volumes []v1.Volume, namespace string, result map[string]string) error {
	for _, vol := range volumes {
		// CSI volumes can only be used as persistent volumes
		if vol.PersistentVolumeClaim == nil {
			continue
		}
		pvcName := vol.PersistentVolumeClaim.ClaimName

		if pvcName == "" {
			return fmt.Errorf("PersistentVolumeClaim had no name")
		}

		pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(namespace, pvcName)

		if err != nil {
			klog.V(5).Infof("Unable to look up PVC info for %s/%s", namespace, pvcName)
			continue
		}

		driverName, volumeHandle := c.getCSIDriverInfo(csiNode, pvc)
		if driverName == "" || volumeHandle == "" {
			klog.V(5).Infof("Could not find a CSI driver name or volume handle, not counting volume")
			continue
		}

		volumeUniqueName := fmt.Sprintf("%s/%s", driverName, volumeHandle)
		volumeLimitKey := volumeutil.GetCSIAttachLimitKey(driverName)
		result[volumeUniqueName] = volumeLimitKey
	}
	return nil
}

// getCSIDriverInfo returns the CSI driver name and volume ID of a given PVC.
// If the PVC is from a migrated in-tree plugin, this function will return
// the information of the CSI driver that the plugin has been migrated to.
func (c *CSIMaxVolumeLimitChecker) getCSIDriverInfo(csiNode *storagev1beta1.CSINode, pvc *v1.PersistentVolumeClaim) (string, string) {
	pvName := pvc.Spec.VolumeName
	namespace := pvc.Namespace
	pvcName := pvc.Name

	if pvName == "" {
		klog.V(5).Infof("Persistent volume had no name for claim %s/%s", namespace, pvcName)
		return c.getCSIDriverInfoFromSC(csiNode, pvc)
	}

	pv, err := c.pvInfo.GetPersistentVolumeInfo(pvName)
	if err != nil {
		klog.V(5).Infof("Unable to look up PV info for PVC %s/%s and PV %s", namespace, pvcName, pvName)
		// If we can't fetch PV associated with PVC, may be it got deleted
		// or PVC was prebound to a PVC that hasn't been created yet.
		// fallback to using StorageClass for volume counting
		return c.getCSIDriverInfoFromSC(csiNode, pvc)
	}

	csiSource := pv.Spec.PersistentVolumeSource.CSI
	if csiSource == nil {
		// We make a fast path for non-CSI volumes that aren't migratable
		if !csilib.IsPVMigratable(pv) {
			return "", ""
		}

		pluginName, err := csilib.GetInTreePluginNameFromSpec(pv, nil)
		if err != nil {
			klog.V(5).Infof("Unable to look up plugin name from PV spec: %v", err)
			return "", ""
		}

		if !isCSIMigrationOn(csiNode, pluginName) {
			klog.V(5).Infof("CSI Migration of plugin %s is not enabled", pluginName)
			return "", ""
		}

		csiPV, err := csilib.TranslateInTreePVToCSI(pv)
		if err != nil {
			klog.V(5).Infof("Unable to translate in-tree volume to CSI: %v", err)
			return "", ""
		}

		if csiPV.Spec.PersistentVolumeSource.CSI == nil {
			klog.V(5).Infof("Unable to get a valid volume source for translated PV %s", pvName)
			return "", ""
		}

		csiSource = csiPV.Spec.PersistentVolumeSource.CSI
	}

	return csiSource.Driver, csiSource.VolumeHandle
}

// getCSIDriverInfoFromSC returns the CSI driver name and a random volume ID of a given PVC's StorageClass.
func (c *CSIMaxVolumeLimitChecker) getCSIDriverInfoFromSC(csiNode *storagev1beta1.CSINode, pvc *v1.PersistentVolumeClaim) (string, string) {
	namespace := pvc.Namespace
	pvcName := pvc.Name
	scName := v1helper.GetPersistentVolumeClaimClass(pvc)

	// If StorageClass is not set or not found, then PVC must be using immediate binding mode
	// and hence it must be bound before scheduling. So it is safe to not count it.
	if scName == "" {
		klog.V(5).Infof("PVC %s/%s has no StorageClass", namespace, pvcName)
		return "", ""
	}

	storageClass, err := c.scInfo.GetStorageClassInfo(scName)
	if err != nil {
		klog.V(5).Infof("Could not get StorageClass for PVC %s/%s: %v", namespace, pvcName, err)
		return "", ""
	}

	// We use random prefix to avoid conflict with volume IDs. If PVC is bound during the execution of the
	// predicate and there is another pod on the same node that uses same volume, then we will overcount
	// the volume and consider both volumes as different.
	volumeHandle := fmt.Sprintf("%s-%s/%s", c.randomVolumeIDPrefix, namespace, pvcName)

	provisioner := storageClass.Provisioner
	if csilib.IsMigratableIntreePluginByName(provisioner) {
		if !isCSIMigrationOn(csiNode, provisioner) {
			klog.V(5).Infof("CSI Migration of plugin %s is not enabled", provisioner)
			return "", ""
		}

		driverName, err := csilib.GetCSINameFromInTreeName(provisioner)
		if err != nil {
			klog.V(5).Infof("Unable to look up driver name from plugin name: %v", err)
			return "", ""
		}
		return driverName, volumeHandle
	}

	return provisioner, volumeHandle
}
