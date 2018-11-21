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

	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// CSIMaxVolumeLimitChecker defines predicate needed for counting CSI volumes
type CSIMaxVolumeLimitChecker struct {
	pvInfo  PersistentVolumeInfo
	pvcInfo PersistentVolumeClaimInfo
}

// NewCSIMaxVolumeLimitPredicate returns a predicate for counting CSI volumes
func NewCSIMaxVolumeLimitPredicate(
	pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo) algorithm.FitPredicate {
	c := &CSIMaxVolumeLimitChecker{
		pvInfo:  pvInfo,
		pvcInfo: pvcInfo,
	}
	return c.attachableLimitPredicate
}

func (c *CSIMaxVolumeLimitChecker) attachableLimitPredicate(
	pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {

	// if feature gate is disable we return
	if !utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		return true, nil, nil
	}
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return true, nil, nil
	}

	nodeVolumeLimits := nodeInfo.VolumeLimits()

	// if node does not have volume limits this predicate should exit
	if len(nodeVolumeLimits) == 0 {
		return true, nil, nil
	}

	// a map of unique volume name/csi volume handle and volume limit key
	newVolumes := make(map[string]string)
	if err := c.filterAttachableVolumes(pod.Spec.Volumes, pod.Namespace, newVolumes); err != nil {
		return false, nil, err
	}

	if len(newVolumes) == 0 {
		return true, nil, nil
	}

	// a map of unique volume name/csi volume handle and volume limit key
	attachedVolumes := make(map[string]string)
	for _, existingPod := range nodeInfo.Pods() {
		if err := c.filterAttachableVolumes(existingPod.Spec.Volumes, existingPod.Namespace, attachedVolumes); err != nil {
			return false, nil, err
		}
	}

	newVolumeCount := map[string]int{}
	attachedVolumeCount := map[string]int{}

	for volumeName, volumeLimitKey := range attachedVolumes {
		if _, ok := newVolumes[volumeName]; ok {
			delete(newVolumes, volumeName)
		}
		attachedVolumeCount[volumeLimitKey]++
	}

	for _, volumeLimitKey := range newVolumes {
		newVolumeCount[volumeLimitKey]++
	}

	for volumeLimitKey, count := range newVolumeCount {
		maxVolumeLimit, ok := nodeVolumeLimits[v1.ResourceName(volumeLimitKey)]
		if ok {
			currentVolumeCount := attachedVolumeCount[volumeLimitKey]
			if currentVolumeCount+count > int(maxVolumeLimit) {
				return false, []algorithm.PredicateFailureReason{ErrMaxVolumeCountExceeded}, nil
			}
		}
	}

	return true, nil, nil
}

func (c *CSIMaxVolumeLimitChecker) filterAttachableVolumes(
	volumes []v1.Volume, namespace string, result map[string]string) error {

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
			klog.V(4).Infof("Unable to look up PVC info for %s/%s", namespace, pvcName)
			continue
		}

		pvName := pvc.Spec.VolumeName
		// TODO - the actual handling of unbound PVCs will be fixed by late binding design.
		if pvName == "" {
			klog.V(4).Infof("Persistent volume had no name for claim %s/%s", namespace, pvcName)
			continue
		}
		pv, err := c.pvInfo.GetPersistentVolumeInfo(pvName)

		if err != nil {
			klog.V(4).Infof("Unable to look up PV info for PVC %s/%s and PV %s", namespace, pvcName, pvName)
			continue
		}

		csiSource := pv.Spec.PersistentVolumeSource.CSI
		if csiSource == nil {
			klog.V(4).Infof("Not considering non-CSI volume %s/%s", namespace, pvcName)
			continue
		}
		driverName := csiSource.Driver
		volumeLimitKey := volumeutil.GetCSIAttachLimitKey(driverName)
		result[csiSource.VolumeHandle] = volumeLimitKey

	}
	return nil
}
