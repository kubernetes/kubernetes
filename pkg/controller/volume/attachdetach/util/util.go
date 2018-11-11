/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// CreateVolumeSpec creates and returns a mutatable volume.Spec object for the
// specified volume. It dereference any PVC to get PV objects, if needed.
func CreateVolumeSpec(podVolume v1.Volume, podNamespace string, pvcLister corelisters.PersistentVolumeClaimLister, pvLister corelisters.PersistentVolumeLister) (*volume.Spec, error) {
	if pvcSource := podVolume.VolumeSource.PersistentVolumeClaim; pvcSource != nil {
		klog.V(10).Infof(
			"Found PVC, ClaimName: %q/%q",
			podNamespace,
			pvcSource.ClaimName)

		// If podVolume is a PVC, fetch the real PV behind the claim
		pvName, pvcUID, err := getPVCFromCacheExtractPV(
			podNamespace, pvcSource.ClaimName, pvcLister)
		if err != nil {
			return nil, fmt.Errorf(
				"error processing PVC %q/%q: %v",
				podNamespace,
				pvcSource.ClaimName,
				err)
		}

		klog.V(10).Infof(
			"Found bound PV for PVC (ClaimName %q/%q pvcUID %v): pvName=%q",
			podNamespace,
			pvcSource.ClaimName,
			pvcUID,
			pvName)

		// Fetch actual PV object
		volumeSpec, err := getPVSpecFromCache(
			pvName, pvcSource.ReadOnly, pvcUID, pvLister)
		if err != nil {
			return nil, fmt.Errorf(
				"error processing PVC %q/%q: %v",
				podNamespace,
				pvcSource.ClaimName,
				err)
		}

		klog.V(10).Infof(
			"Extracted volumeSpec (%v) from bound PV (pvName %q) and PVC (ClaimName %q/%q pvcUID %v)",
			volumeSpec.Name(),
			pvName,
			podNamespace,
			pvcSource.ClaimName,
			pvcUID)

		return volumeSpec, nil
	}

	// Do not return the original volume object, since it's from the shared
	// informer it may be mutated by another consumer.
	clonedPodVolume := podVolume.DeepCopy()

	return volume.NewSpecFromVolume(clonedPodVolume), nil
}

// getPVCFromCacheExtractPV fetches the PVC object with the given namespace and
// name from the shared internal PVC store extracts the name of the PV it is
// pointing to and returns it.
// This method returns an error if a PVC object does not exist in the cache
// with the given namespace/name.
// This method returns an error if the PVC object's phase is not "Bound".
func getPVCFromCacheExtractPV(namespace string, name string, pvcLister corelisters.PersistentVolumeClaimLister) (string, types.UID, error) {
	pvc, err := pvcLister.PersistentVolumeClaims(namespace).Get(name)
	if err != nil {
		return "", "", fmt.Errorf("failed to find PVC %s/%s in PVCInformer cache: %v", namespace, name, err)
	}

	if pvc.Status.Phase != v1.ClaimBound || pvc.Spec.VolumeName == "" {
		return "", "", fmt.Errorf(
			"PVC %s/%s has non-bound phase (%q) or empty pvc.Spec.VolumeName (%q)",
			namespace,
			name,
			pvc.Status.Phase,
			pvc.Spec.VolumeName)
	}

	return pvc.Spec.VolumeName, pvc.UID, nil
}

// getPVSpecFromCache fetches the PV object with the given name from the shared
// internal PV store and returns a volume.Spec representing it.
// This method returns an error if a PV object does not exist in the cache with
// the given name.
// This method deep copies the PV object so the caller may use the returned
// volume.Spec object without worrying about it mutating unexpectedly.
func getPVSpecFromCache(name string, pvcReadOnly bool, expectedClaimUID types.UID, pvLister corelisters.PersistentVolumeLister) (*volume.Spec, error) {
	pv, err := pvLister.Get(name)
	if err != nil {
		return nil, fmt.Errorf("failed to find PV %q in PVInformer cache: %v", name, err)
	}

	if pv.Spec.ClaimRef == nil {
		return nil, fmt.Errorf(
			"found PV object %q but it has a nil pv.Spec.ClaimRef indicating it is not yet bound to the claim",
			name)
	}

	if pv.Spec.ClaimRef.UID != expectedClaimUID {
		return nil, fmt.Errorf(
			"found PV object %q but its pv.Spec.ClaimRef.UID (%q) does not point to claim.UID (%q)",
			name,
			pv.Spec.ClaimRef.UID,
			expectedClaimUID)
	}

	// Do not return the object from the informer, since the store is shared it
	// may be mutated by another consumer.
	clonedPV := pv.DeepCopy()

	return volume.NewSpecFromPersistentVolume(clonedPV, pvcReadOnly), nil
}

// DetermineVolumeAction returns true if volume and pod needs to be added to dswp
// and it returns false if volume and pod needs to be removed from dswp
func DetermineVolumeAction(pod *v1.Pod, desiredStateOfWorld cache.DesiredStateOfWorld, defaultAction bool) bool {
	if pod == nil || len(pod.Spec.Volumes) <= 0 {
		return defaultAction
	}
	nodeName := types.NodeName(pod.Spec.NodeName)
	keepTerminatedPodVolume := desiredStateOfWorld.GetKeepTerminatedPodVolumesForNode(nodeName)

	if util.IsPodTerminated(pod, pod.Status) {
		// if pod is terminate we let kubelet policy dictate if volume
		// should be detached or not
		return keepTerminatedPodVolume
	}
	return defaultAction
}

// ProcessPodVolumes processes the volumes in the given pod and adds them to the
// desired state of the world if addVolumes is true, otherwise it removes them.
func ProcessPodVolumes(pod *v1.Pod, addVolumes bool, desiredStateOfWorld cache.DesiredStateOfWorld, volumePluginMgr *volume.VolumePluginMgr, pvcLister corelisters.PersistentVolumeClaimLister, pvLister corelisters.PersistentVolumeLister) {
	if pod == nil {
		return
	}

	if len(pod.Spec.Volumes) <= 0 {
		klog.V(10).Infof("Skipping processing of pod %q/%q: it has no volumes.",
			pod.Namespace,
			pod.Name)
		return
	}

	nodeName := types.NodeName(pod.Spec.NodeName)
	if nodeName == "" {
		klog.V(10).Infof(
			"Skipping processing of pod %q/%q: it is not scheduled to a node.",
			pod.Namespace,
			pod.Name)
		return
	} else if !desiredStateOfWorld.NodeExists(nodeName) {
		// If the node the pod is scheduled to does not exist in the desired
		// state of the world data structure, that indicates the node is not
		// yet managed by the controller. Therefore, ignore the pod.
		klog.V(4).Infof(
			"Skipping processing of pod %q/%q: it is scheduled to node %q which is not managed by the controller.",
			pod.Namespace,
			pod.Name,
			nodeName)
		return
	}

	// Process volume spec for each volume defined in pod
	for _, podVolume := range pod.Spec.Volumes {
		volumeSpec, err := CreateVolumeSpec(podVolume, pod.Namespace, pvcLister, pvLister)
		if err != nil {
			klog.V(10).Infof(
				"Error processing volume %q for pod %q/%q: %v",
				podVolume.Name,
				pod.Namespace,
				pod.Name,
				err)
			continue
		}

		attachableVolumePlugin, err :=
			volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
		if err != nil || attachableVolumePlugin == nil {
			klog.V(10).Infof(
				"Skipping volume %q for pod %q/%q: it does not implement attacher interface. err=%v",
				podVolume.Name,
				pod.Namespace,
				pod.Name,
				err)
			continue
		}

		uniquePodName := util.GetUniquePodName(pod)
		if addVolumes {
			// Add volume to desired state of world
			_, err := desiredStateOfWorld.AddPod(
				uniquePodName, pod, volumeSpec, nodeName)
			if err != nil {
				klog.V(10).Infof(
					"Failed to add volume %q for pod %q/%q to desiredStateOfWorld. %v",
					podVolume.Name,
					pod.Namespace,
					pod.Name,
					err)
			}

		} else {
			// Remove volume from desired state of world
			uniqueVolumeName, err := util.GetUniqueVolumeNameFromSpec(
				attachableVolumePlugin, volumeSpec)
			if err != nil {
				klog.V(10).Infof(
					"Failed to delete volume %q for pod %q/%q from desiredStateOfWorld. GetUniqueVolumeNameFromSpec failed with %v",
					podVolume.Name,
					pod.Namespace,
					pod.Name,
					err)
				continue
			}
			desiredStateOfWorld.DeletePod(
				uniquePodName, uniqueVolumeName, nodeName)
		}
	}
	return
}
