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
	"errors"
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-helpers/storage/ephemeral"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/kubernetes/pkg/volume/util"
)

func createInTreeVolumeSpec(logger klog.Logger, podVolume *v1.Volume, pod *v1.Pod, vpm *volume.VolumePluginMgr, pvcLister corelisters.PersistentVolumeClaimLister, pvLister corelisters.PersistentVolumeLister, csiMigratedPluginManager csimigration.PluginManager, csiTranslator csimigration.InTreeToCSITranslator) (*volume.Spec, string, error) {
	claimName := ""
	readOnly := false
	if pvcSource := podVolume.VolumeSource.PersistentVolumeClaim; pvcSource != nil {
		claimName = pvcSource.ClaimName
		readOnly = pvcSource.ReadOnly
	}
	isEphemeral := podVolume.VolumeSource.Ephemeral != nil
	if isEphemeral {
		claimName = ephemeral.VolumeClaimName(pod, podVolume)
	}
	if claimName == "" {
		// In-line volume
		return volume.NewSpecFromVolume(podVolume), "", nil
	}
	// The volume is a PVC, dereference the PVC + PV
	logger.V(10).Info("Found PVC", "PVC", klog.KRef(pod.Namespace, claimName))

	// If podVolume is a PVC, fetch the real PV behind the claim
	pvc, err := getPVCFromCache(pod.Namespace, claimName, pvcLister)
	if err != nil {
		return nil, claimName, fmt.Errorf(
			"error processing PVC %q/%q: %w",
			pod.Namespace,
			claimName,
			err)
	}
	if isEphemeral {
		if err := ephemeral.VolumeIsForPod(pod, pvc); err != nil {
			return nil, claimName, err
		}
	}

	pvName, pvcUID := pvc.Spec.VolumeName, pvc.UID
	logger.V(10).Info("Found bound PV for PVC", "PVC", klog.KRef(pod.Namespace, claimName), "pvcUID", pvcUID, "PV", klog.KRef("", pvName))

	// Fetch actual PV object
	volumeSpec, err := getPVSpecFromCache(
		pvName, readOnly, pvcUID, pvLister)
	if err != nil {
		return nil, claimName, fmt.Errorf(
			"error processing PVC %q/%q: %w",
			pod.Namespace,
			claimName,
			err)
	}

	logger.V(10).Info("Extracted volumeSpec from bound PV and PVC", "PVC", klog.KRef(pod.Namespace, claimName), "pvcUID", pvcUID, "PV", klog.KRef("", pvName), "volumeSpecName", volumeSpec.Name())
	return volumeSpec, claimName, nil
}

func CreateVolumeSpec(logger klog.Logger, podVolume v1.Volume, pod *v1.Pod, vpm *volume.VolumePluginMgr, pvcLister corelisters.PersistentVolumeClaimLister, pvLister corelisters.PersistentVolumeLister, csiMigratedPluginManager csimigration.PluginManager, csiTranslator csimigration.InTreeToCSITranslator) (*volume.Spec, error) {
	volumeSpec, claimName, err := createInTreeVolumeSpec(logger, &podVolume, pod, vpm, pvcLister, pvLister, csiMigratedPluginManager, csiTranslator)
	if err != nil {
		return nil, err
	}
	volumeSpec, err = translateInTreeSpecToCSIIfNeeded(logger, volumeSpec, vpm, csiMigratedPluginManager, csiTranslator, pod.Namespace)
	if err != nil {
		return nil, fmt.Errorf(
			"error performing CSI migration checks and translation for PVC %q/%q: %w",
			pod.Namespace,
			claimName,
			err)
	}
	return volumeSpec, nil
}

// CreateVolumeSpec creates and returns a mutatable volume.Spec object for the
// specified volume. It dereference any PVC to get PV objects, if needed.
// A volume.Spec that refers to an in-tree plugin spec is translated to refer
// to a migrated CSI plugin spec if all conditions for CSI migration on a node
// for the in-tree plugin is satisfied.
func CreateVolumeSpecWithNodeMigration(logger klog.Logger, podVolume v1.Volume, pod *v1.Pod, nodeName types.NodeName, vpm *volume.VolumePluginMgr, pvcLister corelisters.PersistentVolumeClaimLister, pvLister corelisters.PersistentVolumeLister, csiMigratedPluginManager csimigration.PluginManager, csiTranslator csimigration.InTreeToCSITranslator) (*volume.Spec, error) {
	volumeSpec, claimName, err := createInTreeVolumeSpec(logger, &podVolume, pod, vpm, pvcLister, pvLister, csiMigratedPluginManager, csiTranslator)
	if err != nil {
		return nil, err
	}
	volumeSpec, err = translateInTreeSpecToCSIOnNodeIfNeeded(logger, volumeSpec, nodeName, vpm, csiMigratedPluginManager, csiTranslator, pod.Namespace)
	if err != nil {
		return nil, fmt.Errorf(
			"error performing CSI migration checks and translation for PVC %q/%q: %w",
			pod.Namespace,
			claimName,
			err)
	}
	return volumeSpec, nil
}

// getPVCFromCache fetches the PVC object with the given namespace and
// name from the shared internal PVC store.
// This method returns an error if a PVC object does not exist in the cache
// with the given namespace/name.
// This method returns an error if the PVC object's phase is not "Bound".
func getPVCFromCache(namespace string, name string, pvcLister corelisters.PersistentVolumeClaimLister) (*v1.PersistentVolumeClaim, error) {
	pvc, err := pvcLister.PersistentVolumeClaims(namespace).Get(name)
	if err != nil {
		return nil, fmt.Errorf("failed to find PVC %s/%s in PVCInformer cache: %v", namespace, name, err)
	}

	if pvc.Status.Phase != v1.ClaimBound || pvc.Spec.VolumeName == "" {
		return nil, fmt.Errorf(
			"PVC %s/%s has non-bound phase (%q) or empty pvc.Spec.VolumeName (%q)",
			namespace,
			name,
			pvc.Status.Phase,
			pvc.Spec.VolumeName)
	}

	return pvc, nil
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

	if util.IsPodTerminated(pod, pod.Status) {
		return false
	}
	return defaultAction
}

// ProcessPodVolumes processes the volumes in the given pod and adds them to the
// desired state of the world if addVolumes is true, otherwise it removes them.
func ProcessPodVolumes(logger klog.Logger, pod *v1.Pod, addVolumes bool, desiredStateOfWorld cache.DesiredStateOfWorld, volumePluginMgr *volume.VolumePluginMgr, pvcLister corelisters.PersistentVolumeClaimLister, pvLister corelisters.PersistentVolumeLister, csiMigratedPluginManager csimigration.PluginManager, csiTranslator csimigration.InTreeToCSITranslator) {
	if pod == nil {
		return
	}
	if len(pod.Spec.Volumes) <= 0 {
		logger.V(10).Info("Skipping processing of pod, it has no volumes", "pod", klog.KObj(pod))
		return
	}

	nodeName := types.NodeName(pod.Spec.NodeName)
	if nodeName == "" {
		logger.V(10).Info("Skipping processing of pod, it is not scheduled to a node", "pod", klog.KObj(pod))
		return
	} else if !desiredStateOfWorld.NodeExists(nodeName) {
		// If the node the pod is scheduled to does not exist in the desired
		// state of the world data structure, that indicates the node is not
		// yet managed by the controller. Therefore, ignore the pod.
		logger.V(4).Info("Skipping processing of pod, it is scheduled to node which is not managed by the controller", "node", klog.KRef("", string(nodeName)), "pod", klog.KObj(pod))
		return
	}

	// Process volume spec for each volume defined in pod
	for _, podVolume := range pod.Spec.Volumes {
		volumeSpec, err := CreateVolumeSpecWithNodeMigration(logger, podVolume, pod, nodeName, volumePluginMgr, pvcLister, pvLister, csiMigratedPluginManager, csiTranslator)
		if err != nil {
			logger.V(10).Info("Error processing volume for pod", "pod", klog.KObj(pod), "volumeName", podVolume.Name, "err", err)
			continue
		}

		attachableVolumePlugin, err :=
			volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
		if err != nil || attachableVolumePlugin == nil {
			logger.V(10).Info("Skipping volume for pod, it does not implement attacher interface", "pod", klog.KObj(pod), "volumeName", podVolume.Name, "err", err)
			continue
		}

		uniquePodName := util.GetUniquePodName(pod)
		if addVolumes {
			// Add volume to desired state of world
			_, err := desiredStateOfWorld.AddPod(
				uniquePodName, pod, volumeSpec, nodeName)
			if err != nil {
				logger.V(10).Info("Failed to add volume for pod to desiredStateOfWorld", "pod", klog.KObj(pod), "volumeName", podVolume.Name, "err", err)
			}

		} else {
			// Remove volume from desired state of world
			uniqueVolumeName, err := util.GetUniqueVolumeNameFromSpec(
				attachableVolumePlugin, volumeSpec)
			if err != nil {
				logger.V(10).Info("Failed to delete volume for pod from desiredStateOfWorld. GetUniqueVolumeNameFromSpec failed", "pod", klog.KObj(pod), "volumeName", podVolume.Name, "err", err)
				continue
			}
			desiredStateOfWorld.DeletePod(
				uniquePodName, uniqueVolumeName, nodeName)
		}
	}
}

func translateInTreeSpecToCSIOnNodeIfNeeded(logger klog.Logger, spec *volume.Spec, nodeName types.NodeName, vpm *volume.VolumePluginMgr, csiMigratedPluginManager csimigration.PluginManager, csiTranslator csimigration.InTreeToCSITranslator, podNamespace string) (*volume.Spec, error) {
	translatedSpec := spec
	migratable, err := csiMigratedPluginManager.IsMigratable(spec)
	if err != nil {
		return nil, err
	}
	if !migratable {
		// Jump out of translation fast so we don't check the node if the spec itself is not migratable
		return spec, nil
	}
	migrationSupportedOnNode, err := isCSIMigrationSupportedOnNode(nodeName, spec, vpm, csiMigratedPluginManager)
	if err != nil {
		return nil, err
	}
	if migratable && migrationSupportedOnNode {
		translatedSpec, err = csimigration.TranslateInTreeSpecToCSI(logger, spec, podNamespace, csiTranslator)
		if err != nil {
			return nil, err
		}
	}
	return translatedSpec, nil
}

func translateInTreeSpecToCSIIfNeeded(logger klog.Logger, spec *volume.Spec, vpm *volume.VolumePluginMgr, csiMigratedPluginManager csimigration.PluginManager, csiTranslator csimigration.InTreeToCSITranslator, podNamespace string) (*volume.Spec, error) {
	migratable, err := csiMigratedPluginManager.IsMigratable(spec)
	if err != nil {
		return nil, err
	}
	if !migratable {
		// Jump out of translation fast so we don't check the node if the spec itself is not migratable
		return spec, nil
	}
	translatedSpec, err := csimigration.TranslateInTreeSpecToCSI(logger, spec, podNamespace, csiTranslator)
	if err != nil {
		return nil, err
	}
	return translatedSpec, nil
}

func isCSIMigrationSupportedOnNode(nodeName types.NodeName, spec *volume.Spec, vpm *volume.VolumePluginMgr, csiMigratedPluginManager csimigration.PluginManager) (bool, error) {
	pluginName, err := csiMigratedPluginManager.GetInTreePluginNameFromSpec(spec.PersistentVolume, spec.Volume)
	if err != nil {
		return false, err
	}

	if len(pluginName) == 0 {
		// Could not find a plugin name from translation directory, assume not translated
		return false, nil
	}

	if csiMigratedPluginManager.IsMigrationCompleteForPlugin(pluginName) {
		// All nodes are expected to have migrated CSI plugin installed and
		// configured when CSI Migration Complete flag is enabled for a plugin.
		// CSI migration is supported even if there is version skew between
		// managers and node.
		return true, nil
	}

	if len(nodeName) == 0 {
		return false, errors.New("nodeName is empty")
	}

	kubeClient := vpm.Host.GetKubeClient()
	if kubeClient == nil {
		// Don't handle the controller/kubelet version skew check and fallback
		// to just checking the feature gates. This can happen if
		// we are in a standalone (headless) Kubelet
		return true, nil
	}

	adcHost, ok := vpm.Host.(volume.AttachDetachVolumeHost)
	if !ok {
		// Don't handle the controller/kubelet version skew check and fallback
		// to just checking the feature gates. This can happen if
		// "enableControllerAttachDetach" is set to true on kubelet
		return true, nil
	}

	if adcHost.CSINodeLister() == nil {
		return false, errors.New("could not find CSINodeLister in attachDetachController")
	}

	csiNode, err := adcHost.CSINodeLister().Get(string(nodeName))
	if err != nil {
		return false, err
	}

	ann := csiNode.GetAnnotations()
	if ann == nil {
		return false, nil
	}

	mpa := ann[v1.MigratedPluginsAnnotationKey]
	tok := strings.Split(mpa, ",")
	mpaSet := sets.NewString(tok...)

	isMigratedOnNode := mpaSet.Has(pluginName)

	return isMigratedOnNode, nil
}
