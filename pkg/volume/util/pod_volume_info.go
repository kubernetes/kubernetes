package util

import (
	"context"
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-helpers/storage/ephemeral"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// PodVolumeInfos is a helper struct that contains information about all volumes in a pod.
type PodVolumeInfos struct {
	pod                      *v1.Pod
	csiMigratedPluginManager csimigration.PluginManager
	intreeToCSITranslator    csimigration.InTreeToCSITranslator
	volumePluginMgr          *volume.VolumePluginMgr
	kubeClient               clientset.Interface

	// Info for each volume, indexed by volume.name.
	volumeInfos map[string]*VolumeSpecInfo
}

// VolumeSpecInfo contains information about a volume in a pod, with a dereferenced PVC and PV (if necessary).
type VolumeSpecInfo struct {
	// Name of the volume in the Pod (i.e. pod.volumes[*].name)
	OuterVolumeName string

	// Unique name of the pod.
	PodName volumetypes.UniquePodName

	// Whether the volume is mounted in the pod.
	Mounted bool

	// Whether the volume is mapped in the pod.
	Mapped bool

	// List of all SELinuxOpts that are used by all containers that mount the volume.
	// (When SELinuxMountReadWriteOncePod feature is enabled).
	SELinuxContexts []*v1.SELinuxOptions

	// Volume spec, with dereferenced PVC and PV.
	Spec *volume.Spec

	// Volume plugin responsible for the volume.
	VolumePlugin volume.VolumePlugin

	// Corresponding PVC.
	PVC *v1.PersistentVolumeClaim

	// Value of GID annotation of corresponding PV.
	VolumeGIDValue string
}

// NewPodVolumeInfos creates a new PodVolumeInfos object for the given pod.
// It does not dereference any PV/PVCs yet, as it is expensive and not always needed.
func NewPodVolumeInfos(
	pod *v1.Pod,
	csiMigratedPluginManager csimigration.PluginManager,
	intreeToCSITranslator csimigration.InTreeToCSITranslator,
	volumePluginMgr *volume.VolumePluginMgr,
	kubeClient clientset.Interface) *PodVolumeInfos {

	volumeInfos := make(map[string]*VolumeSpecInfo)
	volumeNames := make([]string, 0, len(pod.Spec.Volumes))
	for i := range pod.Spec.Volumes {
		name := pod.Spec.Volumes[i].Name
		volumeNames = append(volumeNames, name)
		volumeInfos[name] = &VolumeSpecInfo{
			OuterVolumeName: name,
			PodName:         GetUniquePodName(pod),
		}
	}
	// Collect all that is possible to collect with a single sweep through the pod.
	podutil.VisitContainers(&pod.Spec, podutil.AllFeatureEnabledContainers(), func(container *v1.Container, containerType podutil.ContainerType) bool {
		var seLinuxOptions *v1.SELinuxOptions
		if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
			effectiveContainerSecurity := securitycontext.DetermineEffectiveSecurityContext(pod, container)
			if effectiveContainerSecurity != nil {
				// No DeepCopy, SELinuxOptions is already a copy of Pod's or container's SELinuxOptions
				seLinuxOptions = effectiveContainerSecurity.SELinuxOptions
			}
		}

		if container.VolumeMounts != nil {
			for _, mount := range container.VolumeMounts {
				volInfo, found := volumeInfos[mount.Name]
				if !found {
					klog.Infof("Volume %s is mounted in container %s, but not found in pod volumes", mount.Name, container.Name)
					continue
				}
				volInfo.Mounted = true
				volInfo.SELinuxContexts = append(volInfo.SELinuxContexts, seLinuxOptions.DeepCopy())
			}
		}
		if container.VolumeDevices != nil {
			for _, device := range container.VolumeDevices {
				volInfo, found := volumeInfos[device.Name]
				if !found {
					klog.Infof("Volume %s is mapped in container %s, but not found in pod volumes", device.Name, container.Name)
					continue
				}
				volInfo.Mapped = true
			}
		}
		return true
	})

	ret := &PodVolumeInfos{
		pod:                      pod,
		csiMigratedPluginManager: csiMigratedPluginManager,
		intreeToCSITranslator:    intreeToCSITranslator,
		volumePluginMgr:          volumePluginMgr,
		kubeClient:               kubeClient,

		volumeInfos: volumeInfos,
	}
	return ret
}

// IsVolumeMounted returns true if the volume is mounted in the pod (i.e. there is a container that has it in container.volumeMounts).
func (p *PodVolumeInfos) IsVolumeMounted(volumeName string) bool {
	volumeInfo, found := p.volumeInfos[volumeName]
	if !found {
		return false
	}
	return volumeInfo.Mounted
}

// IsVolumeMapped returns true if the volume is mapped in the pod (i.e. there is a container that has it in container.volumeDevices).
func (p *PodVolumeInfos) IsVolumeMapped(volumeName string) bool {
	volumeInfo, found := p.volumeInfos[volumeName]
	if !found {
		return false
	}
	return volumeInfo.Mapped
}

// GetVolumeInfo returns the VolumeSpecInfo for the given volume in the pod.
// It dereferences the PVC and PV if necessary, and caches the result.
// The first call to this method for a given volume may be expensive and include API calls.
func (p *PodVolumeInfos) GetVolumeInfo(podVolume *v1.Volume) (*VolumeSpecInfo, error) {
	volumeInfo, found := p.volumeInfos[podVolume.Name]
	if !found {
		return nil, fmt.Errorf("podVolume %s not found in pod", podVolume.Name)
	}
	if volumeInfo.Spec != nil {
		// The volume was already populated & cached.
		return volumeInfo, nil
	}

	// Populate the volume info from PVC, PV and whatnot
	err := p.populateFullVolumeInfo(volumeInfo, podVolume)
	return volumeInfo, err
}

// populateFullVolumeInfo populates the VolumeSpecInfo with the full volume information, including dereferencing PVC and PV.
func (p *PodVolumeInfos) populateFullVolumeInfo(info *VolumeSpecInfo, podVolume *v1.Volume) error {
	pvcSource := podVolume.VolumeSource.PersistentVolumeClaim
	isEphemeral := pvcSource == nil && podVolume.VolumeSource.Ephemeral != nil
	if isEphemeral {
		// Generic ephemeral inline volumes are handled the
		// same way as a PVC reference. The only additional
		// constraint (checked below) is that the PVC must be
		// owned by the pod.
		pvcSource = &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: ephemeral.VolumeClaimName(p.pod, podVolume),
		}
	}
	if pvcSource != nil {
		klog.V(5).InfoS("Found PVC", "PVC", klog.KRef(p.pod.Namespace, pvcSource.ClaimName))
		// If podVolume is a PVC, fetch the real PV behind the claim
		pvc, err := p.getPVCExtractPV(pvcSource.ClaimName)
		if err != nil {
			return fmt.Errorf("error processing PVC %s/%s: %v",
				p.pod.Namespace,
				pvcSource.ClaimName,
				err)
		}
		if isEphemeral {
			if err := ephemeral.VolumeIsForPod(p.pod, pvc); err != nil {
				return err
			}
		}
		pvName, pvcUID := pvc.Spec.VolumeName, pvc.UID
		klog.V(5).InfoS("Found bound PV for PVC", "PVC", klog.KRef(p.pod.Namespace, pvcSource.ClaimName), "PVCUID", pvcUID, "PVName", pvName)
		// Fetch actual PV object
		volumeSpec, volumeGidValue, err := p.getPVSpec(pvName, pvcSource.ReadOnly, pvcUID)
		if err != nil {
			return fmt.Errorf("error processing PVC %s/%s: %v",
				p.pod.Namespace,
				pvcSource.ClaimName,
				err)
		}
		klog.V(5).InfoS("Extracted volumeSpec from bound PV and PVC", "PVC", klog.KRef(p.pod.Namespace, pvcSource.ClaimName), "PVCUID", pvcUID, "PVName", pvName, "volumeSpecName", volumeSpec.Name())
		migratable, err := p.csiMigratedPluginManager.IsMigratable(volumeSpec)
		if err != nil {
			return err
		}
		if migratable {
			volumeSpec, err = csimigration.TranslateInTreeSpecToCSI(volumeSpec, p.pod.Namespace, p.intreeToCSITranslator)
			if err != nil {
				return err
			}
		}

		volumeMode, err := GetVolumeMode(volumeSpec)
		if err != nil {
			return err
		}
		// Error if a container has volumeMounts but the volumeMode of PVC isn't Filesystem.
		if info.Mounted && volumeMode != v1.PersistentVolumeFilesystem {
			return fmt.Errorf("volume %s has volumeMode %s, but is specified in volumeMounts",
				podVolume.Name,
				volumeMode)
		}
		// Error if a container has volumeDevices but the volumeMode of PVC isn't Block
		if info.Mapped && volumeMode != v1.PersistentVolumeBlock {
			return fmt.Errorf("volume %s has volumeMode %s, but is specified in volumeDevices",
				podVolume.Name,
				volumeMode)
		}

		volumePlugin, err := p.volumePluginMgr.FindPluginBySpec(volumeSpec)
		if err != nil || volumePlugin == nil {
			return fmt.Errorf(
				"failed to get Plugin from volumeSpec for volume %q err=%v",
				volumeSpec.Name(),
				err)
		}

		info.VolumeGIDValue = volumeGidValue
		info.Spec = volumeSpec
		info.PVC = pvc
		info.VolumePlugin = volumePlugin

		return nil
	}

	// Do not return the original volume object, since the source could mutate it
	clonedPodVolume := podVolume.DeepCopy()

	spec := volume.NewSpecFromVolume(clonedPodVolume)
	migratable, err := p.csiMigratedPluginManager.IsMigratable(spec)
	if err != nil {
		return err
	}
	if migratable {
		spec, err = csimigration.TranslateInTreeSpecToCSI(spec, p.pod.Namespace, p.intreeToCSITranslator)
		if err != nil {
			return err
		}
	}

	volumePlugin, err := p.volumePluginMgr.FindPluginBySpec(spec)
	if err != nil || volumePlugin == nil {
		return fmt.Errorf(
			"failed to get Plugin from volumeSpec for volume %q err=%v",
			spec.Name(),
			err)
	}

	info.Spec = spec
	info.VolumePlugin = volumePlugin
	return nil
}

// getPVCExtractPV fetches the PVC object with the given namespace and name from
// the API server, checks whether PVC is being deleted, extracts the name of the PV
// it is pointing to and returns it.
// An error is returned if the PVC object's phase is not "Bound".
func (p *PodVolumeInfos) getPVCExtractPV(claimName string) (*v1.PersistentVolumeClaim, error) {
	pvc, err := p.kubeClient.CoreV1().PersistentVolumeClaims(p.pod.Namespace).Get(context.TODO(), claimName, metav1.GetOptions{})
	if err != nil || pvc == nil {
		return nil, fmt.Errorf("failed to fetch PVC from API server: %v", err)
	}

	// Pods that uses a PVC that is being deleted must not be started.
	//
	// In case an old kubelet is running without this check or some kubelets
	// have this feature disabled, the worst that can happen is that such
	// pod is scheduled. This was the default behavior in 1.8 and earlier
	// and users should not be that surprised.
	// It should happen only in very rare case when scheduler schedules
	// a pod and user deletes a PVC that's used by it at the same time.
	if pvc.ObjectMeta.DeletionTimestamp != nil {
		return nil, errors.New("PVC is being deleted")
	}

	if pvc.Status.Phase != v1.ClaimBound {
		return nil, errors.New("PVC is not bound")
	}
	if pvc.Spec.VolumeName == "" {
		return nil, errors.New("PVC has empty pvc.Spec.VolumeName")
	}

	return pvc, nil
}

// getPVSpec fetches the PV object with the given name from the API server
// and returns a volume.Spec representing it.
// An error is returned if the call to fetch the PV object fails.
func (p *PodVolumeInfos) getPVSpec(
	name string,
	pvcReadOnly bool,
	expectedClaimUID types.UID) (*volume.Spec, string, error) {
	pv, err := p.kubeClient.CoreV1().PersistentVolumes().Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil || pv == nil {
		return nil, "", fmt.Errorf(
			"failed to fetch PV %s from API server: %v", name, err)
	}

	if pv.Spec.ClaimRef == nil {
		return nil, "", fmt.Errorf(
			"found PV object %s but it has a nil pv.Spec.ClaimRef indicating it is not yet bound to the claim",
			name)
	}

	if pv.Spec.ClaimRef.UID != expectedClaimUID {
		return nil, "", fmt.Errorf(
			"found PV object %s but its pv.Spec.ClaimRef.UID %s does not point to claim.UID %s",
			name,
			pv.Spec.ClaimRef.UID,
			expectedClaimUID)
	}

	volumeGidValue := getPVVolumeGidAnnotationValue(pv)
	return volume.NewSpecFromPersistentVolume(pv, pvcReadOnly), volumeGidValue, nil
}

func getPVVolumeGidAnnotationValue(pv *v1.PersistentVolume) string {
	if volumeGid, ok := pv.Annotations[VolumeGidAnnotationKey]; ok {
		return volumeGid
	}

	return ""
}
