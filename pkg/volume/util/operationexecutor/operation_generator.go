/*
Copyright 2016 The Kubernetes Authors.

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

package operationexecutor

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/record"
	kevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

var _ OperationGenerator = &operationGenerator{}

type operationGenerator struct {
	// Used to fetch objects from the API server like Node in the
	// VerifyControllerAttachedVolume operation.
	kubeClient clientset.Interface

	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr

	// recorder is used to record events in the API server
	recorder record.EventRecorder

	// checkNodeCapabilitiesBeforeMount, if set, enables the CanMount check,
	// which verifies that the components (binaries, etc.) required to mount
	// the volume are available on the underlying node before attempting mount.
	checkNodeCapabilitiesBeforeMount bool
}

// NewOperationGenerator is returns instance of operationGenerator
func NewOperationGenerator(kubeClient clientset.Interface,
	volumePluginMgr *volume.VolumePluginMgr,
	recorder record.EventRecorder,
	checkNodeCapabilitiesBeforeMount bool) OperationGenerator {

	return &operationGenerator{
		kubeClient:      kubeClient,
		volumePluginMgr: volumePluginMgr,
		recorder:        recorder,
		checkNodeCapabilitiesBeforeMount: checkNodeCapabilitiesBeforeMount,
	}
}

// OperationGenerator interface that extracts out the functions from operation_executor to make it dependency injectable
type OperationGenerator interface {
	// Generates the MountVolume function needed to perform the mount of a volume plugin
	GenerateMountVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater) (func() error, error)

	// Generates the UnmountVolume function needed to perform the unmount of a volume plugin
	GenerateUnmountVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) (func() error, error)

	// Generates the AttachVolume function needed to perform attach of a volume plugin
	GenerateAttachVolumeFunc(volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error)

	// Generates the DetachVolume function needed to perform the detach of a volume plugin
	GenerateDetachVolumeFunc(volumeToDetach AttachedVolume, verifySafeToDetach bool, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error)

	// Generates the VolumesAreAttached function needed to verify if volume plugins are attached
	GenerateVolumesAreAttachedFunc(attachedVolumes []AttachedVolume, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error)

	// Generates the UnMountDevice function needed to perform the unmount of a device
	GenerateUnmountDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter mount.Interface) (func() error, error)

	// Generates the function needed to check if the attach_detach controller has attached the volume plugin
	GenerateVerifyControllerAttachedVolumeFunc(volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error)
}

func (og *operationGenerator) GenerateVolumesAreAttachedFunc(
	attachedVolumes []AttachedVolume,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {

	// volumesPerPlugin maps from a volume plugin to a list of volume specs which belong
	// to this type of plugin
	volumesPerPlugin := make(map[string][]*volume.Spec)
	// volumeSpecMap maps from a volume spec to its unique volumeName which will be used
	// when calling MarkVolumeAsDetached
	volumeSpecMap := make(map[*volume.Spec]v1.UniqueVolumeName)
	// Iterate each volume spec and put them into a map index by the pluginName
	for _, volumeAttached := range attachedVolumes {
		volumePlugin, err :=
			og.volumePluginMgr.FindPluginBySpec(volumeAttached.VolumeSpec)
		if err != nil || volumePlugin == nil {
			glog.Errorf(
				"VolumesAreAttached.FindPluginBySpec failed for volume %q (spec.Name: %q) on node %q with error: %v",
				volumeAttached.VolumeName,
				volumeAttached.VolumeSpec.Name(),
				volumeAttached.NodeName,
				err)
		}
		volumeSpecList, pluginExists := volumesPerPlugin[volumePlugin.GetPluginName()]
		if !pluginExists {
			volumeSpecList = []*volume.Spec{}
		}
		volumeSpecList = append(volumeSpecList, volumeAttached.VolumeSpec)
		volumesPerPlugin[volumePlugin.GetPluginName()] = volumeSpecList
		volumeSpecMap[volumeAttached.VolumeSpec] = volumeAttached.VolumeName
	}

	return func() error {

		// For each volume plugin, pass the list of volume specs to VolumesAreAttached to check
		// whether the volumes are still attached.
		for pluginName, volumesSpecs := range volumesPerPlugin {
			attachableVolumePlugin, err :=
				og.volumePluginMgr.FindAttachablePluginByName(pluginName)
			if err != nil || attachableVolumePlugin == nil {
				glog.Errorf(
					"VolumeAreAttached.FindAttachablePluginBySpec failed for plugin %q with: %v",
					pluginName,
					err)
				continue
			}

			volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()
			if newAttacherErr != nil {
				glog.Errorf(
					"VolumesAreAttached failed for getting plugin %q with: %v",
					pluginName,
					newAttacherErr)
				continue
			}

			attached, areAttachedErr := volumeAttacher.VolumesAreAttached(volumesSpecs, nodeName)
			if areAttachedErr != nil {
				glog.Errorf(
					"VolumesAreAttached failed for checking on node %q with: %v",
					nodeName,
					areAttachedErr)
				continue
			}

			for spec, check := range attached {
				if !check {
					actualStateOfWorld.MarkVolumeAsDetached(volumeSpecMap[spec], nodeName)
					glog.V(1).Infof("VerifyVolumesAreAttached determined volume %q (spec.Name: %q) is no longer attached to node %q, therefore it was marked as detached.",
						volumeSpecMap[spec], spec.Name(), nodeName)
				}
			}
		}
		return nil
	}, nil
}

func (og *operationGenerator) GenerateAttachVolumeFunc(
	volumeToAttach VolumeToAttach,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err :=
		og.volumePluginMgr.FindAttachablePluginBySpec(volumeToAttach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"AttachVolume.FindAttachablePluginBySpec failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToAttach.VolumeName,
			volumeToAttach.VolumeSpec.Name(),
			volumeToAttach.NodeName,
			err)
	}

	volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()
	if newAttacherErr != nil {
		return nil, fmt.Errorf(
			"AttachVolume.NewAttacher failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToAttach.VolumeName,
			volumeToAttach.VolumeSpec.Name(),
			volumeToAttach.NodeName,
			newAttacherErr)
	}

	return func() error {
		// Execute attach
		devicePath, attachErr := volumeAttacher.Attach(
			volumeToAttach.VolumeSpec, volumeToAttach.NodeName)

		if attachErr != nil {
			// On failure, return error. Caller will log and retry.
			err := fmt.Errorf(
				"Failed to attach volume %q on node %q with: %v",
				volumeToAttach.VolumeSpec.Name(),
				volumeToAttach.NodeName,
				attachErr)
			for _, pod := range volumeToAttach.ScheduledPods {
				og.recorder.Eventf(pod, v1.EventTypeWarning, kevents.FailedMountVolume, err.Error())
			}
			return err
		}

		glog.Infof(
			"AttachVolume.Attach succeeded for volume %q (spec.Name: %q) from node %q.",
			volumeToAttach.VolumeName,
			volumeToAttach.VolumeSpec.Name(),
			volumeToAttach.NodeName)

		// Update actual state of world
		addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
			v1.UniqueVolumeName(""), volumeToAttach.VolumeSpec, volumeToAttach.NodeName, devicePath)
		if addVolumeNodeErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"AttachVolume.MarkVolumeAsAttached failed for volume %q (spec.Name: %q) from node %q with: %v",
				volumeToAttach.VolumeName,
				volumeToAttach.VolumeSpec.Name(),
				volumeToAttach.NodeName,
				addVolumeNodeErr)
		}

		return nil
	}, nil
}

func (og *operationGenerator) GenerateDetachVolumeFunc(
	volumeToDetach AttachedVolume,
	verifySafeToDetach bool,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err :=
		og.volumePluginMgr.FindAttachablePluginBySpec(volumeToDetach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"DetachVolume.FindAttachablePluginBySpec failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			err)
	}

	volumeName, err :=
		attachableVolumePlugin.GetVolumeName(volumeToDetach.VolumeSpec)
	if err != nil {
		return nil, fmt.Errorf(
			"DetachVolume.GetVolumeName failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			err)
	}

	volumeDetacher, err := attachableVolumePlugin.NewDetacher()
	if err != nil {
		return nil, fmt.Errorf(
			"DetachVolume.NewDetacher failed for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			err)
	}

	return func() error {
		var err error
		if verifySafeToDetach {
			err = og.verifyVolumeIsSafeToDetach(volumeToDetach)
		}
		if err == nil {
			err = volumeDetacher.Detach(volumeName, volumeToDetach.NodeName)
		}
		if err != nil {
			// On failure, add volume back to ReportAsAttached list
			actualStateOfWorld.AddVolumeToReportAsAttached(
				volumeToDetach.VolumeName, volumeToDetach.NodeName)
			return fmt.Errorf(
				"DetachVolume.Detach failed for volume %q (spec.Name: %q) from node %q with: %v",
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name(),
				volumeToDetach.NodeName,
				err)
		}

		glog.Infof(
			"DetachVolume.Detach succeeded for volume %q (spec.Name: %q) from node %q.",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName)

		// Update actual state of world
		actualStateOfWorld.MarkVolumeAsDetached(
			volumeToDetach.VolumeName, volumeToDetach.NodeName)

		return nil
	}, nil
}

func (og *operationGenerator) GerifyVolumeIsSafeToDetach(
	volumeToDetach AttachedVolume) error {
	// Fetch current node object
	node, fetchErr := og.kubeClient.Core().Nodes().Get(string(volumeToDetach.NodeName), metav1.GetOptions{})
	if fetchErr != nil {
		if errors.IsNotFound(fetchErr) {
			glog.Warningf("Node %q not found on API server. DetachVolume will skip safe to detach check.",
				volumeToDetach.NodeName,
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name())
			return nil
		}

		// On failure, return error. Caller will log and retry.
		return fmt.Errorf(
			"DetachVolume failed fetching node from API server for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			fetchErr)
	}

	if node == nil {
		// On failure, return error. Caller will log and retry.
		return fmt.Errorf(
			"DetachVolume failed fetching node from API server for volume %q (spec.Name: %q) from node %q. Error: node object retrieved from API server is nil",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName)
	}

	for _, inUseVolume := range node.Status.VolumesInUse {
		if inUseVolume == volumeToDetach.VolumeName {
			return fmt.Errorf("DetachVolume failed for volume %q (spec.Name: %q) from node %q. Error: volume is still in use by node, according to Node status",
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name(),
				volumeToDetach.NodeName)
		}
	}

	// Volume is not marked as in use by node
	glog.Infof("Verified volume is safe to detach for volume %q (spec.Name: %q) from node %q.",
		volumeToDetach.VolumeName,
		volumeToDetach.VolumeSpec.Name(),
		volumeToDetach.NodeName)
	return nil
}

func (og *operationGenerator) GenerateMountVolumeFunc(
	waitForAttachTimeout time.Duration,
	volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (func() error, error) {
	// Get mounter plugin
	volumePlugin, err :=
		og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
	if err != nil || volumePlugin == nil {
		return nil, fmt.Errorf(
			"MountVolume.FindPluginBySpec failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			err)
	}

	volumeMounter, newMounterErr := volumePlugin.NewMounter(
		volumeToMount.VolumeSpec,
		volumeToMount.Pod,
		volume.VolumeOptions{})
	if newMounterErr != nil {
		return nil, fmt.Errorf(
			"MountVolume.NewMounter failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			newMounterErr)
	}

	// Get attacher, if possible
	attachableVolumePlugin, _ :=
		og.volumePluginMgr.FindAttachablePluginBySpec(volumeToMount.VolumeSpec)
	var volumeAttacher volume.Attacher
	if attachableVolumePlugin != nil {
		volumeAttacher, _ = attachableVolumePlugin.NewAttacher()
	}

	var fsGroup *int64
	if volumeToMount.Pod.Spec.SecurityContext != nil &&
		volumeToMount.Pod.Spec.SecurityContext.FSGroup != nil {
		fsGroup = volumeToMount.Pod.Spec.SecurityContext.FSGroup
	}

	return func() error {
		if volumeAttacher != nil {
			// Wait for attachable volumes to finish attaching
			glog.Infof(
				"Entering MountVolume.WaitForAttach for volume %q (spec.Name: %q) pod %q (UID: %q) DevicePath: %q",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				volumeToMount.DevicePath)

			devicePath, err := volumeAttacher.WaitForAttach(
				volumeToMount.VolumeSpec, volumeToMount.DevicePath, waitForAttachTimeout)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"MountVolume.WaitForAttach failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
			}

			glog.Infof(
				"MountVolume.WaitForAttach succeeded for volume %q (spec.Name: %q) pod %q (UID: %q).",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)

			deviceMountPath, err :=
				volumeAttacher.GetDeviceMountPath(volumeToMount.VolumeSpec)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"MountVolume.GetDeviceMountPath failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
			}

			// Mount device to global mount path
			err = volumeAttacher.MountDevice(
				volumeToMount.VolumeSpec,
				devicePath,
				deviceMountPath)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				err := fmt.Errorf(
					"MountVolume.MountDevice failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					err)
				og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FailedMountVolume, err.Error())
				return err
			}

			glog.Infof(
				"MountVolume.MountDevice succeeded for volume %q (spec.Name: %q) pod %q (UID: %q) device mount path %q",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				deviceMountPath)

			// Update actual state of world to reflect volume is globally mounted
			markDeviceMountedErr := actualStateOfWorld.MarkDeviceAsMounted(
				volumeToMount.VolumeName)
			if markDeviceMountedErr != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"MountVolume.MarkDeviceAsMounted failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					markDeviceMountedErr)
			}
		}

		if og.checkNodeCapabilitiesBeforeMount {
			if canMountErr := volumeMounter.CanMount(); canMountErr != nil {
				errMsg := fmt.Sprintf("Unable to mount volume %v (spec.Name: %v) on pod %v (UID: %v). Verify that your node machine has the required components before attempting to mount this volume type. %s", volumeToMount.VolumeName, volumeToMount.VolumeSpec.Name(), volumeToMount.Pod.Name, volumeToMount.Pod.UID, canMountErr.Error())
				og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FailedMountVolume, errMsg)
				glog.Errorf(errMsg)
				return fmt.Errorf(errMsg)
			}
		}

		// Execute mount
		mountErr := volumeMounter.SetUp(fsGroup)
		if mountErr != nil {
			// On failure, return error. Caller will log and retry.
			err := fmt.Errorf(
				"MountVolume.SetUp failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				mountErr)
			og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FailedMountVolume, err.Error())
			return err
		}

		glog.Infof(
			"MountVolume.SetUp succeeded for volume %q (spec.Name: %q) pod %q (UID: %q).",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID)

		// Update actual state of world
		markVolMountedErr := actualStateOfWorld.MarkVolumeAsMounted(
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			volumeToMount.VolumeName,
			volumeMounter,
			volumeToMount.OuterVolumeSpecName,
			volumeToMount.VolumeGidValue)
		if markVolMountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"MountVolume.MarkVolumeAsMounted failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				markVolMountedErr)
		}

		return nil
	}, nil
}

func (og *operationGenerator) GenerateUnmountVolumeFunc(
	volumeToUnmount MountedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (func() error, error) {
	// Get mountable plugin
	volumePlugin, err :=
		og.volumePluginMgr.FindPluginByName(volumeToUnmount.PluginName)
	if err != nil || volumePlugin == nil {
		return nil, fmt.Errorf(
			"UnmountVolume.FindPluginByName failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) err=%v",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID,
			err)
	}

	volumeUnmounter, newUnmounterErr := volumePlugin.NewUnmounter(
		volumeToUnmount.InnerVolumeSpecName, volumeToUnmount.PodUID)
	if newUnmounterErr != nil {
		return nil, fmt.Errorf(
			"UnmountVolume.NewUnmounter failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) err=%v",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID,
			newUnmounterErr)
	}

	return func() error {
		// Execute unmount
		unmountErr := volumeUnmounter.TearDown()
		if unmountErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"UnmountVolume.TearDown failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToUnmount.VolumeName,
				volumeToUnmount.OuterVolumeSpecName,
				volumeToUnmount.PodName,
				volumeToUnmount.PodUID,
				unmountErr)
		}

		glog.Infof(
			"UnmountVolume.TearDown succeeded for volume %q (OuterVolumeSpecName: %q) pod %q (UID: %q). InnerVolumeSpecName %q. PluginName %q, VolumeGidValue %q",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID,
			volumeToUnmount.InnerVolumeSpecName,
			volumeToUnmount.PluginName,
			volumeToUnmount.VolumeGidValue)

		// Update actual state of world
		markVolMountedErr := actualStateOfWorld.MarkVolumeAsUnmounted(
			volumeToUnmount.PodName, volumeToUnmount.VolumeName)
		if markVolMountedErr != nil {
			// On failure, just log and exit
			glog.Errorf(
				"UnmountVolume.MarkVolumeAsUnmounted failed for volume %q (volume.spec.Name: %q) pod %q (UID: %q) with: %v",
				volumeToUnmount.VolumeName,
				volumeToUnmount.OuterVolumeSpecName,
				volumeToUnmount.PodName,
				volumeToUnmount.PodUID,
				markVolMountedErr)
		}

		return nil
	}, nil
}

func (og *operationGenerator) GenerateUnmountDeviceFunc(
	deviceToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	mounter mount.Interface) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err :=
		og.volumePluginMgr.FindAttachablePluginBySpec(deviceToDetach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"UnmountDevice.FindAttachablePluginBySpec failed for volume %q (spec.Name: %q) with: %v",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name(),
			err)
	}

	volumeDetacher, err := attachableVolumePlugin.NewDetacher()
	if err != nil {
		return nil, fmt.Errorf(
			"UnmountDevice.NewDetacher failed for volume %q (spec.Name: %q) with: %v",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name(),
			err)
	}

	volumeAttacher, err := attachableVolumePlugin.NewAttacher()
	if err != nil {
		return nil, fmt.Errorf(
			"UnmountDevice.NewAttacher failed for volume %q (spec.Name: %q) with: %v",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name(),
			err)
	}

	return func() error {
		deviceMountPath, err :=
			volumeAttacher.GetDeviceMountPath(deviceToDetach.VolumeSpec)
		if err != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"GetDeviceMountPath failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				err)
		}
		refs, err := attachableVolumePlugin.GetDeviceMountRefs(deviceMountPath)

		if err != nil || hasMountRefs(deviceMountPath, refs) {
			if err == nil {
				err = fmt.Errorf("The device mount path %q is still mounted by other references %v", deviceMountPath, refs)
			}
			return fmt.Errorf(
				"GetDeviceMountRefs check failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				err)
		}
		// Execute unmount
		unmountDeviceErr := volumeDetacher.UnmountDevice(deviceMountPath)
		if unmountDeviceErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"UnmountDevice failed for volume %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				unmountDeviceErr)
		}
		// Before logging that UnmountDevice succeeded and moving on,
		// use mounter.PathIsDevice to check if the path is a device,
		// if so use mounter.DeviceOpened to check if the device is in use anywhere
		// else on the system. Retry if it returns true.
		isDevicePath, devicePathErr := mounter.PathIsDevice(deviceToDetach.DevicePath)
		var deviceOpened bool
		var deviceOpenedErr error
		if !isDevicePath && devicePathErr == nil {
			// not a device path or path doesn't exist
			//TODO: refer to #36092
			glog.V(3).Infof("Not checking device path %s", deviceToDetach.DevicePath)
			deviceOpened = false
		} else {
			deviceOpened, deviceOpenedErr = mounter.DeviceOpened(deviceToDetach.DevicePath)
			if deviceOpenedErr != nil {
				return fmt.Errorf(
					"UnmountDevice.DeviceOpened failed for volume %q (spec.Name: %q) with: %v",
					deviceToDetach.VolumeName,
					deviceToDetach.VolumeSpec.Name(),
					deviceOpenedErr)
			}
		}
		// The device is still in use elsewhere. Caller will log and retry.
		if deviceOpened {
			return fmt.Errorf(
				"UnmountDevice failed for volume %q (spec.Name: %q) because the device is in use when it was no longer expected to be in use",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name())
		}

		glog.Infof(
			"UnmountDevice succeeded for volume %q (spec.Name: %q).",
			deviceToDetach.VolumeName,
			deviceToDetach.VolumeSpec.Name())

		// Update actual state of world
		markDeviceUnmountedErr := actualStateOfWorld.MarkDeviceAsUnmounted(
			deviceToDetach.VolumeName)
		if markDeviceUnmountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"MarkDeviceAsUnmounted failed for device %q (spec.Name: %q) with: %v",
				deviceToDetach.VolumeName,
				deviceToDetach.VolumeSpec.Name(),
				markDeviceUnmountedErr)
		}

		return nil
	}, nil
}

func (og *operationGenerator) GenerateVerifyControllerAttachedVolumeFunc(
	volumeToMount VolumeToMount,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (func() error, error) {
	return func() error {
		if !volumeToMount.PluginIsAttachable {
			// If the volume does not implement the attacher interface, it is
			// assumed to be attached and the actual state of the world is
			// updated accordingly.

			addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
				volumeToMount.VolumeName, volumeToMount.VolumeSpec, nodeName, "" /* devicePath */)
			if addVolumeNodeErr != nil {
				// On failure, return error. Caller will log and retry.
				return fmt.Errorf(
					"VerifyControllerAttachedVolume.MarkVolumeAsAttachedByUniqueVolumeName failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					addVolumeNodeErr)
			}

			return nil
		}

		if !volumeToMount.ReportedInUse {
			// If the given volume has not yet been added to the list of
			// VolumesInUse in the node's volume status, do not proceed, return
			// error. Caller will log and retry. The node status is updated
			// periodically by kubelet, so it may take as much as 10 seconds
			// before this clears.
			// Issue #28141 to enable on demand status updates.
			return fmt.Errorf("Volume %q (spec.Name: %q) pod %q (UID: %q) has not yet been added to the list of VolumesInUse in the node's volume status",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)
		}

		// Fetch current node object
		node, fetchErr := og.kubeClient.Core().Nodes().Get(string(nodeName), metav1.GetOptions{})
		if fetchErr != nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"VerifyControllerAttachedVolume failed fetching node from API server. Volume %q (spec.Name: %q) pod %q (UID: %q). Error: %v",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID,
				fetchErr)
		}

		if node == nil {
			// On failure, return error. Caller will log and retry.
			return fmt.Errorf(
				"VerifyControllerAttachedVolume failed. Volume %q (spec.Name: %q) pod %q (UID: %q). Error: node object retrieved from API server is nil",
				volumeToMount.VolumeName,
				volumeToMount.VolumeSpec.Name(),
				volumeToMount.PodName,
				volumeToMount.Pod.UID)
		}

		for _, attachedVolume := range node.Status.VolumesAttached {
			if attachedVolume.Name == volumeToMount.VolumeName {
				addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
					v1.UniqueVolumeName(""), volumeToMount.VolumeSpec, nodeName, attachedVolume.DevicePath)
				glog.Infof("Controller successfully attached volume %q (spec.Name: %q) pod %q (UID: %q) devicePath: %q",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					attachedVolume.DevicePath)

				if addVolumeNodeErr != nil {
					// On failure, return error. Caller will log and retry.
					return fmt.Errorf(
						"VerifyControllerAttachedVolume.MarkVolumeAsAttached failed for volume %q (spec.Name: %q) pod %q (UID: %q) with: %v",
						volumeToMount.VolumeName,
						volumeToMount.VolumeSpec.Name(),
						volumeToMount.PodName,
						volumeToMount.Pod.UID,
						addVolumeNodeErr)
				}
				return nil
			}
		}

		// Volume not attached, return error. Caller will log and retry.
		return fmt.Errorf("Volume %q (spec.Name: %q) pod %q (UID: %q) is not yet attached according to node status",
			volumeToMount.VolumeName,
			volumeToMount.VolumeSpec.Name(),
			volumeToMount.PodName,
			volumeToMount.Pod.UID)
	}, nil
}

func (og *operationGenerator) verifyVolumeIsSafeToDetach(
	volumeToDetach AttachedVolume) error {
	// Fetch current node object
	node, fetchErr := og.kubeClient.Core().Nodes().Get(string(volumeToDetach.NodeName), metav1.GetOptions{})
	if fetchErr != nil {
		if errors.IsNotFound(fetchErr) {
			glog.Warningf("Node %q not found on API server. DetachVolume will skip safe to detach check.",
				volumeToDetach.NodeName,
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name())
			return nil
		}

		// On failure, return error. Caller will log and retry.
		return fmt.Errorf(
			"DetachVolume failed fetching node from API server for volume %q (spec.Name: %q) from node %q with: %v",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName,
			fetchErr)
	}

	if node == nil {
		// On failure, return error. Caller will log and retry.
		return fmt.Errorf(
			"DetachVolume failed fetching node from API server for volume %q (spec.Name: %q) from node %q. Error: node object retrieved from API server is nil",
			volumeToDetach.VolumeName,
			volumeToDetach.VolumeSpec.Name(),
			volumeToDetach.NodeName)
	}

	for _, inUseVolume := range node.Status.VolumesInUse {
		if inUseVolume == volumeToDetach.VolumeName {
			return fmt.Errorf("DetachVolume failed for volume %q (spec.Name: %q) from node %q. Error: volume is still in use by node, according to Node status",
				volumeToDetach.VolumeName,
				volumeToDetach.VolumeSpec.Name(),
				volumeToDetach.NodeName)
		}
	}

	// Volume is not marked as in use by node
	glog.Infof("Verified volume is safe to detach for volume %q (spec.Name: %q) from node %q.",
		volumeToDetach.VolumeName,
		volumeToDetach.VolumeSpec.Name(),
		volumeToDetach.NodeName)
	return nil
}
