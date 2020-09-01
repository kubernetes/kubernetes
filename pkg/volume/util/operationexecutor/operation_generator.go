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
	"context"
	goerrors "errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	volerr "k8s.io/cloud-provider/volume/errors"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	ioutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

const (
	unknownVolumePlugin           string = "UnknownVolumePlugin"
	unknownAttachableVolumePlugin string = "UnknownAttachableVolumePlugin"
)

// InTreeToCSITranslator contains methods required to check migratable status
// and perform translations from InTree PVs and Inline to CSI
type InTreeToCSITranslator interface {
	IsPVMigratable(pv *v1.PersistentVolume) bool
	IsInlineMigratable(vol *v1.Volume) bool
	IsMigratableIntreePluginByName(inTreePluginName string) bool
	GetInTreePluginNameFromSpec(pv *v1.PersistentVolume, vol *v1.Volume) (string, error)
	GetCSINameFromInTreeName(pluginName string) (string, error)
	TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)
	TranslateInTreeInlineVolumeToCSI(volume *v1.Volume) (*v1.PersistentVolume, error)
}

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

	// blkUtil provides volume path related operations for block volume
	blkUtil volumepathhandler.BlockVolumePathHandler

	translator InTreeToCSITranslator
}

// NewOperationGenerator is returns instance of operationGenerator
func NewOperationGenerator(kubeClient clientset.Interface,
	volumePluginMgr *volume.VolumePluginMgr,
	recorder record.EventRecorder,
	checkNodeCapabilitiesBeforeMount bool,
	blkUtil volumepathhandler.BlockVolumePathHandler) OperationGenerator {

	return &operationGenerator{
		kubeClient:                       kubeClient,
		volumePluginMgr:                  volumePluginMgr,
		recorder:                         recorder,
		checkNodeCapabilitiesBeforeMount: checkNodeCapabilitiesBeforeMount,
		blkUtil:                          blkUtil,
		translator:                       csitrans.New(),
	}
}

// OperationGenerator interface that extracts out the functions from operation_executor to make it dependency injectable
type OperationGenerator interface {
	// Generates the MountVolume function needed to perform the mount of a volume plugin
	GenerateMountVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater, isRemount bool) volumetypes.GeneratedOperations

	// Generates the UnmountVolume function needed to perform the unmount of a volume plugin
	GenerateUnmountVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, podsDir string) (volumetypes.GeneratedOperations, error)

	// Generates the AttachVolume function needed to perform attach of a volume plugin
	GenerateAttachVolumeFunc(volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) volumetypes.GeneratedOperations

	// Generates the DetachVolume function needed to perform the detach of a volume plugin
	GenerateDetachVolumeFunc(volumeToDetach AttachedVolume, verifySafeToDetach bool, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the VolumesAreAttached function needed to verify if volume plugins are attached
	GenerateVolumesAreAttachedFunc(attachedVolumes []AttachedVolume, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the UnMountDevice function needed to perform the unmount of a device
	GenerateUnmountDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter hostutil.HostUtils) (volumetypes.GeneratedOperations, error)

	// Generates the function needed to check if the attach_detach controller has attached the volume plugin
	GenerateVerifyControllerAttachedVolumeFunc(volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the MapVolume function needed to perform the map of a volume plugin
	GenerateMapVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the UnmapVolume function needed to perform the unmap of a volume plugin
	GenerateUnmapVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the UnmapDevice function needed to perform the unmap of a device
	GenerateUnmapDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter hostutil.HostUtils) (volumetypes.GeneratedOperations, error)

	// GetVolumePluginMgr returns volume plugin manager
	GetVolumePluginMgr() *volume.VolumePluginMgr

	// GetCSITranslator returns the CSI Translation Library
	GetCSITranslator() InTreeToCSITranslator

	GenerateBulkVolumeVerifyFunc(
		map[types.NodeName][]*volume.Spec,
		string,
		map[*volume.Spec]v1.UniqueVolumeName, ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error)

	GenerateExpandVolumeFunc(*v1.PersistentVolumeClaim, *v1.PersistentVolume) (volumetypes.GeneratedOperations, error)

	// Generates the volume file system resize function, which can resize volume's file system to expected size without unmounting the volume.
	GenerateExpandInUseVolumeFunc(volumeToMount VolumeToMount, actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error)
}

func (og *operationGenerator) GenerateVolumesAreAttachedFunc(
	attachedVolumes []AttachedVolume,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	// volumesPerPlugin maps from a volume plugin to a list of volume specs which belong
	// to this type of plugin
	volumesPerPlugin := make(map[string][]*volume.Spec)
	// volumeSpecMap maps from a volume spec to its unique volumeName which will be used
	// when calling MarkVolumeAsDetached
	volumeSpecMap := make(map[*volume.Spec]v1.UniqueVolumeName)

	// Iterate each volume spec and put them into a map index by the pluginName
	for _, volumeAttached := range attachedVolumes {
		if volumeAttached.VolumeSpec == nil {
			klog.Errorf("VerifyVolumesAreAttached.GenerateVolumesAreAttachedFunc: nil spec for volume %s", volumeAttached.VolumeName)
			continue
		}
		volumePlugin, err :=
			og.volumePluginMgr.FindPluginBySpec(volumeAttached.VolumeSpec)
		if err != nil || volumePlugin == nil {
			klog.Errorf(volumeAttached.GenerateErrorDetailed("VolumesAreAttached.FindPluginBySpec failed", err).Error())
			continue
		}
		volumeSpecList, pluginExists := volumesPerPlugin[volumePlugin.GetPluginName()]
		if !pluginExists {
			volumeSpecList = []*volume.Spec{}
		}
		volumeSpecList = append(volumeSpecList, volumeAttached.VolumeSpec)
		volumesPerPlugin[volumePlugin.GetPluginName()] = volumeSpecList
		// Migration: VolumeSpecMap contains original VolumeName for use in ActualStateOfWorld
		volumeSpecMap[volumeAttached.VolumeSpec] = volumeAttached.VolumeName
	}

	volumesAreAttachedFunc := func() (error, error) {

		// For each volume plugin, pass the list of volume specs to VolumesAreAttached to check
		// whether the volumes are still attached.
		for pluginName, volumesSpecs := range volumesPerPlugin {
			attachableVolumePlugin, err :=
				og.volumePluginMgr.FindAttachablePluginByName(pluginName)
			if err != nil || attachableVolumePlugin == nil {
				klog.Errorf(
					"VolumeAreAttached.FindAttachablePluginBySpec failed for plugin %q with: %v",
					pluginName,
					err)
				continue
			}

			volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()
			if newAttacherErr != nil {
				klog.Errorf(
					"VolumesAreAttached.NewAttacher failed for getting plugin %q with: %v",
					pluginName,
					newAttacherErr)
				continue
			}

			attached, areAttachedErr := volumeAttacher.VolumesAreAttached(volumesSpecs, nodeName)
			if areAttachedErr != nil {
				klog.Errorf(
					"VolumesAreAttached failed for checking on node %q with: %v",
					nodeName,
					areAttachedErr)
				continue
			}

			for spec, check := range attached {
				if !check {
					actualStateOfWorld.MarkVolumeAsDetached(volumeSpecMap[spec], nodeName)
					klog.V(1).Infof("VerifyVolumesAreAttached determined volume %q (spec.Name: %q) is no longer attached to node %q, therefore it was marked as detached.",
						volumeSpecMap[spec], spec.Name(), nodeName)
				}
			}
		}

		return nil, nil
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "verify_volumes_are_attached_per_node",
		OperationFunc:     volumesAreAttachedFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume("<n/a>", nil), "verify_volumes_are_attached_per_node"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil
}

func (og *operationGenerator) GenerateBulkVolumeVerifyFunc(
	pluginNodeVolumes map[types.NodeName][]*volume.Spec,
	pluginName string,
	volumeSpecMap map[*volume.Spec]v1.UniqueVolumeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {

	// Migration: All inputs already should be translated by caller for this
	// function except volumeSpecMap which contains original volume names for
	// use with actualStateOfWorld

	bulkVolumeVerifyFunc := func() (error, error) {
		attachableVolumePlugin, err :=
			og.volumePluginMgr.FindAttachablePluginByName(pluginName)
		if err != nil || attachableVolumePlugin == nil {
			klog.Errorf(
				"BulkVerifyVolume.FindAttachablePluginBySpec failed for plugin %q with: %v",
				pluginName,
				err)
			return nil, nil
		}

		volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()

		if newAttacherErr != nil {
			klog.Errorf(
				"BulkVerifyVolume.NewAttacher failed for getting plugin %q with: %v",
				attachableVolumePlugin,
				newAttacherErr)
			return nil, nil
		}
		bulkVolumeVerifier, ok := volumeAttacher.(volume.BulkVolumeVerifier)

		if !ok {
			klog.Errorf("BulkVerifyVolume failed to type assert attacher %q", bulkVolumeVerifier)
			return nil, nil
		}

		attached, bulkAttachErr := bulkVolumeVerifier.BulkVerifyVolumes(pluginNodeVolumes)
		if bulkAttachErr != nil {
			klog.Errorf("BulkVerifyVolume.BulkVerifyVolumes Error checking volumes are attached with %v", bulkAttachErr)
			return nil, nil
		}

		for nodeName, volumeSpecs := range pluginNodeVolumes {
			for _, volumeSpec := range volumeSpecs {
				nodeVolumeSpecs, nodeChecked := attached[nodeName]

				if !nodeChecked {
					klog.V(2).Infof("VerifyVolumesAreAttached.BulkVerifyVolumes failed for node %q and leaving volume %q as attached",
						nodeName,
						volumeSpec.Name())
					continue
				}

				check := nodeVolumeSpecs[volumeSpec]

				if !check {
					klog.V(2).Infof("VerifyVolumesAreAttached.BulkVerifyVolumes failed for node %q and volume %q",
						nodeName,
						volumeSpec.Name())
					actualStateOfWorld.MarkVolumeAsDetached(volumeSpecMap[volumeSpec], nodeName)
				}
			}
		}

		return nil, nil
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "verify_volumes_are_attached",
		OperationFunc:     bulkVolumeVerifyFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(pluginName, nil), "verify_volumes_are_attached"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil

}

func (og *operationGenerator) GenerateAttachVolumeFunc(
	volumeToAttach VolumeToAttach,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) volumetypes.GeneratedOperations {

	attachVolumeFunc := func() (error, error) {
		attachableVolumePlugin, err :=
			og.volumePluginMgr.FindAttachablePluginBySpec(volumeToAttach.VolumeSpec)
		if err != nil || attachableVolumePlugin == nil {
			return volumeToAttach.GenerateError("AttachVolume.FindAttachablePluginBySpec failed", err)
		}

		volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()
		if newAttacherErr != nil {
			return volumeToAttach.GenerateError("AttachVolume.NewAttacher failed", newAttacherErr)
		}

		// Execute attach
		devicePath, attachErr := volumeAttacher.Attach(
			volumeToAttach.VolumeSpec, volumeToAttach.NodeName)

		if attachErr != nil {
			uncertainNode := volumeToAttach.NodeName
			if derr, ok := attachErr.(*volerr.DanglingAttachError); ok {
				uncertainNode = derr.CurrentNode
			}
			addErr := actualStateOfWorld.MarkVolumeAsUncertain(
				volumeToAttach.VolumeName,
				volumeToAttach.VolumeSpec,
				uncertainNode)
			if addErr != nil {
				klog.Errorf("AttachVolume.MarkVolumeAsUncertain fail to add the volume %q to actual state with %s", volumeToAttach.VolumeName, addErr)
			}

			// On failure, return error. Caller will log and retry.
			return volumeToAttach.GenerateError("AttachVolume.Attach failed", attachErr)
		}

		// Successful attach event is useful for user debugging
		simpleMsg, _ := volumeToAttach.GenerateMsg("AttachVolume.Attach succeeded", "")
		for _, pod := range volumeToAttach.ScheduledPods {
			og.recorder.Eventf(pod, v1.EventTypeNormal, kevents.SuccessfulAttachVolume, simpleMsg)
		}
		klog.Infof(volumeToAttach.GenerateMsgDetailed("AttachVolume.Attach succeeded", ""))

		// Update actual state of world
		addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
			v1.UniqueVolumeName(""), volumeToAttach.VolumeSpec, volumeToAttach.NodeName, devicePath)
		if addVolumeNodeErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToAttach.GenerateError("AttachVolume.MarkVolumeAsAttached failed", addVolumeNodeErr)
		}

		return nil, nil
	}

	eventRecorderFunc := func(err *error) {
		if *err != nil {
			for _, pod := range volumeToAttach.ScheduledPods {
				og.recorder.Eventf(pod, v1.EventTypeWarning, kevents.FailedAttachVolume, (*err).Error())
			}
		}
	}

	attachableVolumePluginName := unknownAttachableVolumePlugin

	// Get attacher plugin
	attachableVolumePlugin, err :=
		og.volumePluginMgr.FindAttachablePluginBySpec(volumeToAttach.VolumeSpec)
	// It's ok to ignore the error, returning error is not expected from this function.
	// If an error case occurred during the function generation, this error case(skipped one) will also trigger an error
	// while the generated function is executed. And those errors will be handled during the execution of the generated
	// function with a back off policy.
	if err == nil && attachableVolumePlugin != nil {
		attachableVolumePluginName = attachableVolumePlugin.GetPluginName()
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "volume_attach",
		OperationFunc:     attachVolumeFunc,
		EventRecorderFunc: eventRecorderFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(attachableVolumePluginName, volumeToAttach.VolumeSpec), "volume_attach"),
	}
}

func (og *operationGenerator) GetVolumePluginMgr() *volume.VolumePluginMgr {
	return og.volumePluginMgr
}

func (og *operationGenerator) GetCSITranslator() InTreeToCSITranslator {
	return og.translator
}

func (og *operationGenerator) GenerateDetachVolumeFunc(
	volumeToDetach AttachedVolume,
	verifySafeToDetach bool,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	var volumeName string
	var attachableVolumePlugin volume.AttachableVolumePlugin
	var pluginName string
	var err error

	if volumeToDetach.VolumeSpec != nil {
		attachableVolumePlugin, err = findDetachablePluginBySpec(volumeToDetach.VolumeSpec, og.volumePluginMgr)
		if err != nil || attachableVolumePlugin == nil {
			return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.findDetachablePluginBySpec failed", err)
		}

		volumeName, err =
			attachableVolumePlugin.GetVolumeName(volumeToDetach.VolumeSpec)
		if err != nil {
			return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.GetVolumeName failed", err)
		}
	} else {
		// Get attacher plugin and the volumeName by splitting the volume unique name in case
		// there's no VolumeSpec: this happens only on attach/detach controller crash recovery
		// when a pod has been deleted during the controller downtime
		pluginName, volumeName, err = util.SplitUniqueName(volumeToDetach.VolumeName)
		if err != nil {
			return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.SplitUniqueName failed", err)
		}

		attachableVolumePlugin, err = og.volumePluginMgr.FindAttachablePluginByName(pluginName)
		if err != nil || attachableVolumePlugin == nil {
			return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.FindAttachablePluginByName failed", err)
		}

	}

	if pluginName == "" {
		pluginName = attachableVolumePlugin.GetPluginName()
	}

	volumeDetacher, err := attachableVolumePlugin.NewDetacher()
	if err != nil {
		return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.NewDetacher failed", err)
	}

	getVolumePluginMgrFunc := func() (error, error) {
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
			return volumeToDetach.GenerateError("DetachVolume.Detach failed", err)
		}

		klog.Infof(volumeToDetach.GenerateMsgDetailed("DetachVolume.Detach succeeded", ""))

		// Update actual state of world
		actualStateOfWorld.MarkVolumeAsDetached(
			volumeToDetach.VolumeName, volumeToDetach.NodeName)

		return nil, nil
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "volume_detach",
		OperationFunc:     getVolumePluginMgrFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(pluginName, volumeToDetach.VolumeSpec), "volume_detach"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil
}

func (og *operationGenerator) GenerateMountVolumeFunc(
	waitForAttachTimeout time.Duration,
	volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	isRemount bool) volumetypes.GeneratedOperations {

	volumePluginName := unknownVolumePlugin
	volumePlugin, err :=
		og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
	if err == nil && volumePlugin != nil {
		volumePluginName = volumePlugin.GetPluginName()
	}

	mountVolumeFunc := func() (error, error) {
		// Get mounter plugin
		volumePlugin, err := og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
		if err != nil || volumePlugin == nil {
			return volumeToMount.GenerateError("MountVolume.FindPluginBySpec failed", err)
		}

		affinityErr := checkNodeAffinity(og, volumeToMount)
		if affinityErr != nil {
			return volumeToMount.GenerateError("MountVolume.NodeAffinity check failed", affinityErr)
		}

		volumeMounter, newMounterErr := volumePlugin.NewMounter(
			volumeToMount.VolumeSpec,
			volumeToMount.Pod,
			volume.VolumeOptions{})
		if newMounterErr != nil {
			return volumeToMount.GenerateError("MountVolume.NewMounter initialization failed", newMounterErr)

		}

		mountCheckError := checkMountOptionSupport(og, volumeToMount, volumePlugin)

		if mountCheckError != nil {
			return volumeToMount.GenerateError("MountVolume.MountOptionSupport check failed", mountCheckError)
		}

		// Get attacher, if possible
		attachableVolumePlugin, _ :=
			og.volumePluginMgr.FindAttachablePluginBySpec(volumeToMount.VolumeSpec)
		var volumeAttacher volume.Attacher
		if attachableVolumePlugin != nil {
			volumeAttacher, _ = attachableVolumePlugin.NewAttacher()
		}

		// get deviceMounter, if possible
		deviceMountableVolumePlugin, _ := og.volumePluginMgr.FindDeviceMountablePluginBySpec(volumeToMount.VolumeSpec)
		var volumeDeviceMounter volume.DeviceMounter
		if deviceMountableVolumePlugin != nil {
			volumeDeviceMounter, _ = deviceMountableVolumePlugin.NewDeviceMounter()
		}

		var fsGroup *int64
		var fsGroupChangePolicy *v1.PodFSGroupChangePolicy
		if podSc := volumeToMount.Pod.Spec.SecurityContext; podSc != nil {
			if podSc.FSGroup != nil {
				fsGroup = podSc.FSGroup
			}
			if podSc.FSGroupChangePolicy != nil {
				fsGroupChangePolicy = podSc.FSGroupChangePolicy
			}
		}

		devicePath := volumeToMount.DevicePath
		if volumeAttacher != nil {
			// Wait for attachable volumes to finish attaching
			klog.Infof(volumeToMount.GenerateMsgDetailed("MountVolume.WaitForAttach entering", fmt.Sprintf("DevicePath %q", volumeToMount.DevicePath)))

			devicePath, err = volumeAttacher.WaitForAttach(
				volumeToMount.VolumeSpec, devicePath, volumeToMount.Pod, waitForAttachTimeout)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("MountVolume.WaitForAttach failed", err)
			}

			klog.Infof(volumeToMount.GenerateMsgDetailed("MountVolume.WaitForAttach succeeded", fmt.Sprintf("DevicePath %q", devicePath)))
		}

		var resizeDone bool
		var resizeError error
		resizeOptions := volume.NodeResizeOptions{
			DevicePath: devicePath,
		}

		if volumeDeviceMounter != nil {
			deviceMountPath, err :=
				volumeDeviceMounter.GetDeviceMountPath(volumeToMount.VolumeSpec)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("MountVolume.GetDeviceMountPath failed", err)
			}

			// Mount device to global mount path
			err = volumeDeviceMounter.MountDevice(
				volumeToMount.VolumeSpec,
				devicePath,
				deviceMountPath)
			if err != nil {
				og.checkForFailedMount(volumeToMount, err)
				og.markDeviceErrorState(volumeToMount, devicePath, deviceMountPath, err, actualStateOfWorld)
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("MountVolume.MountDevice failed", err)
			}

			klog.Infof(volumeToMount.GenerateMsgDetailed("MountVolume.MountDevice succeeded", fmt.Sprintf("device mount path %q", deviceMountPath)))

			// Update actual state of world to reflect volume is globally mounted
			markDeviceMountedErr := actualStateOfWorld.MarkDeviceAsMounted(
				volumeToMount.VolumeName, devicePath, deviceMountPath)
			if markDeviceMountedErr != nil {
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("MountVolume.MarkDeviceAsMounted failed", markDeviceMountedErr)
			}

			// If volume expansion is performed after MountDevice but before SetUp then
			// deviceMountPath and deviceStagePath is going to be the same.
			// Deprecation: Calling NodeExpandVolume after NodeStage/MountDevice will be deprecated
			// in a future version of k8s.
			resizeOptions.DeviceMountPath = deviceMountPath
			resizeOptions.DeviceStagePath = deviceMountPath
			resizeOptions.CSIVolumePhase = volume.CSIVolumeStaged

			// NodeExpandVolume will resize the file system if user has requested a resize of
			// underlying persistent volume and is allowed to do so.
			resizeDone, resizeError = og.nodeExpandVolume(volumeToMount, actualStateOfWorld, resizeOptions)

			if resizeError != nil {
				klog.Errorf("MountVolume.NodeExpandVolume failed with %v", resizeError)
				return volumeToMount.GenerateError("MountVolume.MountDevice failed while expanding volume", resizeError)
			}
		}

		if og.checkNodeCapabilitiesBeforeMount {
			if canMountErr := volumeMounter.CanMount(); canMountErr != nil {
				err = fmt.Errorf(
					"Verify that your node machine has the required components before attempting to mount this volume type. %s",
					canMountErr)
				return volumeToMount.GenerateError("MountVolume.CanMount failed", err)
			}
		}

		// Execute mount
		mountErr := volumeMounter.SetUp(volume.MounterArgs{
			FsUser:              ioutil.FsUserFrom(volumeToMount.Pod),
			FsGroup:             fsGroup,
			DesiredSize:         volumeToMount.DesiredSizeLimit,
			FSGroupChangePolicy: fsGroupChangePolicy,
		})
		// Update actual state of world
		markOpts := MarkVolumeOpts{
			PodName:             volumeToMount.PodName,
			PodUID:              volumeToMount.Pod.UID,
			VolumeName:          volumeToMount.VolumeName,
			Mounter:             volumeMounter,
			OuterVolumeSpecName: volumeToMount.OuterVolumeSpecName,
			VolumeGidVolume:     volumeToMount.VolumeGidValue,
			VolumeSpec:          volumeToMount.VolumeSpec,
			VolumeMountState:    VolumeMounted,
		}
		if mountErr != nil {
			og.checkForFailedMount(volumeToMount, mountErr)
			og.markVolumeErrorState(volumeToMount, markOpts, mountErr, actualStateOfWorld)
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MountVolume.SetUp failed", mountErr)
		}

		_, detailedMsg := volumeToMount.GenerateMsg("MountVolume.SetUp succeeded", "")
		verbosity := klog.Level(1)
		if isRemount {
			verbosity = klog.Level(4)
		}
		klog.V(verbosity).Infof(detailedMsg)
		resizeOptions.DeviceMountPath = volumeMounter.GetPath()
		resizeOptions.CSIVolumePhase = volume.CSIVolumePublished

		// We need to call resizing here again in case resizing was not done during device mount. There could be
		// two reasons of that:
		//	- Volume does not support DeviceMounter interface.
		//	- In case of CSI the volume does not have node stage_unstage capability.
		if !resizeDone {
			_, resizeError = og.nodeExpandVolume(volumeToMount, actualStateOfWorld, resizeOptions)
			if resizeError != nil {
				klog.Errorf("MountVolume.NodeExpandVolume failed with %v", resizeError)
				return volumeToMount.GenerateError("MountVolume.Setup failed while expanding volume", resizeError)
			}
		}

		markVolMountedErr := actualStateOfWorld.MarkVolumeAsMounted(markOpts)
		if markVolMountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MountVolume.MarkVolumeAsMounted failed", markVolMountedErr)
		}

		return nil, nil
	}

	eventRecorderFunc := func(err *error) {
		if *err != nil {
			og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FailedMountVolume, (*err).Error())
		}
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "volume_mount",
		OperationFunc:     mountVolumeFunc,
		EventRecorderFunc: eventRecorderFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(volumePluginName, volumeToMount.VolumeSpec), "volume_mount"),
	}
}

func (og *operationGenerator) checkForFailedMount(volumeToMount VolumeToMount, mountError error) {
	pv := volumeToMount.VolumeSpec.PersistentVolume
	if pv == nil {
		return
	}

	if volumetypes.IsFilesystemMismatchError(mountError) {
		simpleMsg, _ := volumeToMount.GenerateMsg("MountVolume failed", mountError.Error())
		og.recorder.Eventf(pv, v1.EventTypeWarning, kevents.FailedMountOnFilesystemMismatch, simpleMsg)
	}
}

func (og *operationGenerator) markDeviceErrorState(volumeToMount VolumeToMount, devicePath, deviceMountPath string, mountError error, actualStateOfWorld ActualStateOfWorldMounterUpdater) {
	if volumetypes.IsOperationFinishedError(mountError) &&
		actualStateOfWorld.GetDeviceMountState(volumeToMount.VolumeName) == DeviceMountUncertain {
		// Only devices which were uncertain can be marked as unmounted
		markDeviceUnmountError := actualStateOfWorld.MarkDeviceAsUnmounted(volumeToMount.VolumeName)
		if markDeviceUnmountError != nil {
			klog.Errorf(volumeToMount.GenerateErrorDetailed("MountDevice.MarkDeviceAsUnmounted failed", markDeviceUnmountError).Error())
		}
		return
	}

	if volumetypes.IsUncertainProgressError(mountError) &&
		actualStateOfWorld.GetDeviceMountState(volumeToMount.VolumeName) == DeviceNotMounted {
		// only devices which are not mounted can be marked as uncertain. We do not want to mark a device
		// which was previously marked as mounted here as uncertain.
		markDeviceUncertainError := actualStateOfWorld.MarkDeviceAsUncertain(volumeToMount.VolumeName, devicePath, deviceMountPath)
		if markDeviceUncertainError != nil {
			klog.Errorf(volumeToMount.GenerateErrorDetailed("MountDevice.MarkDeviceAsUncertain failed", markDeviceUncertainError).Error())
		}
	}

}

func (og *operationGenerator) markVolumeErrorState(volumeToMount VolumeToMount, markOpts MarkVolumeOpts, mountError error, actualStateOfWorld ActualStateOfWorldMounterUpdater) {
	if volumetypes.IsOperationFinishedError(mountError) &&
		actualStateOfWorld.GetVolumeMountState(volumeToMount.VolumeName, markOpts.PodName) == VolumeMountUncertain {
		t := actualStateOfWorld.MarkVolumeAsUnmounted(volumeToMount.PodName, volumeToMount.VolumeName)
		if t != nil {
			klog.Errorf(volumeToMount.GenerateErrorDetailed("MountVolume.MarkVolumeAsUnmounted failed", t).Error())
		}
		return
	}

	if volumetypes.IsUncertainProgressError(mountError) &&
		actualStateOfWorld.GetVolumeMountState(volumeToMount.VolumeName, markOpts.PodName) == VolumeNotMounted {
		t := actualStateOfWorld.MarkVolumeMountAsUncertain(markOpts)
		if t != nil {
			klog.Errorf(volumeToMount.GenerateErrorDetailed("MountVolume.MarkVolumeMountAsUncertain failed", t).Error())
		}
	}

}

func (og *operationGenerator) GenerateUnmountVolumeFunc(
	volumeToUnmount MountedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	podsDir string) (volumetypes.GeneratedOperations, error) {
	// Get mountable plugin
	volumePlugin, err := og.volumePluginMgr.FindPluginByName(volumeToUnmount.PluginName)
	if err != nil || volumePlugin == nil {
		return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmountVolume.FindPluginByName failed", err)
	}
	volumeUnmounter, newUnmounterErr := volumePlugin.NewUnmounter(
		volumeToUnmount.InnerVolumeSpecName, volumeToUnmount.PodUID)
	if newUnmounterErr != nil {
		return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmountVolume.NewUnmounter failed", newUnmounterErr)
	}

	unmountVolumeFunc := func() (error, error) {
		subpather := og.volumePluginMgr.Host.GetSubpather()

		// Remove all bind-mounts for subPaths
		podDir := filepath.Join(podsDir, string(volumeToUnmount.PodUID))
		if err := subpather.CleanSubPaths(podDir, volumeToUnmount.InnerVolumeSpecName); err != nil {
			return volumeToUnmount.GenerateError("error cleaning subPath mounts", err)
		}

		// Execute unmount
		unmountErr := volumeUnmounter.TearDown()
		if unmountErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToUnmount.GenerateError("UnmountVolume.TearDown failed", unmountErr)
		}

		klog.Infof(
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
			klog.Errorf(volumeToUnmount.GenerateErrorDetailed("UnmountVolume.MarkVolumeAsUnmounted failed", markVolMountedErr).Error())
		}

		return nil, nil
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "volume_unmount",
		OperationFunc:     unmountVolumeFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(volumePlugin.GetPluginName(), volumeToUnmount.VolumeSpec), "volume_unmount"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil
}

func (og *operationGenerator) GenerateUnmountDeviceFunc(
	deviceToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	hostutil hostutil.HostUtils) (volumetypes.GeneratedOperations, error) {
	// Get DeviceMounter plugin
	deviceMountableVolumePlugin, err :=
		og.volumePluginMgr.FindDeviceMountablePluginByName(deviceToDetach.PluginName)
	if err != nil || deviceMountableVolumePlugin == nil {
		return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmountDevice.FindDeviceMountablePluginByName failed", err)
	}

	volumeDeviceUmounter, err := deviceMountableVolumePlugin.NewDeviceUnmounter()
	if err != nil {
		return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmountDevice.NewDeviceUmounter failed", err)
	}

	volumeDeviceMounter, err := deviceMountableVolumePlugin.NewDeviceMounter()
	if err != nil {
		return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmountDevice.NewDeviceMounter failed", err)
	}

	unmountDeviceFunc := func() (error, error) {
		//deviceMountPath := deviceToDetach.DeviceMountPath
		deviceMountPath, err :=
			volumeDeviceMounter.GetDeviceMountPath(deviceToDetach.VolumeSpec)
		if err != nil {
			// On failure other than "does not exist", return error. Caller will log and retry.
			if !strings.Contains(err.Error(), "does not exist") {
				return deviceToDetach.GenerateError("GetDeviceMountPath failed", err)
			}
			// If the mount path could not be found, don't fail the unmount, but instead log a warning and proceed,
			// using the value from deviceToDetach.DeviceMountPath, so that the device can be marked as unmounted
			deviceMountPath = deviceToDetach.DeviceMountPath
			klog.Warningf(deviceToDetach.GenerateMsg(fmt.Sprintf(
				"GetDeviceMountPath failed, but unmount operation will proceed using deviceMountPath=%s: %v", deviceMountPath, err), ""))
		}
		refs, err := deviceMountableVolumePlugin.GetDeviceMountRefs(deviceMountPath)

		if err != nil || util.HasMountRefs(deviceMountPath, refs) {
			if err == nil {
				err = fmt.Errorf("The device mount path %q is still mounted by other references %v", deviceMountPath, refs)
			}
			return deviceToDetach.GenerateError("GetDeviceMountRefs check failed", err)
		}
		// Execute unmount
		unmountDeviceErr := volumeDeviceUmounter.UnmountDevice(deviceMountPath)
		if unmountDeviceErr != nil {
			// On failure, return error. Caller will log and retry.
			return deviceToDetach.GenerateError("UnmountDevice failed", unmountDeviceErr)
		}
		// Before logging that UnmountDevice succeeded and moving on,
		// use hostutil.PathIsDevice to check if the path is a device,
		// if so use hostutil.DeviceOpened to check if the device is in use anywhere
		// else on the system. Retry if it returns true.
		deviceOpened, deviceOpenedErr := isDeviceOpened(deviceToDetach, hostutil)
		if deviceOpenedErr != nil {
			return nil, deviceOpenedErr
		}
		// The device is still in use elsewhere. Caller will log and retry.
		if deviceOpened {
			return deviceToDetach.GenerateError(
				"UnmountDevice failed",
				goerrors.New("the device is in use when it was no longer expected to be in use"))
		}

		klog.Infof(deviceToDetach.GenerateMsg("UnmountDevice succeeded", ""))

		// Update actual state of world
		markDeviceUnmountedErr := actualStateOfWorld.MarkDeviceAsUnmounted(
			deviceToDetach.VolumeName)
		if markDeviceUnmountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return deviceToDetach.GenerateError("MarkDeviceAsUnmounted failed", markDeviceUnmountedErr)
		}

		return nil, nil
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "unmount_device",
		OperationFunc:     unmountDeviceFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(deviceMountableVolumePlugin.GetPluginName(), deviceToDetach.VolumeSpec), "unmount_device"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil
}

// GenerateMapVolumeFunc marks volume as mounted based on following steps.
// If plugin is attachable, call WaitForAttach() and then mark the device
// as mounted. On next step, SetUpDevice is called without dependent of
// plugin type, but this method mainly is targeted for none attachable plugin.
// After setup is done, create symbolic links on both global map path and pod
// device map path. Once symbolic links are created, take fd lock by
// loopback for the device to avoid silent volume replacement. This lock
// will be released once no one uses the device.
// If all steps are completed, the volume is marked as mounted.
func (og *operationGenerator) GenerateMapVolumeFunc(
	waitForAttachTimeout time.Duration,
	volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {

	// Get block volume mapper plugin
	blockVolumePlugin, err :=
		og.volumePluginMgr.FindMapperPluginBySpec(volumeToMount.VolumeSpec)
	if err != nil {
		return volumetypes.GeneratedOperations{}, volumeToMount.GenerateErrorDetailed("MapVolume.FindMapperPluginBySpec failed", err)
	}

	if blockVolumePlugin == nil {
		return volumetypes.GeneratedOperations{}, volumeToMount.GenerateErrorDetailed("MapVolume.FindMapperPluginBySpec failed to find BlockVolumeMapper plugin. Volume plugin is nil.", nil)
	}

	affinityErr := checkNodeAffinity(og, volumeToMount)
	if affinityErr != nil {
		eventErr, detailedErr := volumeToMount.GenerateError("MapVolume.NodeAffinity check failed", affinityErr)
		og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FailedMountVolume, eventErr.Error())
		return volumetypes.GeneratedOperations{}, detailedErr
	}
	blockVolumeMapper, newMapperErr := blockVolumePlugin.NewBlockVolumeMapper(
		volumeToMount.VolumeSpec,
		volumeToMount.Pod,
		volume.VolumeOptions{})
	if newMapperErr != nil {
		eventErr, detailedErr := volumeToMount.GenerateError("MapVolume.NewBlockVolumeMapper initialization failed", newMapperErr)
		og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FailedMapVolume, eventErr.Error())
		return volumetypes.GeneratedOperations{}, detailedErr
	}

	// Get attacher, if possible
	attachableVolumePlugin, _ :=
		og.volumePluginMgr.FindAttachablePluginBySpec(volumeToMount.VolumeSpec)
	var volumeAttacher volume.Attacher
	if attachableVolumePlugin != nil {
		volumeAttacher, _ = attachableVolumePlugin.NewAttacher()
	}

	mapVolumeFunc := func() (simpleErr error, detailedErr error) {
		var devicePath string
		var stagingPath string
		// Set up global map path under the given plugin directory using symbolic link
		globalMapPath, err :=
			blockVolumeMapper.GetGlobalMapPath(volumeToMount.VolumeSpec)
		if err != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.GetGlobalMapPath failed", err)
		}
		if volumeAttacher != nil {
			// Wait for attachable volumes to finish attaching
			klog.Infof(volumeToMount.GenerateMsgDetailed("MapVolume.WaitForAttach entering", fmt.Sprintf("DevicePath %q", volumeToMount.DevicePath)))

			devicePath, err = volumeAttacher.WaitForAttach(
				volumeToMount.VolumeSpec, volumeToMount.DevicePath, volumeToMount.Pod, waitForAttachTimeout)
			if err != nil {
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("MapVolume.WaitForAttach failed", err)
			}

			klog.Infof(volumeToMount.GenerateMsgDetailed("MapVolume.WaitForAttach succeeded", fmt.Sprintf("DevicePath %q", devicePath)))

		}
		// Call SetUpDevice if blockVolumeMapper implements CustomBlockVolumeMapper
		if customBlockVolumeMapper, ok := blockVolumeMapper.(volume.CustomBlockVolumeMapper); ok {
			var mapErr error
			stagingPath, mapErr = customBlockVolumeMapper.SetUpDevice()
			if mapErr != nil {
				og.markDeviceErrorState(volumeToMount, devicePath, globalMapPath, mapErr, actualStateOfWorld)
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("MapVolume.SetUpDevice failed", mapErr)
			}
		}

		// Update actual state of world to reflect volume is globally mounted
		markedDevicePath := devicePath
		markDeviceMappedErr := actualStateOfWorld.MarkDeviceAsMounted(
			volumeToMount.VolumeName, markedDevicePath, globalMapPath)
		if markDeviceMappedErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.MarkDeviceAsMounted failed", markDeviceMappedErr)
		}

		markVolumeOpts := MarkVolumeOpts{
			PodName:             volumeToMount.PodName,
			PodUID:              volumeToMount.Pod.UID,
			VolumeName:          volumeToMount.VolumeName,
			BlockVolumeMapper:   blockVolumeMapper,
			OuterVolumeSpecName: volumeToMount.OuterVolumeSpecName,
			VolumeGidVolume:     volumeToMount.VolumeGidValue,
			VolumeSpec:          volumeToMount.VolumeSpec,
			VolumeMountState:    VolumeMounted,
		}

		// Call MapPodDevice if blockVolumeMapper implements CustomBlockVolumeMapper
		if customBlockVolumeMapper, ok := blockVolumeMapper.(volume.CustomBlockVolumeMapper); ok {
			// Execute driver specific map
			pluginDevicePath, mapErr := customBlockVolumeMapper.MapPodDevice()
			if mapErr != nil {
				// On failure, return error. Caller will log and retry.
				og.markVolumeErrorState(volumeToMount, markVolumeOpts, mapErr, actualStateOfWorld)
				return volumeToMount.GenerateError("MapVolume.MapPodDevice failed", mapErr)
			}

			// From now on, the volume is mapped. Mark it as uncertain on error,
			// so it is is unmapped when corresponding pod is deleted.
			defer func() {
				if simpleErr != nil {
					errText := simpleErr.Error()
					og.markVolumeErrorState(volumeToMount, markVolumeOpts, volumetypes.NewUncertainProgressError(errText), actualStateOfWorld)
				}
			}()

			// if pluginDevicePath is provided, assume attacher may not provide device
			// or attachment flow uses SetupDevice to get device path
			if len(pluginDevicePath) != 0 {
				devicePath = pluginDevicePath
			}
			if len(devicePath) == 0 {
				return volumeToMount.GenerateError("MapVolume failed", goerrors.New("Device path of the volume is empty"))
			}
		}

		// When kubelet is containerized, devicePath may be a symlink at a place unavailable to
		// kubelet, so evaluate it on the host and expect that it links to a device in /dev,
		// which will be available to containerized kubelet. If still it does not exist,
		// AttachFileDevice will fail. If kubelet is not containerized, eval it anyway.
		kvh, ok := og.GetVolumePluginMgr().Host.(volume.KubeletVolumeHost)
		if !ok {
			return volumeToMount.GenerateError("MapVolume type assertion error", fmt.Errorf("volume host does not implement KubeletVolumeHost interface"))
		}
		hu := kvh.GetHostUtil()
		devicePath, err = hu.EvalHostSymlinks(devicePath)
		if err != nil {
			return volumeToMount.GenerateError("MapVolume.EvalHostSymlinks failed", err)
		}

		// Update actual state of world with the devicePath again, if devicePath has changed from markedDevicePath
		// TODO: This can be improved after #82492 is merged and ASW has state.
		if markedDevicePath != devicePath {
			markDeviceMappedErr := actualStateOfWorld.MarkDeviceAsMounted(
				volumeToMount.VolumeName, devicePath, globalMapPath)
			if markDeviceMappedErr != nil {
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("MapVolume.MarkDeviceAsMounted failed", markDeviceMappedErr)
			}
		}

		// Execute common map
		volumeMapPath, volName := blockVolumeMapper.GetPodDeviceMapPath()
		mapErr := ioutil.MapBlockVolume(og.blkUtil, devicePath, globalMapPath, volumeMapPath, volName, volumeToMount.Pod.UID)
		if mapErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.MapBlockVolume failed", mapErr)
		}

		// Device mapping for global map path succeeded
		simpleMsg, detailedMsg := volumeToMount.GenerateMsg("MapVolume.MapPodDevice succeeded", fmt.Sprintf("globalMapPath %q", globalMapPath))
		verbosity := klog.Level(4)
		og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeNormal, kevents.SuccessfulMountVolume, simpleMsg)
		klog.V(verbosity).Infof(detailedMsg)

		// Device mapping for pod device map path succeeded
		simpleMsg, detailedMsg = volumeToMount.GenerateMsg("MapVolume.MapPodDevice succeeded", fmt.Sprintf("volumeMapPath %q", volumeMapPath))
		verbosity = klog.Level(1)
		og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeNormal, kevents.SuccessfulMountVolume, simpleMsg)
		klog.V(verbosity).Infof(detailedMsg)

		resizeOptions := volume.NodeResizeOptions{
			DevicePath:      devicePath,
			DeviceStagePath: stagingPath,
			CSIVolumePhase:  volume.CSIVolumePublished,
		}
		_, resizeError := og.nodeExpandVolume(volumeToMount, actualStateOfWorld, resizeOptions)
		if resizeError != nil {
			klog.Errorf("MapVolume.NodeExpandVolume failed with %v", resizeError)
			return volumeToMount.GenerateError("MapVolume.MarkVolumeAsMounted failed while expanding volume", resizeError)
		}

		markVolMountedErr := actualStateOfWorld.MarkVolumeAsMounted(markVolumeOpts)
		if markVolMountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.MarkVolumeAsMounted failed", markVolMountedErr)
		}

		return nil, nil
	}

	eventRecorderFunc := func(err *error) {
		if *err != nil {
			og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FailedMapVolume, (*err).Error())
		}
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "map_volume",
		OperationFunc:     mapVolumeFunc,
		EventRecorderFunc: eventRecorderFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(blockVolumePlugin.GetPluginName(), volumeToMount.VolumeSpec), "map_volume"),
	}, nil
}

// GenerateUnmapVolumeFunc marks volume as unmonuted based on following steps.
// Remove symbolic links from pod device map path dir and  global map path dir.
// Once those cleanups are done, remove pod device map path dir.
// If all steps are completed, the volume is marked as unmounted.
func (og *operationGenerator) GenerateUnmapVolumeFunc(
	volumeToUnmount MountedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {

	// Get block volume unmapper plugin
	blockVolumePlugin, err :=
		og.volumePluginMgr.FindMapperPluginByName(volumeToUnmount.PluginName)
	if err != nil {
		return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.FindMapperPluginByName failed", err)
	}
	if blockVolumePlugin == nil {
		return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.FindMapperPluginByName failed to find BlockVolumeMapper plugin. Volume plugin is nil.", nil)
	}
	blockVolumeUnmapper, newUnmapperErr := blockVolumePlugin.NewBlockVolumeUnmapper(
		volumeToUnmount.InnerVolumeSpecName, volumeToUnmount.PodUID)
	if newUnmapperErr != nil {
		return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.NewUnmapper failed", newUnmapperErr)
	}

	unmapVolumeFunc := func() (error, error) {
		// pods/{podUid}/volumeDevices/{escapeQualifiedPluginName}/{volumeName}
		podDeviceUnmapPath, volName := blockVolumeUnmapper.GetPodDeviceMapPath()
		// plugins/kubernetes.io/{PluginName}/volumeDevices/{volumePluginDependentPath}/{podUID}
		globalUnmapPath := volumeToUnmount.DeviceMountPath

		// Execute common unmap
		unmapErr := ioutil.UnmapBlockVolume(og.blkUtil, globalUnmapPath, podDeviceUnmapPath, volName, volumeToUnmount.PodUID)
		if unmapErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToUnmount.GenerateError("UnmapVolume.UnmapBlockVolume failed", unmapErr)
		}

		// Call UnmapPodDevice if blockVolumeUnmapper implements CustomBlockVolumeUnmapper
		if customBlockVolumeUnmapper, ok := blockVolumeUnmapper.(volume.CustomBlockVolumeUnmapper); ok {
			// Execute plugin specific unmap
			unmapErr = customBlockVolumeUnmapper.UnmapPodDevice()
			if unmapErr != nil {
				// On failure, return error. Caller will log and retry.
				return volumeToUnmount.GenerateError("UnmapVolume.UnmapPodDevice failed", unmapErr)
			}
		}

		klog.Infof(
			"UnmapVolume succeeded for volume %q (OuterVolumeSpecName: %q) pod %q (UID: %q). InnerVolumeSpecName %q. PluginName %q, VolumeGidValue %q",
			volumeToUnmount.VolumeName,
			volumeToUnmount.OuterVolumeSpecName,
			volumeToUnmount.PodName,
			volumeToUnmount.PodUID,
			volumeToUnmount.InnerVolumeSpecName,
			volumeToUnmount.PluginName,
			volumeToUnmount.VolumeGidValue)

		// Update actual state of world
		markVolUnmountedErr := actualStateOfWorld.MarkVolumeAsUnmounted(
			volumeToUnmount.PodName, volumeToUnmount.VolumeName)
		if markVolUnmountedErr != nil {
			// On failure, just log and exit
			klog.Errorf(volumeToUnmount.GenerateErrorDetailed("UnmapVolume.MarkVolumeAsUnmounted failed", markVolUnmountedErr).Error())
		}

		return nil, nil
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "unmap_volume",
		OperationFunc:     unmapVolumeFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(blockVolumePlugin.GetPluginName(), volumeToUnmount.VolumeSpec), "unmap_volume"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil
}

// GenerateUnmapDeviceFunc marks device as unmounted based on following steps.
// Check under globalMapPath dir if there isn't pod's symbolic links in it.
// If symbolic link isn't there, the device isn't referenced from Pods.
// Call plugin TearDownDevice to clean-up device connection, stored data under
// globalMapPath, these operations depend on plugin implementation.
// Once TearDownDevice is completed, remove globalMapPath dir.
// After globalMapPath is removed, fd lock by loopback for the device can
// be released safely because no one can consume the device at this point.
// At last, device open status will be checked just in case.
// If all steps are completed, the device is marked as unmounted.
func (og *operationGenerator) GenerateUnmapDeviceFunc(
	deviceToDetach AttachedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	hostutil hostutil.HostUtils) (volumetypes.GeneratedOperations, error) {

	blockVolumePlugin, err :=
		og.volumePluginMgr.FindMapperPluginByName(deviceToDetach.PluginName)
	if err != nil {
		return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmapDevice.FindMapperPluginByName failed", err)
	}

	if blockVolumePlugin == nil {
		return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmapDevice.FindMapperPluginByName failed to find BlockVolumeMapper plugin. Volume plugin is nil.", nil)
	}

	blockVolumeUnmapper, newUnmapperErr := blockVolumePlugin.NewBlockVolumeUnmapper(
		deviceToDetach.VolumeSpec.Name(),
		"" /* podUID */)
	if newUnmapperErr != nil {
		return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmapDevice.NewUnmapper failed", newUnmapperErr)
	}

	unmapDeviceFunc := func() (error, error) {
		// Search under globalMapPath dir if all symbolic links from pods have been removed already.
		// If symbolic links are there, pods may still refer the volume.
		globalMapPath := deviceToDetach.DeviceMountPath
		refs, err := og.blkUtil.GetDeviceBindMountRefs(deviceToDetach.DevicePath, globalMapPath)
		if err != nil {
			if os.IsNotExist(err) {
				// Looks like SetupDevice did not complete. Fall through to TearDownDevice and mark the device as unmounted.
				refs = nil
			} else {
				return deviceToDetach.GenerateError("UnmapDevice.GetDeviceBindMountRefs check failed", err)
			}
		}
		if len(refs) > 0 {
			err = fmt.Errorf("The device %q is still referenced from other Pods %v", globalMapPath, refs)
			return deviceToDetach.GenerateError("UnmapDevice failed", err)
		}

		// Call TearDownDevice if blockVolumeUnmapper implements CustomBlockVolumeUnmapper
		if customBlockVolumeUnmapper, ok := blockVolumeUnmapper.(volume.CustomBlockVolumeUnmapper); ok {
			// Execute tear down device
			unmapErr := customBlockVolumeUnmapper.TearDownDevice(globalMapPath, deviceToDetach.DevicePath)
			if unmapErr != nil {
				// On failure, return error. Caller will log and retry.
				return deviceToDetach.GenerateError("UnmapDevice.TearDownDevice failed", unmapErr)
			}
		}

		// Plugin finished TearDownDevice(). Now globalMapPath dir and plugin's stored data
		// on the dir are unnecessary, clean up it.
		removeMapPathErr := og.blkUtil.RemoveMapPath(globalMapPath)
		if removeMapPathErr != nil {
			// On failure, return error. Caller will log and retry.
			return deviceToDetach.GenerateError("UnmapDevice.RemoveMapPath failed", removeMapPathErr)
		}

		// Before logging that UnmapDevice succeeded and moving on,
		// use hostutil.PathIsDevice to check if the path is a device,
		// if so use hostutil.DeviceOpened to check if the device is in use anywhere
		// else on the system. Retry if it returns true.
		deviceOpened, deviceOpenedErr := isDeviceOpened(deviceToDetach, hostutil)
		if deviceOpenedErr != nil {
			return nil, deviceOpenedErr
		}
		// The device is still in use elsewhere. Caller will log and retry.
		if deviceOpened {
			return deviceToDetach.GenerateError(
				"UnmapDevice failed",
				fmt.Errorf("the device is in use when it was no longer expected to be in use"))
		}

		klog.Infof(deviceToDetach.GenerateMsgDetailed("UnmapDevice succeeded", ""))

		// Update actual state of world
		markDeviceUnmountedErr := actualStateOfWorld.MarkDeviceAsUnmounted(
			deviceToDetach.VolumeName)
		if markDeviceUnmountedErr != nil {
			// On failure, return error. Caller will log and retry.
			return deviceToDetach.GenerateError("MarkDeviceAsUnmounted failed", markDeviceUnmountedErr)
		}

		return nil, nil
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "unmap_device",
		OperationFunc:     unmapDeviceFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(blockVolumePlugin.GetPluginName(), deviceToDetach.VolumeSpec), "unmap_device"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil
}

func (og *operationGenerator) GenerateVerifyControllerAttachedVolumeFunc(
	volumeToMount VolumeToMount,
	nodeName types.NodeName,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	volumePlugin, err :=
		og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
	if err != nil || volumePlugin == nil {
		return volumetypes.GeneratedOperations{}, volumeToMount.GenerateErrorDetailed("VerifyControllerAttachedVolume.FindPluginBySpec failed", err)
	}

	verifyControllerAttachedVolumeFunc := func() (error, error) {
		if !volumeToMount.PluginIsAttachable {
			// If the volume does not implement the attacher interface, it is
			// assumed to be attached and the actual state of the world is
			// updated accordingly.

			addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
				volumeToMount.VolumeName, volumeToMount.VolumeSpec, nodeName, "" /* devicePath */)
			if addVolumeNodeErr != nil {
				// On failure, return error. Caller will log and retry.
				return volumeToMount.GenerateError("VerifyControllerAttachedVolume.MarkVolumeAsAttachedByUniqueVolumeName failed", addVolumeNodeErr)
			}

			return nil, nil
		}

		if !volumeToMount.ReportedInUse {
			// If the given volume has not yet been added to the list of
			// VolumesInUse in the node's volume status, do not proceed, return
			// error. Caller will log and retry. The node status is updated
			// periodically by kubelet, so it may take as much as 10 seconds
			// before this clears.
			// Issue #28141 to enable on demand status updates.
			return volumeToMount.GenerateError("Volume has not been added to the list of VolumesInUse in the node's volume status", nil)
		}

		// Fetch current node object
		node, fetchErr := og.kubeClient.CoreV1().Nodes().Get(context.TODO(), string(nodeName), metav1.GetOptions{})
		if fetchErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("VerifyControllerAttachedVolume failed fetching node from API server", fetchErr)
		}

		if node == nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError(
				"VerifyControllerAttachedVolume failed",
				fmt.Errorf("Node object retrieved from API server is nil"))
		}

		for _, attachedVolume := range node.Status.VolumesAttached {
			if attachedVolume.Name == volumeToMount.VolumeName {
				addVolumeNodeErr := actualStateOfWorld.MarkVolumeAsAttached(
					v1.UniqueVolumeName(""), volumeToMount.VolumeSpec, nodeName, attachedVolume.DevicePath)
				klog.Infof(volumeToMount.GenerateMsgDetailed("Controller attach succeeded", fmt.Sprintf("device path: %q", attachedVolume.DevicePath)))
				if addVolumeNodeErr != nil {
					// On failure, return error. Caller will log and retry.
					return volumeToMount.GenerateError("VerifyControllerAttachedVolume.MarkVolumeAsAttached failed", addVolumeNodeErr)
				}
				return nil, nil
			}
		}

		// Volume not attached, return error. Caller will log and retry.
		return volumeToMount.GenerateError("Volume not attached according to node status", nil)
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "verify_controller_attached_volume",
		OperationFunc:     verifyControllerAttachedVolumeFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(volumePlugin.GetPluginName(), volumeToMount.VolumeSpec), "verify_controller_attached_volume"),
		EventRecorderFunc: nil, // nil because we do not want to generate event on error
	}, nil

}

func (og *operationGenerator) verifyVolumeIsSafeToDetach(
	volumeToDetach AttachedVolume) error {
	// Fetch current node object
	node, fetchErr := og.kubeClient.CoreV1().Nodes().Get(context.TODO(), string(volumeToDetach.NodeName), metav1.GetOptions{})
	if fetchErr != nil {
		if errors.IsNotFound(fetchErr) {
			klog.Warningf(volumeToDetach.GenerateMsgDetailed("Node not found on API server. DetachVolume will skip safe to detach check", ""))
			return nil
		}

		// On failure, return error. Caller will log and retry.
		return volumeToDetach.GenerateErrorDetailed("DetachVolume failed fetching node from API server", fetchErr)
	}

	if node == nil {
		// On failure, return error. Caller will log and retry.
		return volumeToDetach.GenerateErrorDetailed(
			"DetachVolume failed fetching node from API server",
			fmt.Errorf("node object retrieved from API server is nil"))
	}

	for _, inUseVolume := range node.Status.VolumesInUse {
		if inUseVolume == volumeToDetach.VolumeName {
			return volumeToDetach.GenerateErrorDetailed(
				"DetachVolume failed",
				fmt.Errorf("volume is still in use by node, according to Node status"))
		}
	}

	// Volume is not marked as in use by node
	klog.Infof(volumeToDetach.GenerateMsgDetailed("Verified volume is safe to detach", ""))
	return nil
}

func (og *operationGenerator) GenerateExpandVolumeFunc(
	pvc *v1.PersistentVolumeClaim,
	pv *v1.PersistentVolume) (volumetypes.GeneratedOperations, error) {

	volumeSpec := volume.NewSpecFromPersistentVolume(pv, false)

	volumePlugin, err := og.volumePluginMgr.FindExpandablePluginBySpec(volumeSpec)
	if err != nil {
		return volumetypes.GeneratedOperations{}, fmt.Errorf("Error finding plugin for expanding volume: %q with error %v", util.GetPersistentVolumeClaimQualifiedName(pvc), err)
	}

	if volumePlugin == nil {
		return volumetypes.GeneratedOperations{}, fmt.Errorf("Can not find plugin for expanding volume: %q", util.GetPersistentVolumeClaimQualifiedName(pvc))
	}

	expandVolumeFunc := func() (error, error) {
		newSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		statusSize := pvc.Status.Capacity[v1.ResourceStorage]
		pvSize := pv.Spec.Capacity[v1.ResourceStorage]
		if pvSize.Cmp(newSize) < 0 {
			updatedSize, expandErr := volumePlugin.ExpandVolumeDevice(
				volumeSpec,
				newSize,
				statusSize)
			if expandErr != nil {
				detailedErr := fmt.Errorf("error expanding volume %q of plugin %q: %v", util.GetPersistentVolumeClaimQualifiedName(pvc), volumePlugin.GetPluginName(), expandErr)
				return detailedErr, detailedErr
			}

			klog.Infof("ExpandVolume succeeded for volume %s", util.GetPersistentVolumeClaimQualifiedName(pvc))

			newSize = updatedSize
			// k8s doesn't have transactions, we can't guarantee that after updating PV - updating PVC will be
			// successful, that is why all PVCs for which pvc.Spec.Size > pvc.Status.Size must be reprocessed
			// until they reflect user requested size in pvc.Status.Size
			updateErr := util.UpdatePVSize(pv, newSize, og.kubeClient)
			if updateErr != nil {
				detailedErr := fmt.Errorf("Error updating PV spec capacity for volume %q with : %v", util.GetPersistentVolumeClaimQualifiedName(pvc), updateErr)
				return detailedErr, detailedErr
			}

			klog.Infof("ExpandVolume.UpdatePV succeeded for volume %s", util.GetPersistentVolumeClaimQualifiedName(pvc))
		}

		fsVolume, _ := util.CheckVolumeModeFilesystem(volumeSpec)
		// No Cloudprovider resize needed, lets mark resizing as done
		// Rest of the volume expand controller code will assume PVC as *not* resized until pvc.Status.Size
		// reflects user requested size.
		if !volumePlugin.RequiresFSResize() || !fsVolume {
			klog.V(4).Infof("Controller resizing done for PVC %s", util.GetPersistentVolumeClaimQualifiedName(pvc))
			err := util.MarkResizeFinished(pvc, newSize, og.kubeClient)
			if err != nil {
				detailedErr := fmt.Errorf("Error marking pvc %s as resized : %v", util.GetPersistentVolumeClaimQualifiedName(pvc), err)
				return detailedErr, detailedErr
			}
			successMsg := fmt.Sprintf("ExpandVolume succeeded for volume %s", util.GetPersistentVolumeClaimQualifiedName(pvc))
			og.recorder.Eventf(pvc, v1.EventTypeNormal, kevents.VolumeResizeSuccess, successMsg)
		} else {
			err := util.MarkForFSResize(pvc, og.kubeClient)
			if err != nil {
				detailedErr := fmt.Errorf("Error updating pvc %s condition for fs resize : %v", util.GetPersistentVolumeClaimQualifiedName(pvc), err)
				klog.Warning(detailedErr)
				return nil, nil
			}
		}
		return nil, nil
	}

	eventRecorderFunc := func(err *error) {
		if *err != nil {
			og.recorder.Eventf(pvc, v1.EventTypeWarning, kevents.VolumeResizeFailed, (*err).Error())
		}
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "expand_volume",
		OperationFunc:     expandVolumeFunc,
		EventRecorderFunc: eventRecorderFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(volumePlugin.GetPluginName(), volumeSpec), "expand_volume"),
	}, nil
}

func (og *operationGenerator) GenerateExpandInUseVolumeFunc(
	volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {

	volumePlugin, err :=
		og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
	if err != nil || volumePlugin == nil {
		return volumetypes.GeneratedOperations{}, volumeToMount.GenerateErrorDetailed("NodeExpandVolume.FindPluginBySpec failed", err)
	}

	fsResizeFunc := func() (error, error) {
		var resizeDone bool
		var simpleErr, detailedErr error
		resizeOptions := volume.NodeResizeOptions{
			VolumeSpec: volumeToMount.VolumeSpec,
			DevicePath: volumeToMount.DevicePath,
		}
		fsVolume, err := util.CheckVolumeModeFilesystem(volumeToMount.VolumeSpec)
		if err != nil {
			return volumeToMount.GenerateError("NodeExpandvolume.CheckVolumeModeFilesystem failed", err)
		}

		if fsVolume {
			volumeMounter, newMounterErr := volumePlugin.NewMounter(
				volumeToMount.VolumeSpec,
				volumeToMount.Pod,
				volume.VolumeOptions{})
			if newMounterErr != nil {
				return volumeToMount.GenerateError("NodeExpandVolume.NewMounter initialization failed", newMounterErr)
			}

			resizeOptions.DeviceMountPath = volumeMounter.GetPath()

			deviceMountableVolumePlugin, _ := og.volumePluginMgr.FindDeviceMountablePluginBySpec(volumeToMount.VolumeSpec)
			var volumeDeviceMounter volume.DeviceMounter
			if deviceMountableVolumePlugin != nil {
				volumeDeviceMounter, _ = deviceMountableVolumePlugin.NewDeviceMounter()
			}

			if volumeDeviceMounter != nil {
				deviceStagePath, err := volumeDeviceMounter.GetDeviceMountPath(volumeToMount.VolumeSpec)
				if err != nil {
					return volumeToMount.GenerateError("NodeExpandVolume.GetDeviceMountPath failed", err)
				}
				resizeOptions.DeviceStagePath = deviceStagePath
			}
		} else {
			// Get block volume mapper plugin
			blockVolumePlugin, err :=
				og.volumePluginMgr.FindMapperPluginBySpec(volumeToMount.VolumeSpec)
			if err != nil {
				return volumeToMount.GenerateError("MapVolume.FindMapperPluginBySpec failed", err)
			}

			if blockVolumePlugin == nil {
				return volumeToMount.GenerateError("MapVolume.FindMapperPluginBySpec failed to find BlockVolumeMapper plugin. Volume plugin is nil.", nil)
			}

			blockVolumeMapper, newMapperErr := blockVolumePlugin.NewBlockVolumeMapper(
				volumeToMount.VolumeSpec,
				volumeToMount.Pod,
				volume.VolumeOptions{})
			if newMapperErr != nil {
				return volumeToMount.GenerateError("MapVolume.NewBlockVolumeMapper initialization failed", newMapperErr)
			}

			// if plugin supports custom mappers lets add DeviceStagePath
			if customBlockVolumeMapper, ok := blockVolumeMapper.(volume.CustomBlockVolumeMapper); ok {
				resizeOptions.DeviceStagePath = customBlockVolumeMapper.GetStagingPath()
			}
		}

		// if we are doing online expansion then volume is already published
		resizeOptions.CSIVolumePhase = volume.CSIVolumePublished
		resizeDone, simpleErr, detailedErr = og.doOnlineExpansion(volumeToMount, actualStateOfWorld, resizeOptions)
		if simpleErr != nil || detailedErr != nil {
			return simpleErr, detailedErr
		}
		if resizeDone {
			return nil, nil
		}
		// This is a placeholder error - we should NEVER reach here.
		err = fmt.Errorf("volume resizing failed for unknown reason")
		return volumeToMount.GenerateError("NodeExpandVolume.NodeExpandVolume failed to resize volume", err)
	}

	eventRecorderFunc := func(err *error) {
		if *err != nil {
			og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.VolumeResizeFailed, (*err).Error())
		}
	}

	return volumetypes.GeneratedOperations{
		OperationName:     "volume_fs_resize",
		OperationFunc:     fsResizeFunc,
		EventRecorderFunc: eventRecorderFunc,
		CompleteFunc:      util.OperationCompleteHook(util.GetFullQualifiedPluginNameForVolume(volumePlugin.GetPluginName(), volumeToMount.VolumeSpec), "volume_fs_resize"),
	}, nil
}

func (og *operationGenerator) doOnlineExpansion(volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	resizeOptions volume.NodeResizeOptions) (bool, error, error) {

	resizeDone, err := og.nodeExpandVolume(volumeToMount, actualStateOfWorld, resizeOptions)
	if err != nil {
		e1, e2 := volumeToMount.GenerateError("NodeExpandVolume.NodeExpandVolume failed", err)
		klog.Errorf(e2.Error())
		return false, e1, e2
	}
	if resizeDone {
		markFSResizedErr := actualStateOfWorld.MarkVolumeAsResized(volumeToMount.PodName, volumeToMount.VolumeName)
		if markFSResizedErr != nil {
			// On failure, return error. Caller will log and retry.
			e1, e2 := volumeToMount.GenerateError("NodeExpandVolume.MarkVolumeAsResized failed", markFSResizedErr)
			return false, e1, e2
		}
		return true, nil, nil
	}
	return false, nil, nil
}

func (og *operationGenerator) nodeExpandVolume(
	volumeToMount VolumeToMount,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	rsOpts volume.NodeResizeOptions) (bool, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		klog.V(4).Infof("Resizing is not enabled for this volume %s", volumeToMount.VolumeName)
		return true, nil
	}

	if volumeToMount.VolumeSpec != nil &&
		volumeToMount.VolumeSpec.InlineVolumeSpecForCSIMigration {
		klog.V(4).Infof("This volume %s is a migrated inline volume and is not resizable", volumeToMount.VolumeName)
		return true, nil
	}

	// Get expander, if possible
	expandableVolumePlugin, _ :=
		og.volumePluginMgr.FindNodeExpandablePluginBySpec(volumeToMount.VolumeSpec)

	if expandableVolumePlugin != nil &&
		expandableVolumePlugin.RequiresFSResize() &&
		volumeToMount.VolumeSpec.PersistentVolume != nil {
		pv := volumeToMount.VolumeSpec.PersistentVolume
		pvc, err := og.kubeClient.CoreV1().PersistentVolumeClaims(pv.Spec.ClaimRef.Namespace).Get(context.TODO(), pv.Spec.ClaimRef.Name, metav1.GetOptions{})
		if err != nil {
			// Return error rather than leave the file system un-resized, caller will log and retry
			return false, fmt.Errorf("MountVolume.NodeExpandVolume get PVC failed : %v", err)
		}

		pvcStatusCap := pvc.Status.Capacity[v1.ResourceStorage]
		pvSpecCap := pv.Spec.Capacity[v1.ResourceStorage]
		if pvcStatusCap.Cmp(pvSpecCap) < 0 {
			// File system resize was requested, proceed
			klog.V(4).Infof(volumeToMount.GenerateMsgDetailed("MountVolume.NodeExpandVolume entering", fmt.Sprintf("DevicePath %q", volumeToMount.DevicePath)))

			if volumeToMount.VolumeSpec.ReadOnly {
				simpleMsg, detailedMsg := volumeToMount.GenerateMsg("MountVolume.NodeExpandVolume failed", "requested read-only file system")
				klog.Warningf(detailedMsg)
				og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FileSystemResizeFailed, simpleMsg)
				og.recorder.Eventf(pvc, v1.EventTypeWarning, kevents.FileSystemResizeFailed, simpleMsg)
				return true, nil
			}
			rsOpts.VolumeSpec = volumeToMount.VolumeSpec
			rsOpts.NewSize = pvSpecCap
			rsOpts.OldSize = pvcStatusCap
			resizeDone, resizeErr := expandableVolumePlugin.NodeExpand(rsOpts)
			if resizeErr != nil {
				// if driver returned FailedPrecondition error that means
				// volume expansion should not be retried on this node but
				// expansion operation should not block mounting
				if volumetypes.IsFailedPreconditionError(resizeErr) {
					actualStateOfWorld.MarkForInUseExpansionError(volumeToMount.VolumeName)
					klog.Errorf(volumeToMount.GenerateErrorDetailed("MountVolume.NodeExapndVolume failed with %v", resizeErr).Error())
					return true, nil
				}
				return false, resizeErr
			}
			// Volume resizing is not done but it did not error out. This could happen if a CSI volume
			// does not have node stage_unstage capability but was asked to resize the volume before
			// node publish. In which case - we must retry resizing after node publish.
			if !resizeDone {
				return false, nil
			}
			simpleMsg, detailedMsg := volumeToMount.GenerateMsg("MountVolume.NodeExpandVolume succeeded", "")
			og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeNormal, kevents.FileSystemResizeSuccess, simpleMsg)
			og.recorder.Eventf(pvc, v1.EventTypeNormal, kevents.FileSystemResizeSuccess, simpleMsg)
			klog.Infof(detailedMsg)
			// File system resize succeeded, now update the PVC's Capacity to match the PV's
			err = util.MarkFSResizeFinished(pvc, pvSpecCap, og.kubeClient)
			if err != nil {
				// On retry, NodeExpandVolume will be called again but do nothing
				return false, fmt.Errorf("MountVolume.NodeExpandVolume update PVC status failed : %v", err)
			}
			return true, nil
		}
	}
	return true, nil
}

func checkMountOptionSupport(og *operationGenerator, volumeToMount VolumeToMount, plugin volume.VolumePlugin) error {
	mountOptions := util.MountOptionFromSpec(volumeToMount.VolumeSpec)

	if len(mountOptions) > 0 && !plugin.SupportsMountOption() {
		return fmt.Errorf("Mount options are not supported for this volume type")
	}
	return nil
}

// checkNodeAffinity looks at the PV node affinity, and checks if the node has the same corresponding labels
// This ensures that we don't mount a volume that doesn't belong to this node
func checkNodeAffinity(og *operationGenerator, volumeToMount VolumeToMount) error {
	pv := volumeToMount.VolumeSpec.PersistentVolume
	if pv != nil {
		nodeLabels, err := og.volumePluginMgr.Host.GetNodeLabels()
		if err != nil {
			return err
		}
		err = util.CheckNodeAffinity(pv, nodeLabels)
		if err != nil {
			return err
		}
	}
	return nil
}

// isDeviceOpened checks the device status if the device is in use anywhere else on the system
func isDeviceOpened(deviceToDetach AttachedVolume, hostUtil hostutil.HostUtils) (bool, error) {
	isDevicePath, devicePathErr := hostUtil.PathIsDevice(deviceToDetach.DevicePath)
	var deviceOpened bool
	var deviceOpenedErr error
	if !isDevicePath && devicePathErr == nil ||
		(devicePathErr != nil && strings.Contains(devicePathErr.Error(), "does not exist")) {
		// not a device path or path doesn't exist
		//TODO: refer to #36092
		klog.V(3).Infof("The path isn't device path or doesn't exist. Skip checking device path: %s", deviceToDetach.DevicePath)
		deviceOpened = false
	} else if devicePathErr != nil {
		return false, deviceToDetach.GenerateErrorDetailed("PathIsDevice failed", devicePathErr)
	} else {
		deviceOpened, deviceOpenedErr = hostUtil.DeviceOpened(deviceToDetach.DevicePath)
		if deviceOpenedErr != nil {
			return false, deviceToDetach.GenerateErrorDetailed("DeviceOpened failed", deviceOpenedErr)
		}
	}
	return deviceOpened, nil
}

// findDetachablePluginBySpec is a variant of VolumePluginMgr.FindAttachablePluginByName() function.
// The difference is that it bypass the CanAttach() check for CSI plugin, i.e. it assumes all CSI plugin supports detach.
// The intention here is that a CSI plugin volume can end up in an Uncertain state,  so that a detach
// operation will help it to detach no matter it actually has the ability to attach/detach.
func findDetachablePluginBySpec(spec *volume.Spec, pm *volume.VolumePluginMgr) (volume.AttachableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if attachableVolumePlugin, ok := volumePlugin.(volume.AttachableVolumePlugin); ok {
		if attachableVolumePlugin.GetPluginName() == "kubernetes.io/csi" {
			return attachableVolumePlugin, nil
		}
		if canAttach, err := attachableVolumePlugin.CanAttach(spec); err != nil {
			return nil, err
		} else if canAttach {
			return attachableVolumePlugin, nil
		}
	}
	return nil, nil
}
