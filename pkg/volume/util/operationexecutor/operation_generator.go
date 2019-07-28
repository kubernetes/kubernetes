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
	goerrors "errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	volerr "k8s.io/cloud-provider/volume/errors"
	csilib "k8s.io/csi-translation-lib"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	kevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	"k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

const (
	unknownVolumePlugin           string = "UnknownVolumePlugin"
	unknownAttachableVolumePlugin string = "UnknownAttachableVolumePlugin"
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

	// blkUtil provides volume path related operations for block volume
	blkUtil volumepathhandler.BlockVolumePathHandler
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
	GenerateUnmountDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter mount.HostUtils) (volumetypes.GeneratedOperations, error)

	// Generates the function needed to check if the attach_detach controller has attached the volume plugin
	GenerateVerifyControllerAttachedVolumeFunc(volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the MapVolume function needed to perform the map of a volume plugin
	GenerateMapVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the UnmapVolume function needed to perform the unmap of a volume plugin
	GenerateUnmapVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error)

	// Generates the UnmapDevice function needed to perform the unmap of a device
	GenerateUnmapDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter mount.HostUtils) (volumetypes.GeneratedOperations, error)

	// GetVolumePluginMgr returns volume plugin manager
	GetVolumePluginMgr() *volume.VolumePluginMgr

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

		// Migration: Must also check the Node since Attach would have been done with in-tree if node is not using Migration
		nu, err := nodeUsingCSIPlugin(og, volumeAttached.VolumeSpec, nodeName)
		if err != nil {
			klog.Errorf(volumeAttached.GenerateErrorDetailed("VolumesAreAttached.NodeUsingCSIPlugin failed", err).Error())
			continue
		}

		var volumePlugin volume.VolumePlugin
		if useCSIPlugin(og.volumePluginMgr, volumeAttached.VolumeSpec) && nu {
			// The volume represented by this spec is CSI and thus should be migrated
			volumePlugin, err = og.volumePluginMgr.FindPluginByName(csi.CSIPluginName)
			if err != nil || volumePlugin == nil {
				klog.Errorf(volumeAttached.GenerateErrorDetailed("VolumesAreAttached.FindPluginByName failed", err).Error())
				continue
			}

			csiSpec, err := translateSpec(volumeAttached.VolumeSpec)
			if err != nil {
				klog.Errorf(volumeAttached.GenerateErrorDetailed("VolumesAreAttached.TranslateSpec failed", err).Error())
				continue
			}
			volumeAttached.VolumeSpec = csiSpec
		} else {
			volumePlugin, err =
				og.volumePluginMgr.FindPluginBySpec(volumeAttached.VolumeSpec)
			if err != nil || volumePlugin == nil {
				klog.Errorf(volumeAttached.GenerateErrorDetailed("VolumesAreAttached.FindPluginBySpec failed", err).Error())
				continue
			}
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
	originalSpec := volumeToAttach.VolumeSpec
	attachVolumeFunc := func() (error, error) {
		var attachableVolumePlugin volume.AttachableVolumePlugin

		nu, err := nodeUsingCSIPlugin(og, volumeToAttach.VolumeSpec, volumeToAttach.NodeName)
		if err != nil {
			return volumeToAttach.GenerateError("AttachVolume.NodeUsingCSIPlugin failed", err)
		}

		// useCSIPlugin will check both CSIMigration and the plugin specific feature gates
		if useCSIPlugin(og.volumePluginMgr, volumeToAttach.VolumeSpec) && nu {
			// The volume represented by this spec is CSI and thus should be migrated
			attachableVolumePlugin, err = og.volumePluginMgr.FindAttachablePluginByName(csi.CSIPluginName)
			if err != nil || attachableVolumePlugin == nil {
				return volumeToAttach.GenerateError("AttachVolume.FindAttachablePluginByName failed", err)
			}

			csiSpec, err := translateSpec(volumeToAttach.VolumeSpec)
			if err != nil {
				return volumeToAttach.GenerateError("AttachVolume.TranslateSpec failed", err)
			}
			volumeToAttach.VolumeSpec = csiSpec
		} else {
			attachableVolumePlugin, err =
				og.volumePluginMgr.FindAttachablePluginBySpec(volumeToAttach.VolumeSpec)
			if err != nil || attachableVolumePlugin == nil {
				return volumeToAttach.GenerateError("AttachVolume.FindAttachablePluginBySpec failed", err)
			}
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
				v1.UniqueVolumeName(""),
				originalSpec,
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
			v1.UniqueVolumeName(""), originalSpec, volumeToAttach.NodeName, devicePath)
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
	// TODO(dyzz) Ignoring this error means that if the plugin is migrated and
	// any transient error is encountered (API unavailable, driver not installed)
	// the operation will have it's metric registered with the in-tree plugin instead
	// of the CSI Driver we migrated to. Fixing this requires a larger refactor that
	// involves determining the plugin_name for the metric generating "CompleteFunc"
	// during the actual "OperationFunc" and not during this generation function

	nu, err := nodeUsingCSIPlugin(og, volumeToAttach.VolumeSpec, volumeToAttach.NodeName)
	if err != nil {
		klog.Errorf("GenerateAttachVolumeFunc failed to check if node is using CSI Plugin, metric for this operation may be inaccurate: %v", err)
	}

	// Need to translate the spec here if the plugin is migrated so that the metrics
	// emitted show the correct (migrated) plugin
	if useCSIPlugin(og.volumePluginMgr, volumeToAttach.VolumeSpec) && nu {
		csiSpec, err := translateSpec(volumeToAttach.VolumeSpec)
		if err == nil {
			volumeToAttach.VolumeSpec = csiSpec
		}
		// If we have an error here we ignore it, the metric emitted will then be for the
		// in-tree plugin. This error case(skipped one) will also trigger an error
		// while the generated function is executed. And those errors will be handled during the execution of the generated
		// function with a back off policy.
	}
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

func (og *operationGenerator) GenerateDetachVolumeFunc(
	volumeToDetach AttachedVolume,
	verifySafeToDetach bool,
	actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	var volumeName string
	var attachableVolumePlugin volume.AttachableVolumePlugin
	var pluginName string
	var err error

	if volumeToDetach.VolumeSpec != nil {
		// Get attacher plugin
		nu, err := nodeUsingCSIPlugin(og, volumeToDetach.VolumeSpec, volumeToDetach.NodeName)
		if err != nil {
			return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.NodeUsingCSIPlugin failed", err)
		}

		// useCSIPlugin will check both CSIMigration and the plugin specific feature gate
		if useCSIPlugin(og.volumePluginMgr, volumeToDetach.VolumeSpec) && nu {
			// The volume represented by this spec is CSI and thus should be migrated
			attachableVolumePlugin, err = og.volumePluginMgr.FindAttachablePluginByName(csi.CSIPluginName)
			if err != nil || attachableVolumePlugin == nil {
				return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.FindAttachablePluginBySpec failed", err)
			}

			csiSpec, err := translateSpec(volumeToDetach.VolumeSpec)
			if err != nil {
				return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.TranslateSpec failed", err)
			}

			volumeToDetach.VolumeSpec = csiSpec
		} else {
			attachableVolumePlugin, err =
				og.volumePluginMgr.FindAttachablePluginBySpec(volumeToDetach.VolumeSpec)
			if err != nil || attachableVolumePlugin == nil {
				return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.FindAttachablePluginBySpec failed", err)
			}
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

		// TODO(dyzz): This case can't distinguish between PV and In-line which is necessary because
		// if it was PV it may have been migrated, but the same plugin with in-line may not have been.
		// Suggestions welcome...
		if csilib.IsMigratableIntreePluginByName(pluginName) && utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) {
			// The volume represented by this spec is CSI and thus should be migrated
			attachableVolumePlugin, err = og.volumePluginMgr.FindAttachablePluginByName(csi.CSIPluginName)
			if err != nil || attachableVolumePlugin == nil {
				return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("AttachVolume.FindAttachablePluginBySpec failed", err)
			}
			// volumeToDetach.VolumeName here is always the in-tree volume name
			// therefore a workaround is required. volumeToDetach.DevicePath
			// is the attachID which happens to be what volumeName is needed for in Detach.
			// Therefore we set volumeName to the attachID. And CSI Detach can detect and use that.
			volumeName = volumeToDetach.DevicePath
		} else {
			attachableVolumePlugin, err = og.volumePluginMgr.FindAttachablePluginByName(pluginName)
			if err != nil || attachableVolumePlugin == nil {
				return volumetypes.GeneratedOperations{}, volumeToDetach.GenerateErrorDetailed("DetachVolume.FindAttachablePluginByName failed", err)
			}
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
	// Get mounter plugin
	originalSpec := volumeToMount.VolumeSpec
	volumePluginName := unknownVolumePlugin
	// Need to translate the spec here if the plugin is migrated so that the metrics
	// emitted show the correct (migrated) plugin
	if useCSIPlugin(og.volumePluginMgr, volumeToMount.VolumeSpec) {
		csiSpec, err := translateSpec(volumeToMount.VolumeSpec)
		if err == nil {
			volumeToMount.VolumeSpec = csiSpec
		}
		// If we have an error here we ignore it, the metric emitted will then be for the
		// in-tree plugin. This error case(skipped one) will also trigger an error
		// while the generated function is executed. And those errors will be handled during the execution of the generated
		// function with a back off policy.
	}
	volumePlugin, err :=
		og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
	if err == nil && volumePlugin != nil {
		volumePluginName = volumePlugin.GetPluginName()
	}

	mountVolumeFunc := func() (error, error) {

		// Get mounter plugin
		if useCSIPlugin(og.volumePluginMgr, volumeToMount.VolumeSpec) {
			csiSpec, err := translateSpec(volumeToMount.VolumeSpec)
			if err != nil {
				return volumeToMount.GenerateError("MountVolume.TranslateSpec failed", err)
			}
			volumeToMount.VolumeSpec = csiSpec
		}

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
		if volumeToMount.Pod.Spec.SecurityContext != nil &&
			volumeToMount.Pod.Spec.SecurityContext.FSGroup != nil {
			fsGroup = volumeToMount.Pod.Spec.SecurityContext.FSGroup
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

			resizeOptions.DeviceMountPath = deviceMountPath
			resizeOptions.CSIVolumePhase = volume.CSIVolumeStaged

			// resizeFileSystem will resize the file system if user has requested a resize of
			// underlying persistent volume and is allowed to do so.
			resizeDone, resizeError = og.resizeFileSystem(volumeToMount, resizeOptions)

			if resizeError != nil {
				klog.Errorf("MountVolume.resizeFileSystem failed with %v", resizeError)
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
			FsGroup:     fsGroup,
			DesiredSize: volumeToMount.DesiredSizeLimit,
		})
		if mountErr != nil {
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
			resizeDone, resizeError = og.resizeFileSystem(volumeToMount, resizeOptions)
			if resizeError != nil {
				klog.Errorf("MountVolume.resizeFileSystem failed with %v", resizeError)
				return volumeToMount.GenerateError("MountVolume.Setup failed while expanding volume", resizeError)
			}
		}

		// Update actual state of world
		markVolMountedErr := actualStateOfWorld.MarkVolumeAsMounted(
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			volumeToMount.VolumeName,
			volumeMounter,
			nil,
			volumeToMount.OuterVolumeSpecName,
			volumeToMount.VolumeGidValue,
			originalSpec)
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

func (og *operationGenerator) resizeFileSystem(volumeToMount VolumeToMount, rsOpts volume.NodeResizeOptions) (bool, error) {
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
		pvc, err := og.kubeClient.CoreV1().PersistentVolumeClaims(pv.Spec.ClaimRef.Namespace).Get(pv.Spec.ClaimRef.Name, metav1.GetOptions{})
		if err != nil {
			// Return error rather than leave the file system un-resized, caller will log and retry
			return false, fmt.Errorf("MountVolume.resizeFileSystem get PVC failed : %v", err)
		}

		pvcStatusCap := pvc.Status.Capacity[v1.ResourceStorage]
		pvSpecCap := pv.Spec.Capacity[v1.ResourceStorage]
		if pvcStatusCap.Cmp(pvSpecCap) < 0 {
			// File system resize was requested, proceed
			klog.V(4).Infof(volumeToMount.GenerateMsgDetailed("MountVolume.resizeFileSystem entering", fmt.Sprintf("DevicePath %q", volumeToMount.DevicePath)))

			if volumeToMount.VolumeSpec.ReadOnly {
				simpleMsg, detailedMsg := volumeToMount.GenerateMsg("MountVolume.resizeFileSystem failed", "requested read-only file system")
				klog.Warningf(detailedMsg)
				og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.FileSystemResizeFailed, simpleMsg)
				return true, nil
			}
			rsOpts.VolumeSpec = volumeToMount.VolumeSpec
			rsOpts.NewSize = pvSpecCap
			rsOpts.OldSize = pvcStatusCap
			resizeDone, resizeErr := expandableVolumePlugin.NodeExpand(rsOpts)
			if resizeErr != nil {
				return false, fmt.Errorf("MountVolume.resizeFileSystem failed : %v", resizeErr)
			}
			// Volume resizing is not done but it did not error out. This could happen if a CSI volume
			// does not have node stage_unstage capability but was asked to resize the volume before
			// node publish. In which case - we must retry resizing after node publish.
			if !resizeDone {
				return false, nil
			}
			simpleMsg, detailedMsg := volumeToMount.GenerateMsg("MountVolume.resizeFileSystem succeeded", "")
			og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeNormal, kevents.FileSystemResizeSuccess, simpleMsg)
			klog.Infof(detailedMsg)
			// File system resize succeeded, now update the PVC's Capacity to match the PV's
			err = util.MarkFSResizeFinished(pvc, pvSpecCap, og.kubeClient)
			if err != nil {
				// On retry, resizeFileSystem will be called again but do nothing
				return false, fmt.Errorf("MountVolume.resizeFileSystem update PVC status failed : %v", err)
			}
			return true, nil
		}
	}
	return true, nil
}

func (og *operationGenerator) GenerateUnmountVolumeFunc(
	volumeToUnmount MountedVolume,
	actualStateOfWorld ActualStateOfWorldMounterUpdater,
	podsDir string) (volumetypes.GeneratedOperations, error) {

	var pluginName string
	if volumeToUnmount.VolumeSpec != nil && useCSIPlugin(og.volumePluginMgr, volumeToUnmount.VolumeSpec) {
		pluginName = csi.CSIPluginName
	} else {
		pluginName = volumeToUnmount.PluginName
	}

	// Get mountable plugin
	volumePlugin, err := og.volumePluginMgr.FindPluginByName(pluginName)
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
	hostutil mount.HostUtils) (volumetypes.GeneratedOperations, error) {

	var pluginName string
	if useCSIPlugin(og.volumePluginMgr, deviceToDetach.VolumeSpec) {
		pluginName = csi.CSIPluginName
		csiSpec, err := translateSpec(deviceToDetach.VolumeSpec)
		if err != nil {
			return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmountDevice.TranslateSpec failed", err)
		}
		deviceToDetach.VolumeSpec = csiSpec
	} else {
		pluginName = deviceToDetach.PluginName
	}

	// Get DeviceMounter plugin
	deviceMountableVolumePlugin, err :=
		og.volumePluginMgr.FindDeviceMountablePluginByName(pluginName)
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
			// On failure, return error. Caller will log and retry.
			return deviceToDetach.GenerateError("GetDeviceMountPath failed", err)
		}
		refs, err := deviceMountableVolumePlugin.GetDeviceMountRefs(deviceMountPath)

		if err != nil || mount.HasMountRefs(deviceMountPath, refs) {
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

	originalSpec := volumeToMount.VolumeSpec
	// Translate to CSI spec if migration enabled
	if useCSIPlugin(og.volumePluginMgr, originalSpec) {
		csiSpec, err := translateSpec(originalSpec)
		if err != nil {
			return volumetypes.GeneratedOperations{}, volumeToMount.GenerateErrorDetailed("MapVolume.TranslateSpec failed", err)
		}

		volumeToMount.VolumeSpec = csiSpec
	}

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

	mapVolumeFunc := func() (error, error) {
		var devicePath string
		// Set up global map path under the given plugin directory using symbolic link
		globalMapPath, err :=
			blockVolumeMapper.GetGlobalMapPath(volumeToMount.VolumeSpec)
		if err != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.GetDeviceMountPath failed", err)
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
		// A plugin doesn't have attacher also needs to map device to global map path with SetUpDevice()
		pluginDevicePath, mapErr := blockVolumeMapper.SetUpDevice()
		if mapErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.SetUp failed", mapErr)
		}

		// if pluginDevicePath is provided, assume attacher may not provide device
		// or attachment flow uses SetupDevice to get device path
		if len(pluginDevicePath) != 0 {
			devicePath = pluginDevicePath
		}
		if len(devicePath) == 0 {
			return volumeToMount.GenerateError("MapVolume failed", goerrors.New("Device path of the volume is empty"))
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

		// Map device to global and pod device map path
		volumeMapPath, volName := blockVolumeMapper.GetPodDeviceMapPath()
		mapErr = blockVolumeMapper.MapDevice(devicePath, globalMapPath, volumeMapPath, volName, volumeToMount.Pod.UID)
		if mapErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.MapDevice failed", mapErr)
		}

		// Take filedescriptor lock to keep a block device opened. Otherwise, there is a case
		// that the block device is silently removed and attached another device with same name.
		// Container runtime can't handler this problem. To avoid unexpected condition fd lock
		// for the block device is required.
		_, err = og.blkUtil.AttachFileDevice(devicePath)
		if err != nil {
			return volumeToMount.GenerateError("MapVolume.AttachFileDevice failed", err)
		}

		// Update actual state of world to reflect volume is globally mounted
		markDeviceMappedErr := actualStateOfWorld.MarkDeviceAsMounted(
			volumeToMount.VolumeName, devicePath, globalMapPath)
		if markDeviceMappedErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToMount.GenerateError("MapVolume.MarkDeviceAsMounted failed", markDeviceMappedErr)
		}

		// Device mapping for global map path succeeded
		simpleMsg, detailedMsg := volumeToMount.GenerateMsg("MapVolume.MapDevice succeeded", fmt.Sprintf("globalMapPath %q", globalMapPath))
		verbosity := klog.Level(4)
		og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeNormal, kevents.SuccessfulMountVolume, simpleMsg)
		klog.V(verbosity).Infof(detailedMsg)

		// Device mapping for pod device map path succeeded
		simpleMsg, detailedMsg = volumeToMount.GenerateMsg("MapVolume.MapDevice succeeded", fmt.Sprintf("volumeMapPath %q", volumeMapPath))
		verbosity = klog.Level(1)
		og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeNormal, kevents.SuccessfulMountVolume, simpleMsg)
		klog.V(verbosity).Infof(detailedMsg)

		// Update actual state of world
		markVolMountedErr := actualStateOfWorld.MarkVolumeAsMounted(
			volumeToMount.PodName,
			volumeToMount.Pod.UID,
			volumeToMount.VolumeName,
			nil,
			blockVolumeMapper,
			volumeToMount.OuterVolumeSpecName,
			volumeToMount.VolumeGidValue,
			originalSpec)
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

	var blockVolumePlugin volume.BlockVolumePlugin
	var err error
	// Translate to CSI spec if migration enabled
	// And get block volume unmapper plugin
	if volumeToUnmount.VolumeSpec != nil && useCSIPlugin(og.volumePluginMgr, volumeToUnmount.VolumeSpec) {
		csiSpec, err := translateSpec(volumeToUnmount.VolumeSpec)
		if err != nil {
			return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.TranslateSpec failed", err)
		}

		volumeToUnmount.VolumeSpec = csiSpec

		blockVolumePlugin, err =
			og.volumePluginMgr.FindMapperPluginByName(csi.CSIPluginName)
		if err != nil {
			return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.FindMapperPluginByName failed", err)
		}
	} else {
		blockVolumePlugin, err =
			og.volumePluginMgr.FindMapperPluginByName(volumeToUnmount.PluginName)
		if err != nil {
			return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.FindMapperPluginByName failed", err)
		}
	}

	var blockVolumeUnmapper volume.BlockVolumeUnmapper
	if blockVolumePlugin == nil {
		return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.FindMapperPluginByName failed to find BlockVolumeMapper plugin. Volume plugin is nil.", nil)
	}
	blockVolumeUnmapper, newUnmapperErr := blockVolumePlugin.NewBlockVolumeUnmapper(
		volumeToUnmount.InnerVolumeSpecName, volumeToUnmount.PodUID)
	if newUnmapperErr != nil {
		return volumetypes.GeneratedOperations{}, volumeToUnmount.GenerateErrorDetailed("UnmapVolume.NewUnmapper failed", newUnmapperErr)
	}

	unmapVolumeFunc := func() (error, error) {
		// Try to unmap volumeName symlink under pod device map path dir
		// pods/{podUid}/volumeDevices/{escapeQualifiedPluginName}/{volumeName}
		podDeviceUnmapPath, volName := blockVolumeUnmapper.GetPodDeviceMapPath()
		unmapDeviceErr := og.blkUtil.UnmapDevice(podDeviceUnmapPath, volName)
		if unmapDeviceErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToUnmount.GenerateError("UnmapVolume.UnmapDevice on pod device map path failed", unmapDeviceErr)
		}
		// Try to unmap podUID symlink under global map path dir
		// plugins/kubernetes.io/{PluginName}/volumeDevices/{volumePluginDependentPath}/{podUID}
		globalUnmapPath := volumeToUnmount.DeviceMountPath
		unmapDeviceErr = og.blkUtil.UnmapDevice(globalUnmapPath, string(volumeToUnmount.PodUID))
		if unmapDeviceErr != nil {
			// On failure, return error. Caller will log and retry.
			return volumeToUnmount.GenerateError("UnmapVolume.UnmapDevice on global map path failed", unmapDeviceErr)
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
	hostutil mount.HostUtils) (volumetypes.GeneratedOperations, error) {

	var blockVolumePlugin volume.BlockVolumePlugin
	var err error
	// Translate to CSI spec if migration enabled
	if useCSIPlugin(og.volumePluginMgr, deviceToDetach.VolumeSpec) {
		csiSpec, err := translateSpec(deviceToDetach.VolumeSpec)
		if err != nil {
			return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmapDevice.TranslateSpec failed", err)
		}

		deviceToDetach.VolumeSpec = csiSpec
		blockVolumePlugin, err =
			og.volumePluginMgr.FindMapperPluginByName(csi.CSIPluginName)
		if err != nil {
			return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmapDevice.FindMapperPluginByName failed", err)
		}
	} else {
		blockVolumePlugin, err =
			og.volumePluginMgr.FindMapperPluginByName(deviceToDetach.PluginName)
		if err != nil {
			return volumetypes.GeneratedOperations{}, deviceToDetach.GenerateErrorDetailed("UnmapDevice.FindMapperPluginByName failed", err)
		}
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
		refs, err := og.blkUtil.GetDeviceSymlinkRefs(deviceToDetach.DevicePath, globalMapPath)
		if err != nil {
			return deviceToDetach.GenerateError("UnmapDevice.GetDeviceSymlinkRefs check failed", err)
		}
		if len(refs) > 0 {
			err = fmt.Errorf("The device %q is still referenced from other Pods %v", globalMapPath, refs)
			return deviceToDetach.GenerateError("UnmapDevice failed", err)
		}

		// The block volume is not referenced from Pods. Release file descriptor lock.
		// This should be done before calling TearDownDevice, because some plugins that do local detach
		// in TearDownDevice will fail in detaching device due to the refcnt on the loopback device.
		klog.V(4).Infof("UnmapDevice: deviceToDetach.DevicePath: %v", deviceToDetach.DevicePath)
		loopPath, err := og.blkUtil.GetLoopDevice(deviceToDetach.DevicePath)
		if err != nil {
			if err.Error() == volumepathhandler.ErrDeviceNotFound {
				klog.Warningf(deviceToDetach.GenerateMsgDetailed("UnmapDevice: Couldn't find loopback device which takes file descriptor lock", fmt.Sprintf("device path: %q", deviceToDetach.DevicePath)))
			} else {
				errInfo := "UnmapDevice.GetLoopDevice failed to get loopback device, " + fmt.Sprintf("device path: %q", deviceToDetach.DevicePath)
				return deviceToDetach.GenerateError(errInfo, err)
			}
		} else {
			if len(loopPath) != 0 {
				err = og.blkUtil.RemoveLoopDevice(loopPath)
				if err != nil {
					return deviceToDetach.GenerateError("UnmapDevice.RemoveLoopDevice failed", err)
				}
			}
		}

		// Execute tear down device
		unmapErr := blockVolumeUnmapper.TearDownDevice(globalMapPath, deviceToDetach.DevicePath)
		if unmapErr != nil {
			// On failure, return error. Caller will log and retry.
			return deviceToDetach.GenerateError("UnmapDevice.TearDownDevice failed", unmapErr)
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
		node, fetchErr := og.kubeClient.CoreV1().Nodes().Get(string(nodeName), metav1.GetOptions{})
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
	node, fetchErr := og.kubeClient.CoreV1().Nodes().Get(string(volumeToDetach.NodeName), metav1.GetOptions{})
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

	fsResizeFunc := func() (error, error) {
		// Need to translate the spec here if the plugin is migrated so that the metrics
		// emitted show the correct (migrated) plugin
		if useCSIPlugin(og.volumePluginMgr, volumeToMount.VolumeSpec) {
			csiSpec, err := translateSpec(volumeToMount.VolumeSpec)
			if err != nil {
				return volumeToMount.GenerateError("VolumeFSResize.translateSpec failed", err)
			}
			volumeToMount.VolumeSpec = csiSpec
		}
		volumePlugin, err :=
			og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
		if err != nil || volumePlugin == nil {
			return volumeToMount.GenerateError("VolumeFSResize.FindPluginBySpec failed", err)
		}

		var resizeDone bool
		var simpleErr, detailedErr error
		resizeOptions := volume.NodeResizeOptions{
			VolumeSpec: volumeToMount.VolumeSpec,
		}

		attachableVolumePlugin, _ :=
			og.volumePluginMgr.FindAttachablePluginBySpec(volumeToMount.VolumeSpec)

		if attachableVolumePlugin != nil {
			volumeAttacher, _ := attachableVolumePlugin.NewAttacher()
			if volumeAttacher != nil {
				resizeOptions.CSIVolumePhase = volume.CSIVolumeStaged
				resizeOptions.DevicePath = volumeToMount.DevicePath
				dmp, err := volumeAttacher.GetDeviceMountPath(volumeToMount.VolumeSpec)
				if err != nil {
					return volumeToMount.GenerateError("VolumeFSResize.GetDeviceMountPath failed", err)
				}
				resizeOptions.DeviceMountPath = dmp
				resizeDone, simpleErr, detailedErr = og.doOnlineExpansion(volumeToMount, actualStateOfWorld, resizeOptions)
				if simpleErr != nil || detailedErr != nil {
					return simpleErr, detailedErr
				}
				if resizeDone {
					return nil, nil
				}
			}
		}
		// if we are here that means volume plugin does not support attach interface
		volumeMounter, newMounterErr := volumePlugin.NewMounter(
			volumeToMount.VolumeSpec,
			volumeToMount.Pod,
			volume.VolumeOptions{})
		if newMounterErr != nil {
			return volumeToMount.GenerateError("VolumeFSResize.NewMounter initialization failed", newMounterErr)
		}

		resizeOptions.DeviceMountPath = volumeMounter.GetPath()
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
		return volumeToMount.GenerateError("VolumeFSResize.resizeFileSystem failed to resize volume", err)
	}

	eventRecorderFunc := func(err *error) {
		if *err != nil {
			og.recorder.Eventf(volumeToMount.Pod, v1.EventTypeWarning, kevents.VolumeResizeFailed, (*err).Error())
		}
	}

	// Need to translate the spec here if the plugin is migrated so that the metrics
	// emitted show the correct (migrated) plugin
	if useCSIPlugin(og.volumePluginMgr, volumeToMount.VolumeSpec) {
		csiSpec, err := translateSpec(volumeToMount.VolumeSpec)
		if err == nil {
			volumeToMount.VolumeSpec = csiSpec
		}
		// If we have an error here we ignore it, the metric emitted will then be for the
		// in-tree plugin. This error case(skipped one) will also trigger an error
		// while the generated function is executed. And those errors will be handled during the execution of the generated
		// function with a back off policy.
	}
	volumePlugin, err :=
		og.volumePluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec)
	if err != nil || volumePlugin == nil {
		return volumetypes.GeneratedOperations{}, volumeToMount.GenerateErrorDetailed("VolumeFSResize.FindPluginBySpec failed", err)
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
	resizeDone, err := og.resizeFileSystem(volumeToMount, resizeOptions)
	if err != nil {
		klog.Errorf("VolumeFSResize.resizeFileSystem failed : %v", err)
		e1, e2 := volumeToMount.GenerateError("VolumeFSResize.resizeFileSystem failed", err)
		return false, e1, e2
	}
	if resizeDone {
		markFSResizedErr := actualStateOfWorld.MarkVolumeAsResized(volumeToMount.PodName, volumeToMount.VolumeName)
		if markFSResizedErr != nil {
			// On failure, return error. Caller will log and retry.
			e1, e2 := volumeToMount.GenerateError("VolumeFSResize.MarkVolumeAsResized failed", markFSResizedErr)
			return false, e1, e2
		}
		return true, nil, nil
	}
	return false, nil, nil
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
	if !utilfeature.DefaultFeatureGate.Enabled(features.PersistentLocalVolumes) {
		return nil
	}

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
func isDeviceOpened(deviceToDetach AttachedVolume, hostUtil mount.HostUtils) (bool, error) {
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

func useCSIPlugin(vpm *volume.VolumePluginMgr, spec *volume.Spec) bool {
	// TODO(#75146) Check whether the driver is installed as well so that
	// we can throw a better error when the driver is not installed.
	// The error should be of the approximate form:
	// fmt.Errorf("in-tree plugin %s is migrated on node %s but driver %s is not installed", pluginName, string(nodeName), driverName)
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) {
		return false
	}
	if csilib.IsPVMigratable(spec.PersistentVolume) || csilib.IsInlineMigratable(spec.Volume) {
		migratable, err := vpm.IsPluginMigratableBySpec(spec)
		if err == nil && migratable {
			return true
		}
	}
	return false
}

func nodeUsingCSIPlugin(og OperationGenerator, spec *volume.Spec, nodeName types.NodeName) (bool, error) {
	vpm := og.GetVolumePluginMgr()

	migratable, err := vpm.IsPluginMigratableBySpec(spec)
	if err != nil {
		return false, err
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) ||
		!utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) ||
		!migratable {
		return false, nil
	}

	if len(nodeName) == 0 {
		return false, goerrors.New("nodeName is empty")
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
		return false, goerrors.New("could not find CSINodeLister in attachDetachController")
	}

	csiNode, err := adcHost.CSINodeLister().Get(string(nodeName))
	if err != nil {
		return false, err
	}

	ann := csiNode.GetAnnotations()
	if ann == nil {
		return false, nil
	}

	var mpaSet sets.String
	mpa := ann[v1.MigratedPluginsAnnotationKey]
	tok := strings.Split(mpa, ",")
	if len(mpa) == 0 {
		mpaSet = sets.NewString()
	} else {
		mpaSet = sets.NewString(tok...)
	}

	pluginName, err := csilib.GetInTreePluginNameFromSpec(spec.PersistentVolume, spec.Volume)
	if err != nil {
		return false, err
	}

	if len(pluginName) == 0 {
		// Could not find a plugin name from translation directory, assume not translated
		return false, nil
	}

	isMigratedOnNode := mpaSet.Has(pluginName)

	if isMigratedOnNode {
		installed := false
		driverName, err := csilib.GetCSINameFromInTreeName(pluginName)
		if err != nil {
			return isMigratedOnNode, err
		}
		for _, driver := range csiNode.Spec.Drivers {
			if driver.Name == driverName {
				installed = true
				break
			}
		}
		if !installed {
			return true, fmt.Errorf("in-tree plugin %s is migrated on node %s but driver %s is not installed", pluginName, string(nodeName), driverName)
		}
	}

	return isMigratedOnNode, nil

}

func translateSpec(spec *volume.Spec) (*volume.Spec, error) {
	var csiPV *v1.PersistentVolume
	var err error
	inlineVolume := false
	if spec.PersistentVolume != nil {
		// TranslateInTreePVToCSI will create a new PV
		csiPV, err = csilib.TranslateInTreePVToCSI(spec.PersistentVolume)
		if err != nil {
			return nil, fmt.Errorf("failed to translate in tree pv to CSI: %v", err)
		}
	} else if spec.Volume != nil {
		// TranslateInTreeInlineVolumeToCSI will create a new PV
		csiPV, err = csilib.TranslateInTreeInlineVolumeToCSI(spec.Volume)
		if err != nil {
			return nil, fmt.Errorf("failed to translate in tree inline volume to CSI: %v", err)
		}
		inlineVolume = true
	} else {
		return &volume.Spec{}, goerrors.New("not a valid volume spec")
	}
	return &volume.Spec{
		PersistentVolume:                csiPV,
		ReadOnly:                        spec.ReadOnly,
		InlineVolumeSpecForCSIMigration: inlineVolume,
	}, nil
}
