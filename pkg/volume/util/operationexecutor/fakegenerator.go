/*
Copyright 2019 The Kubernetes Authors.

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
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// fakeOGCounter is a simple OperationGenerator which counts number of times a function
// has been caled
type fakeOGCounter struct {
	// calledFuncs stores name and count of functions
	calledFuncs map[string]int
	opFunc      func() volumetypes.OperationContext
}

var _ OperationGenerator = &fakeOGCounter{}

// NewFakeOGCounter returns a OperationGenerator
func NewFakeOGCounter(opFunc func() volumetypes.OperationContext) OperationGenerator {
	return &fakeOGCounter{
		calledFuncs: map[string]int{},
		opFunc:      opFunc,
	}
}

func (f *fakeOGCounter) GenerateMountVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater, isRemount bool) volumetypes.GeneratedOperations {
	return f.recordFuncCall("GenerateMountVolumeFunc")
}

func (f *fakeOGCounter) GenerateUnmountVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, podsDir string) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmountVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateAttachVolumeFunc(logger klog.Logger, volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) volumetypes.GeneratedOperations {
	return f.recordFuncCall("GenerateAttachVolumeFunc")
}

func (f *fakeOGCounter) GenerateDetachVolumeFunc(logger klog.Logger, volumeToDetach AttachedVolume, verifySafeToDetach bool, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateDetachVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateVolumesAreAttachedFunc(attachedVolumes []AttachedVolume, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateVolumesAreAttachedFunc"), nil
}

func (f *fakeOGCounter) GenerateUnmountDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, hu hostutil.HostUtils) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmountDeviceFunc"), nil
}

func (f *fakeOGCounter) GenerateVerifyControllerAttachedVolumeFunc(logger klog.Logger, volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateVerifyControllerAttachedVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateMapVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateMapVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateUnmapVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmapVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateUnmapDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, hu hostutil.HostUtils) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmapDeviceFunc"), nil
}

func (f *fakeOGCounter) GetVolumePluginMgr() *volume.VolumePluginMgr {
	return nil
}

func (f *fakeOGCounter) GetCSITranslator() InTreeToCSITranslator {
	return csitrans.New()
}

func (f *fakeOGCounter) GenerateExpandVolumeFunc(*v1.PersistentVolumeClaim, *v1.PersistentVolume) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateExpandVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateExpandAndRecoverVolumeFunc(*v1.PersistentVolumeClaim, *v1.PersistentVolume, string) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateExpandVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateExpandInUseVolumeFunc(volumeToMount VolumeToMount, actualStateOfWorld ActualStateOfWorldMounterUpdater, currentSize resource.Quantity) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateExpandInUseVolumeFunc"), nil
}

func (f *fakeOGCounter) recordFuncCall(name string) volumetypes.GeneratedOperations {
	if _, ok := f.calledFuncs[name]; ok {
		f.calledFuncs[name]++
	}
	ops := volumetypes.GeneratedOperations{
		OperationName: name,
		OperationFunc: f.opFunc,
	}
	return ops
}
