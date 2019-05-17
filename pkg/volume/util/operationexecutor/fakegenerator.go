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
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// fakeOgCounter is a simple OperationGenerator which counts number of times a function
// has been caled
type fakeOgCounter struct {
	// calledFuncs stores name and count of functions
	calledFuncs map[string]int
	opFunc      func() (error, error)
}

var _ OperationGenerator = &fakeOgCounter{}

// NewFakeOgCounter returns a OperationGenerator
func NewFakeOgCounter(opFunc func() (error, error)) OperationGenerator {
	return &fakeOgCounter{
		calledFuncs: map[string]int{},
		opFunc:      opFunc,
	}
}

func (f *fakeOgCounter) GenerateMountVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater, isRemount bool) volumetypes.GeneratedOperations {
	return f.recordFuncCall("GenerateMountVolumeFunc")
}

func (f *fakeOgCounter) GenerateUnmountVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, podsDir string) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmountVolumeFunc"), nil
}

func (f *fakeOgCounter) GenerateAttachVolumeFunc(volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) volumetypes.GeneratedOperations {
	return f.recordFuncCall("GenerateAttachVolumeFunc")
}

func (f *fakeOgCounter) GenerateDetachVolumeFunc(volumeToDetach AttachedVolume, verifySafeToDetach bool, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateDetachVolumeFunc"), nil
}

func (f *fakeOgCounter) GenerateVolumesAreAttachedFunc(attachedVolumes []AttachedVolume, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateVolumesAreAttachedFunc"), nil
}

func (f *fakeOgCounter) GenerateUnmountDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter mount.Interface) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmountDeviceFunc"), nil
}

func (f *fakeOgCounter) GenerateVerifyControllerAttachedVolumeFunc(volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateVerifyControllerAttachedVolumeFunc"), nil
}

func (f *fakeOgCounter) GenerateMapVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateMapVolumeFunc"), nil
}

func (f *fakeOgCounter) GenerateUnmapVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmapVolumeFunc"), nil
}

func (f *fakeOgCounter) GenerateUnmapDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, mounter mount.Interface) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmapDeviceFunc"), nil
}

func (f *fakeOgCounter) GetVolumePluginMgr() *volume.VolumePluginMgr {
	return nil
}

func (f *fakeOgCounter) GenerateBulkVolumeVerifyFunc(
	map[types.NodeName][]*volume.Spec,
	string,
	map[*volume.Spec]v1.UniqueVolumeName, ActualStateOfWorldAttacherUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateBulkVolumeVerifyFunc"), nil
}

func (f *fakeOgCounter) GenerateExpandVolumeFunc(*v1.PersistentVolumeClaim, *v1.PersistentVolume) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateExpandVolumeFunc"), nil
}

func (f *fakeOgCounter) GenerateExpandVolumeFSWithoutUnmountingFunc(volumeToMount VolumeToMount, actualStateOfWorld ActualStateOfWorldMounterUpdater) (volumetypes.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateExpandVolumeFSWithoutUnmountingFunc"), nil
}

func (f *fakeOgCounter) recordFuncCall(name string) volumetypes.GeneratedOperations {
	if _, ok := f.calledFuncs[name]; ok {
		f.calledFuncs[name]++
	}
	ops := volumetypes.GeneratedOperations{
		OperationName: name,
		OperationFunc: f.opFunc,
	}
	return ops
}
