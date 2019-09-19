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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/util/nestedpendingoperations"
)

// fakeOGCounter is a simple OperationGenerator which counts number of times a function
// has been caled
type fakeOGCounter struct {
	// calledFuncs stores name and count of functions
	calledFuncs map[string]int
	opFunc      func() (error, error)
}

var _ OperationGenerator = &fakeOGCounter{}

// NewFakeOGCounter returns a OperationGenerator
func NewFakeOGCounter(opFunc func() (error, error)) OperationGenerator {
	return &fakeOGCounter{
		calledFuncs: map[string]int{},
		opFunc:      opFunc,
	}
}

func (f *fakeOGCounter) GenerateMountVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater, isRemount bool) nestedpendingoperations.GeneratedOperations {
	return f.recordFuncCall("GenerateMountVolumeFunc")
}

func (f *fakeOGCounter) GenerateUnmountVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, podsDir string) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmountVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateAttachVolumeFunc(volumeToAttach VolumeToAttach, actualStateOfWorld ActualStateOfWorldAttacherUpdater) nestedpendingoperations.GeneratedOperations {
	return f.recordFuncCall("GenerateAttachVolumeFunc")
}

func (f *fakeOGCounter) GenerateDetachVolumeFunc(volumeToDetach AttachedVolume, verifySafeToDetach bool, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateDetachVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateVolumesAreAttachedFunc(attachedVolumes []AttachedVolume, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateVolumesAreAttachedFunc"), nil
}

func (f *fakeOGCounter) GenerateUnmountDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, hu hostutil.HostUtils) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmountDeviceFunc"), nil
}

func (f *fakeOGCounter) GenerateVerifyControllerAttachedVolumeFunc(volumeToMount VolumeToMount, nodeName types.NodeName, actualStateOfWorld ActualStateOfWorldAttacherUpdater) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateVerifyControllerAttachedVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateMapVolumeFunc(waitForAttachTimeout time.Duration, volumeToMount VolumeToMount, actualStateOfWorldMounterUpdater ActualStateOfWorldMounterUpdater) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateMapVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateUnmapVolumeFunc(volumeToUnmount MountedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmapVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateUnmapDeviceFunc(deviceToDetach AttachedVolume, actualStateOfWorld ActualStateOfWorldMounterUpdater, hu hostutil.HostUtils) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateUnmapDeviceFunc"), nil
}

func (f *fakeOGCounter) GetVolumePluginMgr() *volume.VolumePluginMgr {
	return nil
}

func (f *fakeOGCounter) GenerateBulkVolumeVerifyFunc(
	map[types.NodeName][]*volume.Spec,
	string,
	map[*volume.Spec]v1.UniqueVolumeName, ActualStateOfWorldAttacherUpdater) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateBulkVolumeVerifyFunc"), nil
}

func (f *fakeOGCounter) GenerateExpandVolumeFunc(*v1.PersistentVolumeClaim, *v1.PersistentVolume) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateExpandVolumeFunc"), nil
}

func (f *fakeOGCounter) GenerateExpandInUseVolumeFunc(volumeToMount VolumeToMount, actualStateOfWorld ActualStateOfWorldMounterUpdater) (nestedpendingoperations.GeneratedOperations, error) {
	return f.recordFuncCall("GenerateExpandInUseVolumeFunc"), nil
}

func (f *fakeOGCounter) recordFuncCall(name string) nestedpendingoperations.GeneratedOperations {
	if _, ok := f.calledFuncs[name]; ok {
		f.calledFuncs[name]++
	}
	ops := nestedpendingoperations.GeneratedOperations{
		OperationName: name,
		OperationFunc: f.opFunc,
	}
	return ops
}
