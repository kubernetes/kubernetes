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

package cache

import (
	"fmt"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// ActualStateOfWorld defines a set of thread-safe operations supported on
// the snapshot controller's actual state of the world cache.
type ActualStateOfWorld interface {
	operationexecutor.ActualStateOfWorldSnapshotUpdater
	AddVolume(volumeSpec *volume.Spec, pvc *api.PersistentVolumeClaim, snapshotName string) (api.UniqueVolumeName, error)
	DeleteVolume(api.UniqueVolumeName)
	VolumeExists(volumeName api.UniqueVolumeName) bool
	GetVolumesToSnapshot() []VolumeToSnapshot
}

// VolumeToSnapshot represents a volume that needs to be snapshotted
type VolumeToSnapshot struct {
	operationexecutor.VolumeToSnapshot
	// The user specified name for the snapshot
	SnapshotName string
}

type actualStateOfWorld struct {
	volumesToSnapshot map[api.UniqueVolumeName]VolumeToSnapshot

	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr

	sync.RWMutex
}

// NewActualStateOfWorld returns a new instance of ActualStateOfWorld.
func NewActualStateOfWorld(volumePluginMgr *volume.VolumePluginMgr) ActualStateOfWorld {
	return &actualStateOfWorld{
		volumesToSnapshot: make(map[api.UniqueVolumeName]VolumeToSnapshot),
		volumePluginMgr:   volumePluginMgr,
	}
}

func (asw *actualStateOfWorld) MarkVolumeAsSnapshotted(volumeName api.UniqueVolumeName) {
	asw.DeleteVolume(volumeName)
}

func (asw *actualStateOfWorld) AddVolume(
	volumeSpec *volume.Spec,
	pvc *api.PersistentVolumeClaim,
	snapshotName string) (api.UniqueVolumeName, error) {
	asw.Lock()
	defer asw.Unlock()

	snapshottableVolumePlugin, err := asw.volumePluginMgr.FindSnapshottablePluginBySpec(volumeSpec)
	if err != nil || snapshottableVolumePlugin == nil {
		return "", fmt.Errorf(
			"Failed to get AttachablePlugin from volumeSpec for volumeSpec %q err=%v",
			volumeSpec.Name(),
			err)
	}

	uniqueVolumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(
		snapshottableVolumePlugin, volumeSpec)
	if err != nil {
		return "", fmt.Errorf(
			"Failed to GetUniqueVolumeNameFromSpec for volumeSpec %q err=%v",
			volumeSpec.Name(),
			err)
	}

	_, volumeExists := asw.volumesToSnapshot[uniqueVolumeName]
	if !volumeExists {
		volumeObj := VolumeToSnapshot{
			VolumeToSnapshot: operationexecutor.VolumeToSnapshot{
				VolumeName:            uniqueVolumeName,
				VolumeSpec:            volumeSpec,
				PersistentVolumeClaim: pvc,
			},
			SnapshotName: snapshotName}
		asw.volumesToSnapshot[uniqueVolumeName] = volumeObj
	}

	return uniqueVolumeName, nil
}

func (asw *actualStateOfWorld) DeleteVolume(volumeName api.UniqueVolumeName) {
	asw.Lock()
	defer asw.Unlock()

	if _, volumeExists := asw.volumesToSnapshot[volumeName]; !volumeExists {
		return
	}

	delete(asw.volumesToSnapshot, volumeName)
}

func (asw *actualStateOfWorld) VolumeExists(volumeName api.UniqueVolumeName) bool {
	asw.RLock()
	defer asw.RUnlock()

	if _, volumeExists := asw.volumesToSnapshot[volumeName]; volumeExists {
		return true
	}

	return false
}

func (asw *actualStateOfWorld) GetVolumesToSnapshot() []VolumeToSnapshot {
	asw.RLock()
	defer asw.RUnlock()

	volumesToSnapshot := make(
		[]VolumeToSnapshot, 0, len(asw.volumesToSnapshot))
	for _, volumeObj := range asw.volumesToSnapshot {
		volumesToSnapshot = append(volumesToSnapshot, volumeObj)
	}

	return volumesToSnapshot
}
