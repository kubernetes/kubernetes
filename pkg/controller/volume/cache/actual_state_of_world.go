/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

/*
Package cache implements data structures used by the attach/detach controller
to keep track of volumes, the nodes they are attached to, and the pods that
reference them.
*/
package cache

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/volume"
)

// ActualStateOfWorld defines a set of thread-safe operations supported on
// the attach/detach controller's actual state of the world cache.
// This cache contains volumes->nodes i.e. a set of all volumes and the nodes
// the attach/detach controller believes are successfully attached.
type ActualStateOfWorld interface {
	// AddVolumeNode adds the given volume and node to the underlying store
	// indicating the specified volume is attached to the specified node.
	// A unique volumeName is generated from the volumeSpec and returned on
	// success.
	// If the volume/node combo already exists, this is a no-op.
	// If volumeSpec is not an attachable volume plugin, an error is returned.
	// If no volume with the name volumeName exists in the store, the volume is
	// added.
	// If no node with the name nodeName exists in list of attached nodes for
	// the specified volume, the node is added.
	AddVolumeNode(volumeSpec *volume.Spec, nodeName string) (string, error)

	// MarkVolumeNodeSafeToDetach marks the given volume as safe to detach from
	// the given node.
	// If no volume with the name volumeName exists in the store, an error is
	// returned.
	// If no node with the name nodeName exists in list of attached nodes for
	// the specified volume, an error is returned.
	MarkVolumeNodeSafeToDetach(volumeName, nodeName string) error

	// MarkDesireToDetach returns the difference between the current time  and
	// the DetachRequestedTime for the given volume/node combo. If the
	// DetachRequestedTime is zero, it is set to the current time.
	// If no volume with the name volumeName exists in the store, an error is
	// returned.
	// If no node with the name nodeName exists in list of attached nodes for
	// the specified volume, an error is returned.
	MarkDesireToDetach(volumeName, nodeName string) (time.Duration, error)

	// DeleteVolumeNode removes the given volume and node from the underlying
	// store indicating the specified volume is no longer attached to the
	// specified node.
	// If the volume/node combo does not exist, this is a no-op.
	// If after deleting the node, the specified volume contains no other child
	// nodes, the volume is also deleted.
	DeleteVolumeNode(volumeName, nodeName string)

	// VolumeNodeExists returns true if the specified volume/node combo exists
	// in the underlying store indicating the specified volume is attached to
	// the specified node.
	VolumeNodeExists(volumeName, nodeName string) bool

	// GetAttachedVolumes generates and returns a list of volumes/node pairs
	// reflecting which volumes are attached to which nodes based on the
	// current actual state of the world.
	GetAttachedVolumes() []AttachedVolume
}

// AttachedVolume represents a volume that is attached to a node.
type AttachedVolume struct {
	// VolumeName is the unique identifier for the volume that is attached.
	VolumeName string

	// VolumeSpec is the volume spec containing the specification for the
	// volume that is attached.
	VolumeSpec *volume.Spec

	// NodeName is the identifier for the node that the volume is attached to.
	NodeName string

	// SafeToDetach indicates that this volume has been been unmounted from the
	// node and is safe to detach.
	// The value is set by MarkVolumeNodeSafeToDetach(...) and  reset on
	// AddVolumeNode(...) calls.
	SafeToDetach bool

	// DetachRequestedTime is used to capture the desire to detach this volume.
	// When the volume is newly created this value is set to time zero.
	// It is set to current time, when MarkDesireToDetach(...) is called, if it
	// was previously set to zero (other wise its value remains the same).
	// It is reset to zero on AddVolumeNode(...) calls.
	DetachRequestedTime time.Time
}

// NewActualStateOfWorld returns a new instance of ActualStateOfWorld.
func NewActualStateOfWorld(volumePluginMgr *volume.VolumePluginMgr) ActualStateOfWorld {
	return &actualStateOfWorld{
		attachedVolumes: make(map[string]attachedVolume),
		volumePluginMgr: volumePluginMgr,
	}
}

type actualStateOfWorld struct {
	// attachedVolumes is a map containing the set of volumes the attach/detach
	// controller believes to be successfully attached to the nodes it is
	// managing. The key in this map is the name of the volume and the value is
	// an object containing more information about the attached volume.
	attachedVolumes map[string]attachedVolume
	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr
	sync.RWMutex
}

// The volume object represents a volume the the attach/detach controller
// believes to be succesfully attached to a node it is managing.
type attachedVolume struct {
	// volumeName contains the unique identifier for this volume.
	volumeName string

	// spec is the volume spec containing the specification for this volume.
	// Used to generate the volume plugin object, and passed to attach/detach
	// methods.
	spec *volume.Spec

	// nodesAttachedTo is a map containing the set of nodes this volume has
	// successfully been attached to. The key in this map is the name of the
	// node and the value is a node object containing more information about
	// the node.
	nodesAttachedTo map[string]nodeAttachedTo
}

// The nodeAttachedTo object represents a node that .
type nodeAttachedTo struct {
	// nodeName contains the name of this node.
	nodeName string

	// safeToDetach indicates that this node/volume combo has been unmounted
	// by the node and is safe to detach
	safeToDetach bool

	// detachRequestedTime used to capture the desire to detach this volume
	detachRequestedTime time.Time
}

func (asw *actualStateOfWorld) AddVolumeNode(volumeSpec *volume.Spec, nodeName string) (string, error) {
	asw.Lock()
	defer asw.Unlock()

	attachableVolumePlugin, err := asw.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return "", fmt.Errorf(
			"failed to get AttachablePlugin from volumeSpec for volume %q err=%v",
			volumeSpec.Name(),
			err)
	}

	volumeName, err := attachableVolumePlugin.GetUniqueVolumeName(volumeSpec)
	if err != nil {
		return "", fmt.Errorf(
			"failed to GetUniqueVolumeName from AttachablePlugin for volumeSpec %q err=%v",
			volumeSpec.Name(),
			err)
	}

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		volumeObj = attachedVolume{
			volumeName:      volumeName,
			spec:            volumeSpec,
			nodesAttachedTo: make(map[string]nodeAttachedTo),
		}
		asw.attachedVolumes[volumeName] = volumeObj
	}

	nodeObj, nodeExists := volumeObj.nodesAttachedTo[nodeName]
	if !nodeExists || nodeObj.safeToDetach || !nodeObj.detachRequestedTime.IsZero() {
		// Create object if it doesn't exist.
		// Reset safeToDeatch and detachRequestedTime values if it does.
		volumeObj.nodesAttachedTo[nodeName] = nodeAttachedTo{
			nodeName:            nodeName,
			safeToDetach:        false,
			detachRequestedTime: time.Time{},
		}
	}

	return volumeName, nil
}

func (asw *actualStateOfWorld) MarkVolumeNodeSafeToDetach(
	volumeName, nodeName string) error {
	asw.Lock()
	defer asw.Unlock()
	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return fmt.Errorf(
			"failed to MarkVolumeNodeSafeToDetach(volumeName=%q, nodeName=%q) volumeName does not exist",
			volumeName,
			nodeName)
	}

	nodeObj, nodeExists := volumeObj.nodesAttachedTo[nodeName]
	if !nodeExists {
		return fmt.Errorf(
			"failed to MarkVolumeNodeSafeToDetach(volumeName=%q, nodeName=%q) nodeName does not exist",
			volumeName,
			nodeName)
	}

	// Reset safe to detach
	nodeObj.safeToDetach = true
	volumeObj.nodesAttachedTo[nodeName] = nodeObj

	return nil
}

func (asw *actualStateOfWorld) MarkDesireToDetach(
	volumeName, nodeName string) (time.Duration, error) {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return time.Millisecond * 0, fmt.Errorf(
			"failed to MarkVolumeNodeSafeToDetach(volumeName=%q, nodeName=%q) volumeName does not exist",
			volumeName,
			nodeName)
	}

	nodeObj, nodeExists := volumeObj.nodesAttachedTo[nodeName]
	if !nodeExists {
		return time.Millisecond * 0, fmt.Errorf(
			"failed to MarkVolumeNodeSafeToDetach(volumeName=%q, nodeName=%q) nodeName does not exist",
			volumeName,
			nodeName)
	}

	if nodeObj.detachRequestedTime.IsZero() {
		nodeObj.detachRequestedTime = time.Now()
		volumeObj.nodesAttachedTo[nodeName] = nodeObj
	}

	return time.Since(volumeObj.nodesAttachedTo[nodeName].detachRequestedTime), nil
}

func (asw *actualStateOfWorld) DeleteVolumeNode(volumeName, nodeName string) {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		return
	}

	_, nodeExists := volumeObj.nodesAttachedTo[nodeName]
	if nodeExists {
		delete(asw.attachedVolumes[volumeName].nodesAttachedTo, nodeName)
	}

	if len(volumeObj.nodesAttachedTo) == 0 {
		delete(asw.attachedVolumes, volumeName)
	}
}

func (asw *actualStateOfWorld) VolumeNodeExists(volumeName, nodeName string) bool {
	asw.RLock()
	defer asw.RUnlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if volumeExists {
		if _, nodeExists := volumeObj.nodesAttachedTo[nodeName]; nodeExists {
			return true
		}
	}

	return false
}

func (asw *actualStateOfWorld) GetAttachedVolumes() []AttachedVolume {
	asw.RLock()
	defer asw.RUnlock()

	attachedVolumes := make([]AttachedVolume, 0 /* len */, len(asw.attachedVolumes) /* cap */)
	for volumeName, volumeObj := range asw.attachedVolumes {
		for nodeName, nodeObj := range volumeObj.nodesAttachedTo {
			attachedVolumes = append(
				attachedVolumes,
				AttachedVolume{
					NodeName:            nodeName,
					VolumeName:          volumeName,
					VolumeSpec:          volumeObj.spec,
					SafeToDetach:        nodeObj.safeToDetach,
					DetachRequestedTime: nodeObj.detachRequestedTime})
		}
	}

	return attachedVolumes
}
