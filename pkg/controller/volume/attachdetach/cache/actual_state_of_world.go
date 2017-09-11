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

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// ActualStateOfWorld defines a set of thread-safe operations supported on
// the attach/detach controller's actual state of the world cache.
// This cache contains volumes->nodes i.e. a set of all volumes and the nodes
// the attach/detach controller believes are successfully attached.
// Note: This is distinct from the ActualStateOfWorld implemented by the kubelet
// volume manager. They both keep track of different objects. This contains
// attach/detach controller specific state.
type ActualStateOfWorld interface {
	// ActualStateOfWorld must implement the methods required to allow
	// operationexecutor to interact with it.
	operationexecutor.ActualStateOfWorldAttacherUpdater

	// AddVolumeNode adds the given volume and node to the underlying store
	// indicating the specified volume is attached to the specified node.
	// A unique volume name is generated from the volumeSpec and returned on
	// success.
	// If volumeSpec is not an attachable volume plugin, an error is returned.
	// If no volume with the name volumeName exists in the store, the volume is
	// added.
	// If no node with the name nodeName exists in list of attached nodes for
	// the specified volume, the node is added.
	AddVolumeNode(uniqueName v1.UniqueVolumeName, volumeSpec *volume.Spec, nodeName types.NodeName, devicePath string) (v1.UniqueVolumeName, error)

	// SetVolumeMountedByNode sets the MountedByNode value for the given volume
	// and node. When set to true the mounted parameter indicates the volume
	// is mounted by the given node, indicating it may not be safe to detach.
	// If the forceUnmount is set to true the MountedByNode value would be reset
	// to false even it was not set yet (this is required during a controller
	// crash recovery).
	// If no volume with the name volumeName exists in the store, an error is
	// returned.
	// If no node with the name nodeName exists in list of attached nodes for
	// the specified volume, an error is returned.
	SetVolumeMountedByNode(volumeName v1.UniqueVolumeName, nodeName types.NodeName, mounted bool, forceUnmount bool) error

	// SetNodeStatusUpdateNeeded sets statusUpdateNeeded for the specified
	// node to true indicating the AttachedVolume field in the Node's Status
	// object needs to be updated by the node updater again.
	// If the specifed node does not exist in the nodesToUpdateStatusFor list,
	// log the error and return
	SetNodeStatusUpdateNeeded(nodeName types.NodeName)

	// ResetDetachRequestTime resets the detachRequestTime to 0 which indicates there is no detach
	// request any more for the volume
	ResetDetachRequestTime(volumeName v1.UniqueVolumeName, nodeName types.NodeName)

	// SetDetachRequestTime sets the detachRequestedTime to current time if this is no
	// previous request (the previous detachRequestedTime is zero) and return the time elapsed
	// since last request
	SetDetachRequestTime(volumeName v1.UniqueVolumeName, nodeName types.NodeName) (time.Duration, error)

	// DeleteVolumeNode removes the given volume and node from the underlying
	// store indicating the specified volume is no longer attached to the
	// specified node.
	// If the volume/node combo does not exist, this is a no-op.
	// If after deleting the node, the specified volume contains no other child
	// nodes, the volume is also deleted.
	DeleteVolumeNode(volumeName v1.UniqueVolumeName, nodeName types.NodeName)

	// VolumeNodeExists returns true if the specified volume/node combo exists
	// in the underlying store indicating the specified volume is attached to
	// the specified node.
	VolumeNodeExists(volumeName v1.UniqueVolumeName, nodeName types.NodeName) bool

	// GetAttachedVolumes generates and returns a list of volumes/node pairs
	// reflecting which volumes are attached to which nodes based on the
	// current actual state of the world.
	GetAttachedVolumes() []AttachedVolume

	// GetAttachedVolumes generates and returns a list of volumes attached to
	// the specified node reflecting which volumes are attached to that node
	// based on the current actual state of the world.
	GetAttachedVolumesForNode(nodeName types.NodeName) []AttachedVolume

	GetAttachedVolumesPerNode() map[types.NodeName][]operationexecutor.AttachedVolume

	// GetNodesForVolume returns the nodes on which the volume is attached
	GetNodesForVolume(volumeName v1.UniqueVolumeName) []types.NodeName

	// GetVolumesToReportAttached returns a map containing the set of nodes for
	// which the VolumesAttached Status field in the Node API object should be
	// updated. The key in this map is the name of the node to update and the
	// value is list of volumes that should be reported as attached (note that
	// this may differ from the actual list of attached volumes for the node
	// since volumes should be removed from this list as soon a detach operation
	// is considered, before the detach operation is triggered).
	GetVolumesToReportAttached() map[types.NodeName][]v1.AttachedVolume

	// GetNodesToUpdateStatusFor returns the map of nodeNames to nodeToUpdateStatusFor
	GetNodesToUpdateStatusFor() map[types.NodeName]nodeToUpdateStatusFor
}

// AttachedVolume represents a volume that is attached to a node.
type AttachedVolume struct {
	operationexecutor.AttachedVolume

	// MountedByNode indicates that this volume has been been mounted by the
	// node and is unsafe to detach.
	// The value is set and unset by SetVolumeMountedByNode(...).
	MountedByNode bool

	// DetachRequestedTime is used to capture the desire to detach this volume.
	// When the volume is newly created this value is set to time zero.
	// It is set to current time, when SetDetachRequestTime(...) is called, if it
	// was previously set to zero (other wise its value remains the same).
	// It is reset to zero on ResetDetachRequestTime(...) calls.
	DetachRequestedTime time.Time
}

// NewActualStateOfWorld returns a new instance of ActualStateOfWorld.
func NewActualStateOfWorld(volumePluginMgr *volume.VolumePluginMgr) ActualStateOfWorld {
	return &actualStateOfWorld{
		attachedVolumes:        make(map[v1.UniqueVolumeName]attachedVolume),
		nodesToUpdateStatusFor: make(map[types.NodeName]nodeToUpdateStatusFor),
		volumePluginMgr:        volumePluginMgr,
	}
}

type actualStateOfWorld struct {
	// attachedVolumes is a map containing the set of volumes the attach/detach
	// controller believes to be successfully attached to the nodes it is
	// managing. The key in this map is the name of the volume and the value is
	// an object containing more information about the attached volume.
	attachedVolumes map[v1.UniqueVolumeName]attachedVolume

	// nodesToUpdateStatusFor is a map containing the set of nodes for which to
	// update the VolumesAttached Status field. The key in this map is the name
	// of the node and the value is an object containing more information about
	// the node (including the list of volumes to report attached).
	nodesToUpdateStatusFor map[types.NodeName]nodeToUpdateStatusFor

	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr

	sync.RWMutex
}

// The volume object represents a volume the the attach/detach controller
// believes to be successfully attached to a node it is managing.
type attachedVolume struct {
	// volumeName contains the unique identifier for this volume.
	volumeName v1.UniqueVolumeName

	// spec is the volume spec containing the specification for this volume.
	// Used to generate the volume plugin object, and passed to attach/detach
	// methods.
	spec *volume.Spec

	// nodesAttachedTo is a map containing the set of nodes this volume has
	// successfully been attached to. The key in this map is the name of the
	// node and the value is a node object containing more information about
	// the node.
	nodesAttachedTo map[types.NodeName]nodeAttachedTo

	// devicePath contains the path on the node where the volume is attached
	devicePath string
}

// The nodeAttachedTo object represents a node that has volumes attached to it.
type nodeAttachedTo struct {
	// nodeName contains the name of this node.
	nodeName types.NodeName

	// mountedByNode indicates that this node/volume combo is mounted by the
	// node and is unsafe to detach
	mountedByNode bool

	// number of times SetVolumeMountedByNode has been called to set the value
	// of mountedByNode to true. This is used to prevent mountedByNode from
	// being reset during the period between attach and mount when volumesInUse
	// status for the node may not be set.
	mountedByNodeSetCount uint

	// detachRequestedTime used to capture the desire to detach this volume
	detachRequestedTime time.Time
}

// nodeToUpdateStatusFor is an object that reflects a node that has one or more
// volume attached. It keeps track of the volumes that should be reported as
// attached in the Node's Status API object.
type nodeToUpdateStatusFor struct {
	// nodeName contains the name of this node.
	nodeName types.NodeName

	// statusUpdateNeeded indicates that the value of the VolumesAttached field
	// in the Node's Status API object should be updated. This should be set to
	// true whenever a volume is added or deleted from
	// volumesToReportAsAttached. It should be reset whenever the status is
	// updated.
	statusUpdateNeeded bool

	// volumesToReportAsAttached is the list of volumes that should be reported
	// as attached in the Node's status (note that this may differ from the
	// actual list of attached volumes since volumes should be removed from this
	// list as soon a detach operation is considered, before the detach
	// operation is triggered).
	volumesToReportAsAttached map[v1.UniqueVolumeName]v1.UniqueVolumeName
}

func (asw *actualStateOfWorld) MarkVolumeAsAttached(
	uniqueName v1.UniqueVolumeName, volumeSpec *volume.Spec, nodeName types.NodeName, devicePath string) error {
	_, err := asw.AddVolumeNode(uniqueName, volumeSpec, nodeName, devicePath)
	return err
}

func (asw *actualStateOfWorld) MarkVolumeAsDetached(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) {
	asw.DeleteVolumeNode(volumeName, nodeName)
}

func (asw *actualStateOfWorld) RemoveVolumeFromReportAsAttached(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) error {
	asw.Lock()
	defer asw.Unlock()
	return asw.removeVolumeFromReportAsAttached(volumeName, nodeName)
}

func (asw *actualStateOfWorld) AddVolumeToReportAsAttached(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) {
	asw.Lock()
	defer asw.Unlock()
	asw.addVolumeToReportAsAttached(volumeName, nodeName)
}

func (asw *actualStateOfWorld) AddVolumeNode(
	uniqueName v1.UniqueVolumeName, volumeSpec *volume.Spec, nodeName types.NodeName, devicePath string) (v1.UniqueVolumeName, error) {
	asw.Lock()
	defer asw.Unlock()

	var volumeName v1.UniqueVolumeName
	if volumeSpec != nil {
		attachableVolumePlugin, err := asw.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
		if err != nil || attachableVolumePlugin == nil {
			return "", fmt.Errorf(
				"failed to get AttachablePlugin from volumeSpec for volume %q err=%v",
				volumeSpec.Name(),
				err)
		}

		volumeName, err = volumehelper.GetUniqueVolumeNameFromSpec(
			attachableVolumePlugin, volumeSpec)
		if err != nil {
			return "", fmt.Errorf(
				"failed to GetUniqueVolumeNameFromSpec for volumeSpec %q err=%v",
				volumeSpec.Name(),
				err)
		}
	} else {
		// volumeSpec is nil
		// This happens only on controller startup when reading the volumes from node
		// status; if the pods using the volume have been removed and are unreachable
		// the volumes should be detached immediately and the spec is not needed
		volumeName = uniqueName
	}

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists {
		volumeObj = attachedVolume{
			volumeName:      volumeName,
			spec:            volumeSpec,
			nodesAttachedTo: make(map[types.NodeName]nodeAttachedTo),
			devicePath:      devicePath,
		}
	} else {
		// If volume object already exists, it indicates that the information would be out of date.
		// Update the fields for volume object except the nodes attached to the volumes.
		volumeObj.devicePath = devicePath
		volumeObj.spec = volumeSpec
		glog.V(2).Infof("Volume %q is already added to attachedVolume list to node %q, update device path %q",
			volumeName,
			nodeName,
			devicePath)
	}
	asw.attachedVolumes[volumeName] = volumeObj

	_, nodeExists := volumeObj.nodesAttachedTo[nodeName]
	if !nodeExists {
		// Create object if it doesn't exist.
		volumeObj.nodesAttachedTo[nodeName] = nodeAttachedTo{
			nodeName:              nodeName,
			mountedByNode:         true, // Assume mounted, until proven otherwise
			mountedByNodeSetCount: 0,
			detachRequestedTime:   time.Time{},
		}
	} else {
		glog.V(5).Infof("Volume %q is already added to attachedVolume list to the node %q",
			volumeName,
			nodeName)
	}

	asw.addVolumeToReportAsAttached(volumeName, nodeName)
	return volumeName, nil
}

func (asw *actualStateOfWorld) SetVolumeMountedByNode(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName, mounted bool, forceUnmount bool) error {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, nodeObj, err := asw.getNodeAndVolume(volumeName, nodeName)
	if err != nil {
		return fmt.Errorf("Failed to SetVolumeMountedByNode with error: %v", err)
	}

	if mounted {
		// Increment set count
		nodeObj.mountedByNodeSetCount = nodeObj.mountedByNodeSetCount + 1
	} else {
		// Do not allow value to be reset unless it has been set at least once
		if nodeObj.mountedByNodeSetCount == 0 && !forceUnmount {
			return nil
		}
	}

	nodeObj.mountedByNode = mounted
	volumeObj.nodesAttachedTo[nodeName] = nodeObj
	glog.V(4).Infof("SetVolumeMountedByNode volume %v to the node %q mounted %t",
		volumeName,
		nodeName,
		mounted)
	return nil
}

func (asw *actualStateOfWorld) ResetDetachRequestTime(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, nodeObj, err := asw.getNodeAndVolume(volumeName, nodeName)
	if err != nil {
		glog.Errorf("Failed to ResetDetachRequestTime with error: %v", err)
		return
	}
	nodeObj.detachRequestedTime = time.Time{}
	volumeObj.nodesAttachedTo[nodeName] = nodeObj
}

func (asw *actualStateOfWorld) SetDetachRequestTime(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) (time.Duration, error) {
	asw.Lock()
	defer asw.Unlock()

	volumeObj, nodeObj, err := asw.getNodeAndVolume(volumeName, nodeName)
	if err != nil {
		return 0, fmt.Errorf("Failed to set detach request time with error: %v", err)
	}
	// If there is no previous detach request, set it to the current time
	if nodeObj.detachRequestedTime.IsZero() {
		nodeObj.detachRequestedTime = time.Now()
		volumeObj.nodesAttachedTo[nodeName] = nodeObj
		glog.V(4).Infof("Set detach request time to current time for volume %v on node %q",
			volumeName,
			nodeName)
	}
	return time.Since(nodeObj.detachRequestedTime), nil
}

// Get the volume and node object from actual state of world
// This is an internal function and caller should acquire and release the lock
//
// Note that this returns disconnected objects, so if you change the volume object you must set it back with
// `asw.attachedVolumes[volumeName]=volumeObj`.
//
// If you change the node object you must use `volumeObj.nodesAttachedTo[nodeName] = nodeObj`
// This is correct, because if volumeObj is empty this function returns an error, and nodesAttachedTo
// map is a reference type, and thus mutating the copy changes the original map.
func (asw *actualStateOfWorld) getNodeAndVolume(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) (attachedVolume, nodeAttachedTo, error) {

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if volumeExists {
		nodeObj, nodeExists := volumeObj.nodesAttachedTo[nodeName]
		if nodeExists {
			return volumeObj, nodeObj, nil
		}
	}

	return attachedVolume{}, nodeAttachedTo{}, fmt.Errorf("volume %v is no longer attached to the node %q",
		volumeName,
		nodeName)
}

// Remove the volumeName from the node's volumesToReportAsAttached list
// This is an internal function and caller should acquire and release the lock
func (asw *actualStateOfWorld) removeVolumeFromReportAsAttached(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) error {

	nodeToUpdate, nodeToUpdateExists := asw.nodesToUpdateStatusFor[nodeName]
	if nodeToUpdateExists {
		_, nodeToUpdateVolumeExists :=
			nodeToUpdate.volumesToReportAsAttached[volumeName]
		if nodeToUpdateVolumeExists {
			nodeToUpdate.statusUpdateNeeded = true
			delete(nodeToUpdate.volumesToReportAsAttached, volumeName)
			asw.nodesToUpdateStatusFor[nodeName] = nodeToUpdate
			return nil
		}
	}
	return fmt.Errorf("volume %q does not exist in volumesToReportAsAttached list or node %q does not exist in nodesToUpdateStatusFor list",
		volumeName,
		nodeName)

}

// Add the volumeName to the node's volumesToReportAsAttached list
// This is an internal function and caller should acquire and release the lock
func (asw *actualStateOfWorld) addVolumeToReportAsAttached(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) {
	// In case the volume/node entry is no longer in attachedVolume list, skip the rest
	if _, _, err := asw.getNodeAndVolume(volumeName, nodeName); err != nil {
		glog.V(4).Infof("Volume %q is no longer attached to node %q", volumeName, nodeName)
		return
	}
	nodeToUpdate, nodeToUpdateExists := asw.nodesToUpdateStatusFor[nodeName]
	if !nodeToUpdateExists {
		// Create object if it doesn't exist
		nodeToUpdate = nodeToUpdateStatusFor{
			nodeName:                  nodeName,
			statusUpdateNeeded:        true,
			volumesToReportAsAttached: make(map[v1.UniqueVolumeName]v1.UniqueVolumeName),
		}
		asw.nodesToUpdateStatusFor[nodeName] = nodeToUpdate
		glog.V(4).Infof("Add new node %q to nodesToUpdateStatusFor", nodeName)
	}
	_, nodeToUpdateVolumeExists :=
		nodeToUpdate.volumesToReportAsAttached[volumeName]
	if !nodeToUpdateVolumeExists {
		nodeToUpdate.statusUpdateNeeded = true
		nodeToUpdate.volumesToReportAsAttached[volumeName] = volumeName
		asw.nodesToUpdateStatusFor[nodeName] = nodeToUpdate
		glog.V(4).Infof("Report volume %q as attached to node %q", volumeName, nodeName)
	}
}

// Update the flag statusUpdateNeeded to indicate whether node status is already updated or
// needs to be updated again by the node status updater.
// If the specifed node does not exist in the nodesToUpdateStatusFor list, log the error and return
// This is an internal function and caller should acquire and release the lock
func (asw *actualStateOfWorld) updateNodeStatusUpdateNeeded(nodeName types.NodeName, needed bool) error {
	nodeToUpdate, nodeToUpdateExists := asw.nodesToUpdateStatusFor[nodeName]
	if !nodeToUpdateExists {
		// should not happen
		errMsg := fmt.Sprintf("Failed to set statusUpdateNeeded to needed %t because nodeName=%q  does not exist",
			needed, nodeName)
		glog.Errorf(errMsg)
		return fmt.Errorf(errMsg)
	}

	nodeToUpdate.statusUpdateNeeded = needed
	asw.nodesToUpdateStatusFor[nodeName] = nodeToUpdate

	return nil
}

func (asw *actualStateOfWorld) SetNodeStatusUpdateNeeded(nodeName types.NodeName) {
	asw.Lock()
	defer asw.Unlock()
	if err := asw.updateNodeStatusUpdateNeeded(nodeName, true); err != nil {
		glog.Errorf("Failed to update statusUpdateNeeded field in actual state of world: %v", err)
	}
}

func (asw *actualStateOfWorld) DeleteVolumeNode(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) {
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

	// Remove volume from volumes to report as attached
	asw.removeVolumeFromReportAsAttached(volumeName, nodeName)
}

func (asw *actualStateOfWorld) VolumeNodeExists(
	volumeName v1.UniqueVolumeName, nodeName types.NodeName) bool {
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
	for _, volumeObj := range asw.attachedVolumes {
		for _, nodeObj := range volumeObj.nodesAttachedTo {
			attachedVolumes = append(
				attachedVolumes,
				getAttachedVolume(&volumeObj, &nodeObj))
		}
	}

	return attachedVolumes
}

func (asw *actualStateOfWorld) GetAttachedVolumesForNode(
	nodeName types.NodeName) []AttachedVolume {
	asw.RLock()
	defer asw.RUnlock()

	attachedVolumes := make(
		[]AttachedVolume, 0 /* len */, len(asw.attachedVolumes) /* cap */)
	for _, volumeObj := range asw.attachedVolumes {
		for actualNodeName, nodeObj := range volumeObj.nodesAttachedTo {
			if actualNodeName == nodeName {
				attachedVolumes = append(
					attachedVolumes,
					getAttachedVolume(&volumeObj, &nodeObj))
			}
		}
	}

	return attachedVolumes
}

func (asw *actualStateOfWorld) GetAttachedVolumesPerNode() map[types.NodeName][]operationexecutor.AttachedVolume {
	asw.RLock()
	defer asw.RUnlock()

	attachedVolumesPerNode := make(map[types.NodeName][]operationexecutor.AttachedVolume)
	for _, volumeObj := range asw.attachedVolumes {
		for nodeName, nodeObj := range volumeObj.nodesAttachedTo {
			volumes := attachedVolumesPerNode[nodeName]
			volumes = append(volumes, getAttachedVolume(&volumeObj, &nodeObj).AttachedVolume)
			attachedVolumesPerNode[nodeName] = volumes
		}
	}

	return attachedVolumesPerNode
}

func (asw *actualStateOfWorld) GetNodesForVolume(volumeName v1.UniqueVolumeName) []types.NodeName {
	asw.RLock()
	defer asw.RUnlock()

	volumeObj, volumeExists := asw.attachedVolumes[volumeName]
	if !volumeExists || len(volumeObj.nodesAttachedTo) == 0 {
		return []types.NodeName{}
	}

	nodes := []types.NodeName{}
	for k := range volumeObj.nodesAttachedTo {
		nodes = append(nodes, k)
	}
	return nodes
}

func (asw *actualStateOfWorld) GetVolumesToReportAttached() map[types.NodeName][]v1.AttachedVolume {
	asw.RLock()
	defer asw.RUnlock()

	volumesToReportAttached := make(map[types.NodeName][]v1.AttachedVolume)
	for nodeName, nodeToUpdateObj := range asw.nodesToUpdateStatusFor {
		if nodeToUpdateObj.statusUpdateNeeded {
			attachedVolumes := make(
				[]v1.AttachedVolume,
				len(nodeToUpdateObj.volumesToReportAsAttached) /* len */)
			i := 0
			for _, volume := range nodeToUpdateObj.volumesToReportAsAttached {
				attachedVolumes[i] = v1.AttachedVolume{
					Name:       volume,
					DevicePath: asw.attachedVolumes[volume].devicePath,
				}
				i++
			}
			volumesToReportAttached[nodeToUpdateObj.nodeName] = attachedVolumes
		}
		// When GetVolumesToReportAttached is called by node status updater, the current status
		// of this node will be updated, so set the flag statusUpdateNeeded to false indicating
		// the current status is already updated.
		if err := asw.updateNodeStatusUpdateNeeded(nodeName, false); err != nil {
			glog.Errorf("Failed to update statusUpdateNeeded field when getting volumes: %v", err)
		}
	}

	return volumesToReportAttached
}

func (asw *actualStateOfWorld) GetNodesToUpdateStatusFor() map[types.NodeName]nodeToUpdateStatusFor {
	return asw.nodesToUpdateStatusFor
}

func getAttachedVolume(
	attachedVolume *attachedVolume,
	nodeAttachedTo *nodeAttachedTo) AttachedVolume {
	return AttachedVolume{
		AttachedVolume: operationexecutor.AttachedVolume{
			VolumeName:         attachedVolume.volumeName,
			VolumeSpec:         attachedVolume.spec,
			NodeName:           nodeAttachedTo.nodeName,
			DevicePath:         attachedVolume.devicePath,
			PluginIsAttachable: true,
		},
		MountedByNode:       nodeAttachedTo.mountedByNode,
		DetachRequestedTime: nodeAttachedTo.detachRequestedTime}
}
