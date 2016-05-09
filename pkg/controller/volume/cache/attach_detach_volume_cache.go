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
Package cache implements a data structure used by the attach/detach controller
to keep track of volumes, the nodes they are attached to, and the pods that
reference them. It is thread-safe.
*/
package cache

import (
	"fmt"
	"sync"
)

// AttachDetachVolumeCache defines the set of operations the volume cache
// supports.
type AttachDetachVolumeCache interface {
	// AddVolume adds the given volume to the list of volumes managed by the
	// attach detach controller.
	// If the volume already exists, this is a no-op.
	AddVolume(volumeName string)

	// AddNode adds the given node to the list of nodes the specified volume is
	// attached to.
	// If no volume with the name volumeName exists in the list of managed
	// volumes, an error is returned.
	// If the node already exists for the specified volume, this is a no-op.
	AddNode(nodeName, volumeName string) error

	// AddPod adds the given pod to the list of pods that are scheduled to
	// the specified node and referencing the specified volume.
	// If no node with the name nodeName exists in the list of attached nodes,
	// an error is returned.
	// If no volume with the name volumeName exists in the list of managed
	// volumes, an error is returned.
	// If the pod already exists for the specified volume, this is a no-op.
	AddPod(podName, nodeName, volumeName string) error

	// DeleteVolume removes the given volume from the list of volumes managed
	// by the attach detach controller.
	// If no volume with the name volumeName exists in the list of managed
	// volumes, an error is returned.
	// All attachedNodes must be deleted from the volume before it is deleted.
	// If the specified volume contains 1 or more attachedNodes, an error is
	// returned.
	DeleteVolume(volumeName string) error

	// DeleteNode removes the given node from the list of nodes the specified
	// volume is attached to.
	// If no node with the name nodeName exists in the list of attached nodes,
	// an error is returned.
	// If no volume with the name volumeName exists in the list of managed
	// volumes, an error is returned.
	// All scheduledPods must be deleted from the node before it is deleted.
	// If the specified node contains 1 or more scheduledPods, an error is
	// returned.
	DeleteNode(nodeName, volumeName string) error

	// DeletePod removes the given pod from the list of pods that are scheduled
	// to the specified node and referencing the specified volume.
	// If no pod with the name podName exists for the specified volume/node, an
	// error is returned.
	// If no node with the name nodeName exists in the list of attached nodes,
	// an error is returned.
	// If no volume with the name volumeName exists in the list of managed
	// volumes, an error is returned.
	DeletePod(podName, nodeName, volumeName string) error

	// VolumeExists returns true if the volume with the specified name exists
	// in the list of volumes managed by the attach detach controller.
	VolumeExists(volumeName string) bool

	// NodeExists returns true if the node with the specified name exists in
	// the list of nodes the specified volume is attached to.
	// If no volume with the name volumeName exists in the list of managed
	// volumes, an error is returned.
	NodeExists(nodeName, volumeName string) (bool, error)

	// PodExists returns true if the pod with the specified name exists in the
	// list of pods that are scheduled to the specified node and referencing
	// the specified volume.
	// If no node with the name nodeName exists in the list of attached nodes,
	// an error is returned.
	// If no volume with the name volumeName exists in the list of managed
	// volumes, an error is returned.
	PodExists(podName, nodeName, volumeName string) (bool, error)
}

// NewAttachDetachVolumeCache returns a new instance of the
// AttachDetachVolumeCache.
func NewAttachDetachVolumeCache() AttachDetachVolumeCache {
	return &attachDetachVolumeCache{
		volumesManaged: make(map[string]volume),
	}
}

type attachDetachVolumeCache struct {
	// volumesManaged is a map containing the set of volumes managed by the
	// attach/detach controller. The key in this map is the name of the unique
	// volume identifier and the value is a volume object containing more
	// information about the volume.
	volumesManaged map[string]volume
	sync.RWMutex
}

// The volume object represents a volume that is being tracked by the attach
// detach controller.
type volume struct {
	// name contains the unique identifer for this volume.
	name string

	// attachedNodes is a map containing the set of nodes this volume has
	// succesfully been attached to. The key in this map is the name of the
	// node and the value is a node object containing more information about
	// the node.
	attachedNodes map[string]node
}

// The node object represents a node that a volume is attached to.
type node struct {
	// name contains the name of this node.
	name string

	// scheduledPods is a map containing the set of pods that are scheduled to
	// this node and referencing the underlying volume. The key in the map is
	// the name of the pod and the value is a pod object containing more
	// information about the pod.
	scheduledPods map[string]pod
}

// The pod object represents a pod that is scheduled to a node and referncing
// the underlying volume.
type pod struct {
	// name contains the name of this pod.
	name string
}

// AddVolume adds the given volume to the list of volumes managed by the attach
// detach controller.
// If the volume already exists, this is a no-op.
func (vc *attachDetachVolumeCache) AddVolume(volumeName string) {
	vc.Lock()
	defer vc.Unlock()
	if _, exists := vc.volumesManaged[volumeName]; !exists {
		vc.volumesManaged[volumeName] = volume{
			name:          volumeName,
			attachedNodes: make(map[string]node),
		}
	}
}

// AddNode adds the given node to the list of nodes the specified volume is
// attached to.
// If no volume with the name volumeName exists in the list of managed volumes,
// an error is returned.
// If the node already exists for the specified volume, this is a no-op.
func (vc *attachDetachVolumeCache) AddNode(nodeName, volumeName string) error {
	vc.Lock()
	defer vc.Unlock()

	vol, volExists := vc.volumesManaged[volumeName]
	if !volExists {
		return fmt.Errorf(
			"failed to add node %q to volume %q--no volume with that name exists in the list of managed volumes",
			nodeName,
			volumeName)
	}

	if _, nodeExists := vol.attachedNodes[nodeName]; !nodeExists {
		vc.volumesManaged[volumeName].attachedNodes[nodeName] = node{
			name:          nodeName,
			scheduledPods: make(map[string]pod),
		}
	}

	return nil
}

// AddPod adds the given pod to the list of pods that are scheduled to the
// specified node and referencing the specified volume.
// If no node with the name nodeName exists in the list of attached nodes,
// an error is returned.
// If no volume with the name volumeName exists in the list of managed
// volumes, an error is returned.
// If the pod already exists for the specified volume, this is a no-op.
func (vc *attachDetachVolumeCache) AddPod(podName, nodeName, volumeName string) error {
	vc.Lock()
	defer vc.Unlock()

	volObj, volExists := vc.volumesManaged[volumeName]
	if !volExists {
		return fmt.Errorf(
			"failed to add pod %q to node %q volume %q--no volume with that name exists in the list of managed volumes",
			podName,
			nodeName,
			volumeName)
	}

	nodeObj, nodeExists := volObj.attachedNodes[nodeName]
	if !nodeExists {
		return fmt.Errorf(
			"failed to add pod %q to node %q volume %q--no node with that name exists in the list of attached nodes for that volume",
			podName,
			nodeName,
			volumeName)
	}

	if _, podExists := nodeObj.scheduledPods[podName]; !podExists {
		vc.volumesManaged[volumeName].attachedNodes[nodeName].scheduledPods[podName] =
			pod{
				name: podName,
			}
	}

	return nil
}

// DeleteVolume removes the given volume from the list of volumes managed by
// the attach detach controller.
// If no volume with the name volumeName exists in the list of managed volumes,
// an error is returned.
// All attachedNodes must be deleted from the volume before it is deleted.
// If the specified volume contains 1 or more attachedNodes, an error is
// returned.
func (vc *attachDetachVolumeCache) DeleteVolume(volumeName string) error {
	vc.Lock()
	defer vc.Unlock()

	volObj, volExists := vc.volumesManaged[volumeName]
	if !volExists {
		return fmt.Errorf(
			"failed to delete volume %q--no volume with that name exists in the list of managed volumes",
			volumeName)
	}

	if len(volObj.attachedNodes) > 0 {
		return fmt.Errorf(
			"failed to remove volume %q from list of managed volumes--the volume still contains %v nodes in its list of attached nodes",
			volumeName,
			len(volObj.attachedNodes))
	}

	delete(
		vc.volumesManaged,
		volumeName)
	return nil
}

// DeleteNode removes the given node from the list of nodes the specified
// volume is attached to.
// If no node with the name nodeName exists in the list of attached nodes, an
// error is returned.
// If no volume with the name volumeName exists in the list of managed
// volumes, an error is returned.
// All scheduledPods must be deleted from the node before it is deleted.
// If the specified node contains 1 or more scheduledPods, an error is
// returned.
func (vc *attachDetachVolumeCache) DeleteNode(nodeName, volumeName string) error {
	vc.Lock()
	defer vc.Unlock()

	volObj, volExists := vc.volumesManaged[volumeName]
	if !volExists {
		return fmt.Errorf(
			"failed to delete node %q from volume %q--no volume with that name exists in the list of managed volumes",
			nodeName,
			volumeName)
	}

	nodeObj, nodeExists := volObj.attachedNodes[nodeName]
	if !nodeExists {
		return fmt.Errorf(
			"failed to delete node %q from volume %q--no node with the that name exists in the list of attached nodes for that volume",
			nodeName,
			volumeName)
	}

	if len(nodeObj.scheduledPods) > 0 {
		return fmt.Errorf(
			"failed to remove node %q from volume %q--the node still contains %v pods in its list of scheduled pods",
			nodeName,
			volumeName,
			len(nodeObj.scheduledPods))
	}

	delete(
		vc.volumesManaged[volumeName].attachedNodes,
		nodeName)
	return nil
}

// DeletePod removes the given pod from the list of pods that are scheduled
// to the specified node and referencing the specified volume.
// If no pod with the name podName exists for the specified volume/node, an
// error is returned.
// If no node with the name nodeName exists in the list of attached nodes,
// an error is returned.
// If no volume with the name volumeName exists in the list of managed
// volumes, an error is returned.
func (vc *attachDetachVolumeCache) DeletePod(podName, nodeName, volumeName string) error {
	vc.Lock()
	defer vc.Unlock()

	volObj, volExists := vc.volumesManaged[volumeName]
	if !volExists {
		return fmt.Errorf(
			"failed to delete pod %q from node %q volume %q--no volume with that name exists in the list of managed volumes",
			podName,
			nodeName,
			volumeName)
	}

	nodeObj, nodeExists := volObj.attachedNodes[nodeName]
	if !nodeExists {
		return fmt.Errorf(
			"failed to delete pod %q from node %q volume %q--no node with that name exists in the list of attached nodes for that volume",
			podName,
			nodeName,
			volumeName)
	}

	if _, podExists := nodeObj.scheduledPods[podName]; !podExists {
		return fmt.Errorf(
			"failed to delete pod %q from node %q volume %q--no pod with that name exists in the list of scheduled pods under that node/volume",
			podName,
			nodeName,
			volumeName)
	}

	delete(
		vc.volumesManaged[volumeName].attachedNodes[nodeName].scheduledPods,
		podName)
	return nil
}

// VolumeExists returns true if the volume with the specified name exists in
// the list of volumes managed by the attach detach controller.
func (vc *attachDetachVolumeCache) VolumeExists(volumeName string) bool {
	vc.RLock()
	defer vc.RUnlock()

	_, volExists := vc.volumesManaged[volumeName]
	return volExists
}

// NodeExists returns true if the node with the specified name exists in the
// list of nodes the specified volume is attached to.
// If no volume with the name volumeName exists in the list of managed
// volumes, an error is returned.
func (vc *attachDetachVolumeCache) NodeExists(nodeName, volumeName string) (bool, error) {
	vc.RLock()
	defer vc.RUnlock()

	volObj, volExists := vc.volumesManaged[volumeName]
	if !volExists {
		return false,
			fmt.Errorf(
				"failed to check if node %q exists under volume %q--no volume with that name exists in the list of managed volumes",
				nodeName,
				volumeName)
	}

	_, nodeExists := volObj.attachedNodes[nodeName]
	return nodeExists, nil
}

// PodExists returns true if the pod with the specified name exists in the list
// of pods that are scheduled to the specified node and referencing the
// specified volume.
// If no node with the name nodeName exists in the list of attached nodes, an
// error is returned.
// If no volume with the name volumeName exists in the list of managed volumes,
// an error is returned.
func (vc *attachDetachVolumeCache) PodExists(podName, nodeName, volumeName string) (bool, error) {
	vc.RLock()
	defer vc.RUnlock()

	volObj, volExists := vc.volumesManaged[volumeName]
	if !volExists {
		return false,
			fmt.Errorf(
				"failed to check if node %q exists under volume %q--no volume with that name exists in the list of managed volumes",
				nodeName,
				volumeName)
	}

	nodeObj, nodeExists := volObj.attachedNodes[nodeName]
	if !nodeExists {
		return false, fmt.Errorf(
			"failed to check if pod %q exists under node %q volume %q--no node with that name exists in the list of attached nodes for that volume",
			podName,
			nodeName,
			volumeName)
	}

	_, podExists := nodeObj.scheduledPods[podName]
	return podExists, nil
}
