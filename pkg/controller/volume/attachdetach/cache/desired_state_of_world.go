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

	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// DesiredStateOfWorld defines a set of thread-safe operations supported on
// the attach/detach controller's desired state of the world cache.
// This cache contains nodes->volumes->pods where nodes are all the nodes
// managed by the attach/detach controller, volumes are all the volumes that
// should be attached to the specified node, and pods are the pods that
// reference the volume and are scheduled to that node.
// Note: This is distinct from the DesiredStateOfWorld implemented by the
// kubelet volume manager. The both keep track of different objects. This
// contains attach/detach controller specific state.
type DesiredStateOfWorld interface {
	// AddNode adds the given node to the list of nodes managed by the attach/
	// detach controller.
	// If the node already exists this is a no-op.
	// keepTerminatedPodVolumes is a property of the node that determines
	// if for terminated pods volumes should be mounted and attached.
	AddNode(nodeName k8stypes.NodeName, keepTerminatedPodVolumes bool)

	// AddPod adds the given pod to the list of pods that reference the
	// specified volume and is scheduled to the specified node.
	// A unique volumeName is generated from the volumeSpec and returned on
	// success.
	// If the pod already exists under the specified volume, this is a no-op.
	// If volumeSpec is not an attachable volume plugin, an error is returned.
	// If no volume with the name volumeName exists in the list of volumes that
	// should be attached to the specified node, the volume is implicitly added.
	// If no node with the name nodeName exists in list of nodes managed by the
	// attach/detach attached controller, an error is returned.
	AddPod(podName types.UniquePodName, pod *v1.Pod, volumeSpec *volume.Spec, nodeName k8stypes.NodeName) (v1.UniqueVolumeName, error)

	// DeleteNode removes the given node from the list of nodes managed by the
	// attach/detach controller.
	// If the node does not exist this is a no-op.
	// If the node exists but has 1 or more child volumes, an error is returned.
	DeleteNode(nodeName k8stypes.NodeName) error

	// DeletePod removes the given pod from the list of pods that reference the
	// specified volume and are scheduled to the specified node.
	// If no pod exists in the list of pods that reference the specified volume
	// and are scheduled to the specified node, this is a no-op.
	// If a node with the name nodeName does not exist in the list of nodes
	// managed by the attach/detach attached controller, this is a no-op.
	// If no volume with the name volumeName exists in the list of managed
	// volumes under the specified node, this is a no-op.
	// If after deleting the pod, the specified volume contains no other child
	// pods, the volume is also deleted.
	DeletePod(podName types.UniquePodName, volumeName v1.UniqueVolumeName, nodeName k8stypes.NodeName)

	// NodeExists returns true if the node with the specified name exists in
	// the list of nodes managed by the attach/detach controller.
	NodeExists(nodeName k8stypes.NodeName) bool

	// VolumeExists returns true if the volume with the specified name exists
	// in the list of volumes that should be attached to the specified node by
	// the attach detach controller.
	VolumeExists(volumeName v1.UniqueVolumeName, nodeName k8stypes.NodeName) bool

	// GetVolumesToAttach generates and returns a list of volumes to attach
	// and the nodes they should be attached to based on the current desired
	// state of the world.
	GetVolumesToAttach() []VolumeToAttach

	// GetPodToAdd generates and returns a map of pods based on the current desired
	// state of world
	GetPodToAdd() map[types.UniquePodName]PodToAdd

	// GetKeepTerminatedPodVolumesForNode determines if node wants volumes to be
	// mounted and attached for terminated pods
	GetKeepTerminatedPodVolumesForNode(k8stypes.NodeName) bool
}

// VolumeToAttach represents a volume that should be attached to a node.
type VolumeToAttach struct {
	operationexecutor.VolumeToAttach
}

// PodToAdd represents a pod that references the underlying volume and is
// scheduled to the underlying node.
type PodToAdd struct {
	// pod contains the api object of pod
	Pod *v1.Pod

	// volumeName contains the unique identifier for this volume.
	VolumeName v1.UniqueVolumeName

	// nodeName contains the name of this node.
	NodeName k8stypes.NodeName
}

// NewDesiredStateOfWorld returns a new instance of DesiredStateOfWorld.
func NewDesiredStateOfWorld(volumePluginMgr *volume.VolumePluginMgr) DesiredStateOfWorld {
	return &desiredStateOfWorld{
		nodesManaged:    make(map[k8stypes.NodeName]nodeManaged),
		volumePluginMgr: volumePluginMgr,
	}
}

type desiredStateOfWorld struct {
	// nodesManaged is a map containing the set of nodes managed by the attach/
	// detach controller. The key in this map is the name of the node and the
	// value is a node object containing more information about the node.
	nodesManaged map[k8stypes.NodeName]nodeManaged
	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr
	sync.RWMutex
}

// nodeManaged represents a node that is being managed by the attach/detach
// controller.
type nodeManaged struct {
	// nodeName contains the name of this node.
	nodeName k8stypes.NodeName

	// volumesToAttach is a map containing the set of volumes that should be
	// attached to this node. The key in the map is the name of the volume and
	// the value is a pod object containing more information about the volume.
	volumesToAttach map[v1.UniqueVolumeName]volumeToAttach

	// keepTerminatedPodVolumes determines if for terminated pods(on this node) - volumes
	// should be kept mounted and attached.
	keepTerminatedPodVolumes bool
}

// The volume object represents a volume that should be attached to a node.
type volumeToAttach struct {
	// multiAttachErrorReported indicates whether the multi-attach error has been reported for the given volume.
	// It is used to to prevent reporting the error from being reported more than once for a given volume.
	multiAttachErrorReported bool

	// volumeName contains the unique identifier for this volume.
	volumeName v1.UniqueVolumeName

	// spec is the volume spec containing the specification for this volume.
	// Used to generate the volume plugin object, and passed to attach/detach
	// methods.
	spec *volume.Spec

	// scheduledPods is a map containing the set of pods that reference this
	// volume and are scheduled to the underlying node. The key in the map is
	// the name of the pod and the value is a pod object containing more
	// information about the pod.
	scheduledPods map[types.UniquePodName]pod
}

// The pod represents a pod that references the underlying volume and is
// scheduled to the underlying node.
type pod struct {
	// podName contains the unique identifier for this pod
	podName types.UniquePodName

	// pod object contains the api object of pod
	podObj *v1.Pod
}

func (dsw *desiredStateOfWorld) AddNode(nodeName k8stypes.NodeName, keepTerminatedPodVolumes bool) {
	dsw.Lock()
	defer dsw.Unlock()

	if _, nodeExists := dsw.nodesManaged[nodeName]; !nodeExists {
		dsw.nodesManaged[nodeName] = nodeManaged{
			nodeName:                 nodeName,
			volumesToAttach:          make(map[v1.UniqueVolumeName]volumeToAttach),
			keepTerminatedPodVolumes: keepTerminatedPodVolumes,
		}
	}
}

func (dsw *desiredStateOfWorld) AddPod(
	podName types.UniquePodName,
	podToAdd *v1.Pod,
	volumeSpec *volume.Spec,
	nodeName k8stypes.NodeName) (v1.UniqueVolumeName, error) {
	dsw.Lock()
	defer dsw.Unlock()

	nodeObj, nodeExists := dsw.nodesManaged[nodeName]
	if !nodeExists {
		return "", fmt.Errorf(
			"no node with the name %q exists in the list of managed nodes",
			nodeName)
	}

	attachableVolumePlugin, err := dsw.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return "", fmt.Errorf(
			"failed to get AttachablePlugin from volumeSpec for volume %q err=%v",
			volumeSpec.Name(),
			err)
	}

	volumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(
		attachableVolumePlugin, volumeSpec)
	if err != nil {
		return "", fmt.Errorf(
			"failed to GetUniqueVolumeNameFromSpec for volumeSpec %q err=%v",
			volumeSpec.Name(),
			err)
	}

	volumeObj, volumeExists := nodeObj.volumesToAttach[volumeName]
	if !volumeExists {
		volumeObj = volumeToAttach{
			multiAttachErrorReported: false,
			volumeName:               volumeName,
			spec:                     volumeSpec,
			scheduledPods:            make(map[types.UniquePodName]pod),
		}
		dsw.nodesManaged[nodeName].volumesToAttach[volumeName] = volumeObj
	}
	if _, podExists := volumeObj.scheduledPods[podName]; !podExists {
		dsw.nodesManaged[nodeName].volumesToAttach[volumeName].scheduledPods[podName] =
			pod{
				podName: podName,
				podObj:  podToAdd,
			}
	}

	return volumeName, nil
}

func (dsw *desiredStateOfWorld) DeleteNode(nodeName k8stypes.NodeName) error {
	dsw.Lock()
	defer dsw.Unlock()

	nodeObj, nodeExists := dsw.nodesManaged[nodeName]
	if !nodeExists {
		return nil
	}

	if len(nodeObj.volumesToAttach) > 0 {
		return fmt.Errorf(
			"failed to delete node %q from list of nodes managed by attach/detach controller--the node still contains %v volumes in its list of volumes to attach",
			nodeName,
			len(nodeObj.volumesToAttach))
	}

	delete(
		dsw.nodesManaged,
		nodeName)
	return nil
}

func (dsw *desiredStateOfWorld) DeletePod(
	podName types.UniquePodName,
	volumeName v1.UniqueVolumeName,
	nodeName k8stypes.NodeName) {
	dsw.Lock()
	defer dsw.Unlock()

	nodeObj, nodeExists := dsw.nodesManaged[nodeName]
	if !nodeExists {
		return
	}

	volumeObj, volumeExists := nodeObj.volumesToAttach[volumeName]
	if !volumeExists {
		return
	}
	if _, podExists := volumeObj.scheduledPods[podName]; !podExists {
		return
	}

	delete(
		dsw.nodesManaged[nodeName].volumesToAttach[volumeName].scheduledPods,
		podName)

	if len(volumeObj.scheduledPods) == 0 {
		delete(
			dsw.nodesManaged[nodeName].volumesToAttach,
			volumeName)
	}
}

func (dsw *desiredStateOfWorld) NodeExists(nodeName k8stypes.NodeName) bool {
	dsw.RLock()
	defer dsw.RUnlock()

	_, nodeExists := dsw.nodesManaged[nodeName]
	return nodeExists
}

func (dsw *desiredStateOfWorld) VolumeExists(
	volumeName v1.UniqueVolumeName, nodeName k8stypes.NodeName) bool {
	dsw.RLock()
	defer dsw.RUnlock()

	nodeObj, nodeExists := dsw.nodesManaged[nodeName]
	if nodeExists {
		if _, volumeExists := nodeObj.volumesToAttach[volumeName]; volumeExists {
			return true
		}
	}

	return false
}

// GetKeepTerminatedPodVolumesForNode determines if node wants volumes to be
// mounted and attached for terminated pods
func (dsw *desiredStateOfWorld) GetKeepTerminatedPodVolumesForNode(nodeName k8stypes.NodeName) bool {
	dsw.RLock()
	defer dsw.RUnlock()

	if nodeName == "" {
		return false
	}
	if node, ok := dsw.nodesManaged[nodeName]; ok {
		return node.keepTerminatedPodVolumes
	}
	return false
}

func (dsw *desiredStateOfWorld) GetVolumesToAttach() []VolumeToAttach {
	dsw.RLock()
	defer dsw.RUnlock()

	volumesToAttach := make([]VolumeToAttach, 0 /* len */, len(dsw.nodesManaged) /* cap */)
	for nodeName, nodeObj := range dsw.nodesManaged {
		for volumeName, volumeObj := range nodeObj.volumesToAttach {
			volumesToAttach = append(volumesToAttach,
				VolumeToAttach{
					VolumeToAttach: operationexecutor.VolumeToAttach{
						MultiAttachErrorReported: volumeObj.multiAttachErrorReported,
						VolumeName:               volumeName,
						VolumeSpec:               volumeObj.spec,
						NodeName:                 nodeName,
						ScheduledPods:            getPodsFromMap(volumeObj.scheduledPods),
					}})
		}
	}

	return volumesToAttach
}

// Construct a list of v1.Pod objects from the given pod map
func getPodsFromMap(podMap map[types.UniquePodName]pod) []*v1.Pod {
	pods := make([]*v1.Pod, 0, len(podMap))
	for _, pod := range podMap {
		pods = append(pods, pod.podObj)
	}
	return pods
}

func (dsw *desiredStateOfWorld) GetPodToAdd() map[types.UniquePodName]PodToAdd {
	dsw.RLock()
	defer dsw.RUnlock()

	pods := make(map[types.UniquePodName]PodToAdd)
	for nodeName, nodeObj := range dsw.nodesManaged {
		for volumeName, volumeObj := range nodeObj.volumesToAttach {
			for podUID, pod := range volumeObj.scheduledPods {
				pods[podUID] = PodToAdd{
					Pod:        pod.podObj,
					VolumeName: volumeName,
					NodeName:   nodeName,
				}
			}
		}
	}
	return pods
}
