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

// Package populator implements interfaces that monitor and keep the states of the
// desired_state_of_word in sync with the "ground truth" from informer.
package populator

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/v1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/util"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// DesiredStateOfWorldPopulator periodically verifies that the pods in the
// desired state of th world still exist, if not, it removes them.
type DesiredStateOfWorldPopulator interface {
	Run(stopCh <-chan struct{})
}

// NewDesiredStateOfWorldPopulator returns a new instance of DesiredStateOfWorldPopulator.
// loopSleepDuration - the amount of time the populator loop sleeps between
//     successive executions
// podManager - the kubelet podManager that is the source of truth for the pods
//     that exist on this host
// desiredStateOfWorld - the cache to populate
func NewDesiredStateOfWorldPopulator(
	loopSleepDuration time.Duration,
	podLister corelisters.PodLister,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	volumePluginMgr *volume.VolumePluginMgr,
	pvcLister corelisters.PersistentVolumeClaimLister,
	pvLister corelisters.PersistentVolumeLister) DesiredStateOfWorldPopulator {
	return &desiredStateOfWorldPopulator{
		loopSleepDuration:   loopSleepDuration,
		podLister:           podLister,
		desiredStateOfWorld: desiredStateOfWorld,
		volumePluginMgr:     volumePluginMgr,
		pvcLister:           pvcLister,
		pvLister:            pvLister,
	}
}

type desiredStateOfWorldPopulator struct {
	loopSleepDuration   time.Duration
	podLister           corelisters.PodLister
	desiredStateOfWorld cache.DesiredStateOfWorld
	volumePluginMgr     *volume.VolumePluginMgr
	pvcLister           corelisters.PersistentVolumeClaimLister
	pvLister            corelisters.PersistentVolumeLister
}

func (dswp *desiredStateOfWorldPopulator) Run(stopCh <-chan struct{}) {
	wait.Until(dswp.populatorLoopFunc(), dswp.loopSleepDuration, stopCh)
}

func (dswp *desiredStateOfWorldPopulator) populatorLoopFunc() func() {
	return func() {
		dswp.findAndRemoveDeletedPods()
		dswp.findAndAddActivePods()
	}
}

// Iterate through all pods in desired state of world, and remove if they no
// longer exist in the informer
func (dswp *desiredStateOfWorldPopulator) findAndRemoveDeletedPods() {
	for dswPodUID, dswPodToAdd := range dswp.desiredStateOfWorld.GetPodToAdd() {
		dswPodKey, err := kcache.MetaNamespaceKeyFunc(dswPodToAdd.Pod)
		if err != nil {
			glog.Errorf("MetaNamespaceKeyFunc failed for pod %q (UID %q) with: %v", dswPodKey, dswPodUID, err)
			continue
		}

		// Retrieve the pod object from pod informer with the namespace key
		namespace, name, err := kcache.SplitMetaNamespaceKey(dswPodKey)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("error splitting dswPodKey %q: %v", dswPodKey, err))
			continue
		}
		informerPod, err := dswp.podLister.Pods(namespace).Get(name)
		switch {
		case errors.IsNotFound(err):
			// if we can't find the pod, we need to delete it below
		case err != nil:
			glog.Errorf("podLister Get failed for pod %q (UID %q) with %v", dswPodKey, dswPodUID, err)
			continue
		default:
			informerPodUID := volumehelper.GetUniquePodName(informerPod)
			// Check whether the unique identifier of the pod from dsw matches the one retrieved from pod informer
			if informerPodUID == dswPodUID {
				glog.V(10).Infof("Verified pod %q (UID %q) from dsw exists in pod informer.", dswPodKey, dswPodUID)
				continue
			}
		}

		// the pod from dsw does not exist in pod informer, or it does not match the unique identifer retrieved
		// from the informer, delete it from dsw
		glog.V(1).Infof("Removing pod %q (UID %q) from dsw because it does not exist in pod informer.", dswPodKey, dswPodUID)
		dswp.desiredStateOfWorld.DeletePod(dswPodUID, dswPodToAdd.VolumeName, dswPodToAdd.NodeName)
	}
}

func (dswp *desiredStateOfWorldPopulator) isPodTerminated(pod *v1.Pod) bool {
	return pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded || (pod.DeletionTimestamp != nil && notRunning(pod.Status.ContainerStatuses))
}

// notRunning returns true if every status is terminated or waiting, or the status list
// is empty.
func notRunning(statuses []v1.ContainerStatus) bool {
	for _, status := range statuses {
		if status.State.Terminated == nil && status.State.Waiting == nil {
			return false
		}
	}
	return true
}

func (dswp *desiredStateOfWorldPopulator) findAndAddActivePods() {
	pods, err := dswp.podLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("podLister List failed for pods %v", err)
		return
	}
	for _, pod := range pods {
		if dswp.isPodTerminated(pod) {
			// Do not add volumes for terminated pods
			continue
		}

		nodeName := types.NodeName(pod.Spec.NodeName)
		if nodeName == "" {
			glog.V(10).Infof(
				"Skipping processing of pod %q/%q: it is not scheduled to a node.",
				pod.Namespace,
				pod.Name)
			continue
		} else if !dswp.desiredStateOfWorld.NodeExists(nodeName) {
			// If the node the pod is scheduled to does not exist in the desired
			// state of the world data structure, that indicates the node is not
			// yet managed by the controller. Therefore, ignore the pod.
			glog.V(10).Infof(
				"Skipping processing of pod %q/%q: it is scheduled to node %q which is not managed by the controller.",
				pod.Namespace,
				pod.Name,
				nodeName)
			continue
		}
		// check if the pod has volumes
		if len(pod.Spec.Volumes) <= 0 {
			glog.V(10).Infof("Skipping processing of pod %q/%q: it has no volumes.",
				pod.Namespace,
				pod.Name)
		} else {
			for _, podVolume := range pod.Spec.Volumes {
				uniquePodName := volumehelper.GetUniquePodName(pod)
				volumeSpec, err := util.CreateVolumeSpec(podVolume, pod.Namespace, dswp.pvcLister, dswp.pvLister)
				if err != nil {
					glog.V(10).Infof(
						"Error processing volume %q for pod %q/%q: %v",
						podVolume.Name,
						pod.Namespace,
						pod.Name,
						err)
					continue
				}

				attachableVolumePlugin, err := dswp.volumePluginMgr.FindAttachablePluginBySpec(volumeSpec)
				if err != nil || attachableVolumePlugin == nil {
					glog.V(10).Infof(
						"failed to get AttachablePlugin from volumeSpec for volume %q err: %v",
						volumeSpec.Name(),
						err)
					continue
				}
				volumeName, err := volumehelper.GetUniqueVolumeNameFromSpec(
					attachableVolumePlugin, volumeSpec)
				if err != nil {
					glog.V(10).Infof(
						"failed to GetUniqueVolumeNameFromSpec for volumeSpec %q err: %v",
						volumeSpec.Name(),
						err)
					continue
				}

				// check if the pod and its volume is added to dsw
				added := dswp.desiredStateOfWorld.PodExists(uniquePodName, volumeName, nodeName)
				if added {
					glog.V(10).Infof("Pod references the specified volume is already added to dsw")
					continue
				}

				dswp.desiredStateOfWorld.AddPod(uniquePodName, pod, volumeSpec, nodeName)
			}
		}
	}

}
