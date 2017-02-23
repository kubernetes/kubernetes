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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
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
	kubeClient clientset.Interface,
	loopSleepDuration time.Duration,
	podLister corelisters.PodLister,
	desiredStateOfWorld cache.DesiredStateOfWorld) DesiredStateOfWorldPopulator {
	return &desiredStateOfWorldPopulator{
		kubeClient:          kubeClient,
		loopSleepDuration:   loopSleepDuration,
		podLister:           podLister,
		desiredStateOfWorld: desiredStateOfWorld,
	}
}

type desiredStateOfWorldPopulator struct {
	kubeClient          clientset.Interface
	loopSleepDuration   time.Duration
	podLister           corelisters.PodLister
	desiredStateOfWorld cache.DesiredStateOfWorld
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

func (dswp *desiredStateOfWorldPopulator) isPodVolumePreviouslyAdded(podName volumetypes.UniquePodName, volumeSpec *volume.Spec, nodeName types.NodeName) (bool, error) {
	return dswp.desiredStateOfWorld.IsPodVolumePreviouslyAdded(podName, volumeSpec, nodeName)
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
				volumeSpec, err := dswp.createVolumeSpec(podVolume, pod.Namespace)
				if err != nil {
					glog.V(10).Infof(
						"Error processing volume %q for pod %q/%q: %v",
						podVolume.Name,
						pod.Namespace,
						pod.Name,
						err)
					continue
				}

				// check if the pod and its volume is added to dsw
				added, err := dswp.isPodVolumePreviouslyAdded(uniquePodName, volumeSpec, nodeName)
				if err != nil {
					glog.V(10).Infof("Check whether pod references the specified volume is added to dsw falied: %v", err)
					continue
				} else if added {
					glog.V(10).Infof("Pod references the specified volume is already added to dsw")
					continue
				}

				dswp.desiredStateOfWorld.AddPod(uniquePodName, pod, volumeSpec, nodeName)
			}
		}
	}

}

// createVolumeSpec creates and returns a volume.Spec object for the
// specified volume. It dereference any PVC to get PV objects, if needed.
func (dswp *desiredStateOfWorldPopulator) createVolumeSpec(podVolume v1.Volume, podNamespace string) (*volume.Spec, error) {
	if pvcSource :=
		podVolume.VolumeSource.PersistentVolumeClaim; pvcSource != nil {
		glog.V(10).Infof(
			"Found PVC, ClaimName: %q/%q",
			podNamespace,
			pvcSource.ClaimName)

		// If podVolume is a PVC, fetch the real PV behind the claim
		pvName, pvcUID, err := dswp.getPVCExtractPV(
			podNamespace, pvcSource.ClaimName)
		if err != nil {
			return nil, fmt.Errorf(
				"error processing PVC %q/%q: %v",
				podNamespace,
				pvcSource.ClaimName,
				err)
		}

		glog.V(10).Infof(
			"Found bound PV for PVC (ClaimName %q/%q pvcUID %v): pvName=%q",
			podNamespace,
			pvcSource.ClaimName,
			pvcUID,
			pvName)

		// Fetch actual PV object
		volumeSpec, err :=
			dswp.getPVSpec(pvName, pvcSource.ReadOnly, pvcUID)
		if err != nil {
			return nil, fmt.Errorf(
				"error processing PVC %q/%q: %v",
				podNamespace,
				pvcSource.ClaimName,
				err)
		}

		glog.V(10).Infof(
			"Extracted volumeSpec (%v) from bound PV (pvName %q) and PVC (ClaimName %q/%q pvcUID %v)",
			volumeSpec.Name,
			pvName,
			podNamespace,
			pvcSource.ClaimName,
			pvcUID)

		return volumeSpec, nil
	}

	// Do not return the original volume object, since the source could mutate it
	clonedPodVolumeObj, err := api.Scheme.DeepCopy(&podVolume)
	if err != nil || clonedPodVolumeObj == nil {
		return nil, fmt.Errorf(
			"failed to deep copy %q volume object. err=%v", podVolume.Name, err)
	}

	clonedPodVolume, ok := clonedPodVolumeObj.(*v1.Volume)
	if !ok {
		return nil, fmt.Errorf(
			"failed to cast clonedPodVolume %#v to v1.Volume",
			clonedPodVolumeObj)
	}

	return volume.NewSpecFromVolume(clonedPodVolume), nil
}

// getPVCExtractPV fetches the PVC object with the given namespace and name from
// the API server extracts the name of the PV it is pointing to and returns it.
// An error is returned if the PVC object's phase is not "Bound".
func (dswp *desiredStateOfWorldPopulator) getPVCExtractPV(namespace string, claimName string) (string, types.UID, error) {
	pvc, err :=
		dswp.kubeClient.Core().PersistentVolumeClaims(namespace).Get(claimName, metav1.GetOptions{})
	if err != nil || pvc == nil {
		return "", "", fmt.Errorf(
			"failed to fetch PVC %s/%s from API server. err=%v",
			namespace,
			claimName,
			err)
	}

	if pvc.Status.Phase != v1.ClaimBound || pvc.Spec.VolumeName == "" {
		return "", "", fmt.Errorf(
			"PVC %s/%s has non-bound phase (%q) or empty pvc.Spec.VolumeName (%q)",
			namespace,
			claimName,
			pvc.Status.Phase,
			pvc.Spec.VolumeName)
	}

	return pvc.Spec.VolumeName, pvc.UID, nil
}

// getPVSpec fetches the PV object with the given name from the API server
// and returns a volume.Spec representing it.
// An error is returned if the call to fetch the PV object fails.
func (dswp *desiredStateOfWorldPopulator) getPVSpec(name string, pvcReadOnly bool, expectedClaimUID types.UID) (*volume.Spec, error) {
	pv, err := dswp.kubeClient.Core().PersistentVolumes().Get(name, metav1.GetOptions{})
	if err != nil || pv == nil {
		return nil, fmt.Errorf(
			"failed to fetch PV %q from API server. err=%v", name, err)
	}

	if pv.Spec.ClaimRef == nil {
		return nil, fmt.Errorf(
			"found PV object %q but it has a nil pv.Spec.ClaimRef indicating it is not yet bound to the claim",
			name)
	}

	if pv.Spec.ClaimRef.UID != expectedClaimUID {
		return nil, fmt.Errorf(
			"found PV object %q but its pv.Spec.ClaimRef.UID (%q) does not point to claim.UID (%q)",
			name,
			pv.Spec.ClaimRef.UID,
			expectedClaimUID)
	}

	return volume.NewSpecFromPersistentVolume(pv, pvcReadOnly), nil
}
