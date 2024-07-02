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
	"context"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	corelisters "k8s.io/client-go/listers/core/v1"
	kcache "k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/util"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

// DesiredStateOfWorldPopulator periodically verifies that the pods in the
// desired state of the world still exist, if not, it removes them.
// It also loops through the list of active pods and ensures that
// each one exists in the desired state of the world cache
// if it has volumes.
type DesiredStateOfWorldPopulator interface {
	Run(ctx context.Context)
}

// NewDesiredStateOfWorldPopulator returns a new instance of DesiredStateOfWorldPopulator.
// loopSleepDuration - the amount of time the populator loop sleeps between
// successive executions
//
// podManager - the kubelet podManager that is the source of truth for the pods
// that exist on this host
//
// desiredStateOfWorld - the cache to populate
func NewDesiredStateOfWorldPopulator(
	loopSleepDuration time.Duration,
	listPodsRetryDuration time.Duration,
	podLister corelisters.PodLister,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	volumePluginMgr *volume.VolumePluginMgr,
	pvcLister corelisters.PersistentVolumeClaimLister,
	pvLister corelisters.PersistentVolumeLister,
	csiMigratedPluginManager csimigration.PluginManager,
	intreeToCSITranslator csimigration.InTreeToCSITranslator) DesiredStateOfWorldPopulator {
	return &desiredStateOfWorldPopulator{
		loopSleepDuration:        loopSleepDuration,
		listPodsRetryDuration:    listPodsRetryDuration,
		podLister:                podLister,
		desiredStateOfWorld:      desiredStateOfWorld,
		volumePluginMgr:          volumePluginMgr,
		pvcLister:                pvcLister,
		pvLister:                 pvLister,
		csiMigratedPluginManager: csiMigratedPluginManager,
		intreeToCSITranslator:    intreeToCSITranslator,
	}
}

type desiredStateOfWorldPopulator struct {
	loopSleepDuration        time.Duration
	podLister                corelisters.PodLister
	desiredStateOfWorld      cache.DesiredStateOfWorld
	volumePluginMgr          *volume.VolumePluginMgr
	pvcLister                corelisters.PersistentVolumeClaimLister
	pvLister                 corelisters.PersistentVolumeLister
	listPodsRetryDuration    time.Duration
	timeOfLastListPods       time.Time
	csiMigratedPluginManager csimigration.PluginManager
	intreeToCSITranslator    csimigration.InTreeToCSITranslator
}

func (dswp *desiredStateOfWorldPopulator) Run(ctx context.Context) {
	wait.UntilWithContext(ctx, dswp.populatorLoopFunc(ctx), dswp.loopSleepDuration)
}

func (dswp *desiredStateOfWorldPopulator) populatorLoopFunc(ctx context.Context) func(ctx context.Context) {
	return func(ctx context.Context) {
		logger := klog.FromContext(ctx)
		dswp.findAndRemoveDeletedPods(logger)

		// findAndAddActivePods is called periodically, independently of the main
		// populator loop.
		if time.Since(dswp.timeOfLastListPods) < dswp.listPodsRetryDuration {
			logger.V(5).Info(
				"Skipping findAndAddActivePods(). Not permitted until the retry time is reached",
				"retryTime", dswp.timeOfLastListPods.Add(dswp.listPodsRetryDuration),
				"retryDuration", dswp.listPodsRetryDuration)

			return
		}
		dswp.findAndAddActivePods(logger)
	}
}

// Iterate through all pods in desired state of world, and remove if they no
// longer exist in the informer
func (dswp *desiredStateOfWorldPopulator) findAndRemoveDeletedPods(logger klog.Logger) {
	for dswPodUID, dswPodToAdd := range dswp.desiredStateOfWorld.GetPodToAdd() {
		dswPodKey, err := kcache.MetaNamespaceKeyFunc(dswPodToAdd.Pod)
		if err != nil {
			logger.Error(err, "MetaNamespaceKeyFunc failed for pod", "podName", dswPodKey, "podUID", dswPodUID)
			continue
		}

		// Retrieve the pod object from pod informer with the namespace key
		namespace, name, err := kcache.SplitMetaNamespaceKey(dswPodKey)
		if err != nil {
			utilruntime.HandleErrorWithContext(klog.NewContext(context.Background(), logger), err, "Error splitting dswPodKey", "key", dswPodKey)
			continue
		}
		informerPod, err := dswp.podLister.Pods(namespace).Get(name)
		switch {
		case errors.IsNotFound(err):
			// if we can't find the pod, we need to delete it below
		case err != nil:
			logger.Error(err, "podLister Get failed for pod", "podName", dswPodKey, "podUID", dswPodUID)
			continue
		default:
			volumeActionFlag := util.DetermineVolumeAction(
				informerPod,
				dswp.desiredStateOfWorld,
				true /* default volume action */)

			if volumeActionFlag {
				informerPodUID := volutil.GetUniquePodName(informerPod)
				// Check whether the unique identifier of the pod from dsw matches the one retrieved from pod informer
				if informerPodUID == dswPodUID {
					logger.V(10).Info("Verified podfrom dsw exists in pod informer", "podName", dswPodKey, "podUID", dswPodUID)
					continue
				}
			}
		}

		// the pod from dsw does not exist in pod informer, or it does not match the unique identifier retrieved
		// from the informer, delete it from dsw
		logger.V(1).Info("Removing pod from dsw because it does not exist in pod informer", "podName", dswPodKey, "podUID", dswPodUID)
		dswp.desiredStateOfWorld.DeletePod(dswPodUID, dswPodToAdd.VolumeName, dswPodToAdd.NodeName)
	}

	// check if the existing volumes changes its attachability
	for _, volumeToAttach := range dswp.desiredStateOfWorld.GetVolumesToAttach() {
		// IsAttachableVolume() will result in a fetch of CSIDriver object if the volume plugin type is CSI.
		// The result is returned from CSIDriverLister which is from local cache. So this is not an expensive call.
		volumeAttachable := volutil.IsAttachableVolume(volumeToAttach.VolumeSpec, dswp.volumePluginMgr)
		if !volumeAttachable {
			logger.Info("Volume changes from attachable to non-attachable", "volumeName", volumeToAttach.VolumeName)
			for _, scheduledPod := range volumeToAttach.ScheduledPods {
				podUID := volutil.GetUniquePodName(scheduledPod)
				dswp.desiredStateOfWorld.DeletePod(podUID, volumeToAttach.VolumeName, volumeToAttach.NodeName)
				logger.V(4).Info("Removing podUID and volume on node from desired state of world"+
					" because of the change of volume attachability", "node", klog.KRef("", string(volumeToAttach.NodeName)), "podUID", podUID, "volumeName", volumeToAttach.VolumeName)
			}
		}
	}
}

func (dswp *desiredStateOfWorldPopulator) findAndAddActivePods(logger klog.Logger) {
	pods, err := dswp.podLister.List(labels.Everything())
	if err != nil {
		logger.Error(err, "PodLister List failed")
		return
	}
	dswp.timeOfLastListPods = time.Now()

	for _, pod := range pods {
		if volutil.IsPodTerminated(pod, pod.Status) {
			// Do not add volumes for terminated pods
			continue
		}
		util.ProcessPodVolumes(logger, pod, true,
			dswp.desiredStateOfWorld, dswp.volumePluginMgr, dswp.pvcLister, dswp.pvLister, dswp.csiMigratedPluginManager, dswp.intreeToCSITranslator)

	}

}
