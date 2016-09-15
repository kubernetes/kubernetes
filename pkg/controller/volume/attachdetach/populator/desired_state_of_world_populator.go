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
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	kcache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// DesiredStateOfWorldPopulator periodically verifies that the pods in the
// desired state of th world still exist, if not, it removes them.
// TODO: it also loops through the list of active pods and ensures that
// each one exists in the desired state of the world cache
// if it has volumes.
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
	podInformer kcache.SharedInformer,
	desiredStateOfWorld cache.DesiredStateOfWorld) DesiredStateOfWorldPopulator {
	return &desiredStateOfWorldPopulator{
		loopSleepDuration:   loopSleepDuration,
		podInformer:         podInformer,
		desiredStateOfWorld: desiredStateOfWorld,
	}
}

type desiredStateOfWorldPopulator struct {
	loopSleepDuration   time.Duration
	podInformer         kcache.SharedInformer
	desiredStateOfWorld cache.DesiredStateOfWorld
}

func (dswp *desiredStateOfWorldPopulator) Run(stopCh <-chan struct{}) {
	wait.Until(dswp.populatorLoopFunc(), dswp.loopSleepDuration, stopCh)
}

func (dswp *desiredStateOfWorldPopulator) populatorLoopFunc() func() {
	return func() {
		dswp.findAndRemoveDeletedPods()
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
		informerPodObj, exists, err :=
			dswp.podInformer.GetStore().GetByKey(dswPodKey)
		if err != nil {
			glog.Errorf(
				"podInformer GetByKey failed for pod %q (UID %q) with %v",
				dswPodKey,
				dswPodUID,
				err)
			continue
		}
		if exists && informerPodObj == nil {
			glog.Info(
				"podInformer GetByKey found pod, but informerPodObj is nil for pod %q (UID %q)",
				dswPodKey,
				dswPodUID)
			continue
		}

		if exists {
			informerPod, ok := informerPodObj.(*api.Pod)
			if !ok {
				glog.Errorf("Failed to cast obj %#v to pod object for pod %q (UID %q)", informerPod, dswPodKey, dswPodUID)
				continue
			}
			informerPodUID := volumehelper.GetUniquePodName(informerPod)
			// Check whether the unique identifier of the pod from dsw matches the one retrieved from pod informer
			if informerPodUID == dswPodUID {
				glog.V(10).Infof(
					"Verified pod %q (UID %q) from dsw exists in pod informer.", dswPodKey, dswPodUID)
				continue

			}
		}
		// the pod from dsw does not exist in pod informer, or it does not match the unique identifer retrieved
		// from the informer, delete it from dsw
		glog.V(1).Infof(
			"Removing pod %q (UID %q) from dsw because it does not exist in pod informer.", dswPodKey, dswPodUID)
		dswp.desiredStateOfWorld.DeletePod(dswPodUID, dswPodToAdd.VolumeName, dswPodToAdd.NodeName)
	}
}
