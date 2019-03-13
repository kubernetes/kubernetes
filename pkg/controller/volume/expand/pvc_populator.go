/*
Copyright 2017 The Kubernetes Authors.

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

// Package reconciler implements interfaces that attempt to reconcile the
// desired state of the with the actual state of the world by triggering
// actions.

package expand

import (
	"fmt"
	"time"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/controller/volume/events"
	"k8s.io/kubernetes/pkg/controller/volume/expand/cache"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// PVCPopulator iterates through PVCs and checks if for bound PVCs
// their size doesn't match with Persistent Volume size
type PVCPopulator interface {
	Run(stopCh <-chan struct{})
}

type pvcPopulator struct {
	loopPeriod      time.Duration
	resizeMap       cache.VolumeResizeMap
	pvcLister       corelisters.PersistentVolumeClaimLister
	pvLister        corelisters.PersistentVolumeLister
	kubeClient      clientset.Interface
	volumePluginMgr *volume.VolumePluginMgr
	recorder        record.EventRecorder
}

func NewPVCPopulator(
	loopPeriod time.Duration,
	resizeMap cache.VolumeResizeMap,
	pvcLister corelisters.PersistentVolumeClaimLister,
	pvLister corelisters.PersistentVolumeLister,
	volumePluginMgr *volume.VolumePluginMgr,
	kubeClient clientset.Interface) PVCPopulator {
	populator := &pvcPopulator{
		loopPeriod:      loopPeriod,
		pvcLister:       pvcLister,
		pvLister:        pvLister,
		resizeMap:       resizeMap,
		volumePluginMgr: volumePluginMgr,
		kubeClient:      kubeClient,
	}
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	populator.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "volume_expand"})
	return populator
}

func (populator *pvcPopulator) Run(stopCh <-chan struct{}) {
	wait.Until(populator.Sync, populator.loopPeriod, stopCh)
}

func (populator *pvcPopulator) Sync() {
	pvcs, err := populator.pvcLister.List(labels.Everything())
	if err != nil {
		klog.Errorf("Listing PVCs failed in populator : %v", err)
		return
	}

	for _, pvc := range pvcs {
		pv, err := getPersistentVolume(pvc, populator.pvLister)

		if err != nil {
			klog.V(5).Infof("Error getting persistent volume for PVC %q : %v", pvc.UID, err)
			continue
		}

		// Filter PVCs for which the corresponding volume plugins don't allow expansion.
		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		pvcStatusSize := pvc.Status.Capacity[v1.ResourceStorage]
		volumeSpec := volume.NewSpecFromPersistentVolume(pv, false)
		volumePlugin, err := populator.volumePluginMgr.FindExpandablePluginBySpec(volumeSpec)
		if (err != nil || volumePlugin == nil) && pvcStatusSize.Cmp(pvcSize) < 0 {
			err = fmt.Errorf("didn't find a plugin capable of expanding the volume; " +
				"waiting for an external controller to process this PVC")
			eventType := v1.EventTypeNormal
			if err != nil {
				eventType = v1.EventTypeWarning
			}
			populator.recorder.Event(pvc, eventType, events.ExternalExpanding,
				fmt.Sprintf("Ignoring the PVC: %v.", err))
			klog.V(3).Infof("Ignoring the PVC %q (uid: %q) : %v.",
				util.GetPersistentVolumeClaimQualifiedName(pvc), pvc.UID, err)
			continue
		}

		// We are only going to add PVCs which are:
		//  - bound
		//  - pvc.Spec.Size > pvc.Status.Size
		// These 2 checks are already performed in AddPVCUpdate function before adding pvc for resize
		// and hence we do not repeat those checks here.
		populator.resizeMap.AddPVCUpdate(pvc, pv)
	}
}
