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
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/volume/expand/cache"
)

// PVCPopulator iterates through PVCs and checks if for bound PVCs
// their size doesn't match with Persistent Volume size
type PVCPopulator interface {
	Run(stopCh <-chan struct{})
}

type pvcPopulator struct {
	loopPeriod time.Duration
	resizeMap  cache.VolumeResizeMap
	pvcLister  corelisters.PersistentVolumeClaimLister
	pvLister   corelisters.PersistentVolumeLister
	kubeClient clientset.Interface
}

func NewPVCPopulator(
	loopPeriod time.Duration,
	resizeMap cache.VolumeResizeMap,
	pvcLister corelisters.PersistentVolumeClaimLister,
	pvLister corelisters.PersistentVolumeLister,
	kubeClient clientset.Interface) PVCPopulator {
	populator := &pvcPopulator{
		loopPeriod: loopPeriod,
		pvcLister:  pvcLister,
		pvLister:   pvLister,
		resizeMap:  resizeMap,
		kubeClient: kubeClient,
	}
	return populator
}

func (populator *pvcPopulator) Run(stopCh <-chan struct{}) {
	wait.Until(populator.Sync, populator.loopPeriod, stopCh)
}

func (populator *pvcPopulator) Sync() {
	pvcs, err := populator.pvcLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("Listing PVCs failed in populator : %v", err)
		return
	}

	for _, pvc := range pvcs {
		pv, err := getPersistentVolume(pvc, populator.pvLister)

		if err != nil {
			glog.V(5).Infof("Error getting persistent volume for pvc %q : %v", pvc.UID, err)
			continue
		}
		populator.resizeMap.AddPVCUpdate(pvc, pv)
	}
}
