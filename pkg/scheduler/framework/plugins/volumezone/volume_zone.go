/*
Copyright 2019 The Kubernetes Authors.

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

package volumezone

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/runtime"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	"k8s.io/klog"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// VolumeZone is a plugin that checks volume zone.
type VolumeZone struct {
	pvLister  corelisters.PersistentVolumeLister
	pvcLister corelisters.PersistentVolumeClaimLister
	scLister  storagelisters.StorageClassLister
}

var _ framework.FilterPlugin = &VolumeZone{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "VolumeZone"

	// ErrReasonConflict is used for NoVolumeZoneConflict predicate error.
	ErrReasonConflict = "node(s) had no available volume zone"
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *VolumeZone) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
//
// It evaluates if a pod can fit due to the volumes it requests, given
// that some volumes may have zone scheduling constraints.  The requirement is that any
// volume zone-labels must match the equivalent zone-labels on the node.  It is OK for
// the node to have more zone-label constraints (for example, a hypothetical replicated
// volume might allow region-wide access)
//
// Currently this is only supported with PersistentVolumeClaims, and looks to the labels
// only on the bound PersistentVolume.
//
// Working with volumes declared inline in the pod specification (i.e. not
// using a PersistentVolume) is likely to be harder, as it would require
// determining the zone of a volume during scheduling, and that is likely to
// require calling out to the cloud provider.  It seems that we are moving away
// from inline volume declarations anyway.
func (pl *VolumeZone) Filter(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return nil
	}
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}
	nodeConstraints := make(map[string]string)
	for k, v := range node.ObjectMeta.Labels {
		if k != v1.LabelZoneFailureDomain && k != v1.LabelZoneRegion {
			continue
		}
		nodeConstraints[k] = v
	}
	if len(nodeConstraints) == 0 {
		// The node has no zone constraints, so we're OK to schedule.
		// In practice, when using zones, all nodes must be labeled with zone labels.
		// We want to fast-path this case though.
		return nil
	}

	for i := range pod.Spec.Volumes {
		volume := pod.Spec.Volumes[i]
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		pvcName := volume.PersistentVolumeClaim.ClaimName
		if pvcName == "" {
			return framework.NewStatus(framework.Error, "PersistentVolumeClaim had no name")
		}
		pvc, err := pl.pvcLister.PersistentVolumeClaims(pod.Namespace).Get(pvcName)
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}

		if pvc == nil {
			return framework.NewStatus(framework.Error, fmt.Sprintf("PersistentVolumeClaim was not found: %q", pvcName))
		}

		pvName := pvc.Spec.VolumeName
		if pvName == "" {
			scName := v1helper.GetPersistentVolumeClaimClass(pvc)
			if len(scName) == 0 {
				return framework.NewStatus(framework.Error, fmt.Sprint("PersistentVolumeClaim had no pv name and storageClass name"))
			}

			class, _ := pl.scLister.Get(scName)
			if class == nil {
				return framework.NewStatus(framework.Error, fmt.Sprintf("StorageClass %q claimed by PersistentVolumeClaim %q not found", scName, pvcName))

			}
			if class.VolumeBindingMode == nil {
				return framework.NewStatus(framework.Error, fmt.Sprintf("VolumeBindingMode not set for StorageClass %q", scName))
			}
			if *class.VolumeBindingMode == storage.VolumeBindingWaitForFirstConsumer {
				// Skip unbound volumes
				continue
			}

			return framework.NewStatus(framework.Error, fmt.Sprint("PersistentVolume had no name"))
		}

		pv, err := pl.pvLister.Get(pvName)
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}

		if pv == nil {
			return framework.NewStatus(framework.Error, fmt.Sprintf("PersistentVolume was not found: %q", pvName))
		}

		for k, v := range pv.ObjectMeta.Labels {
			if k != v1.LabelZoneFailureDomain && k != v1.LabelZoneRegion {
				continue
			}
			nodeV, _ := nodeConstraints[k]
			volumeVSet, err := volumehelpers.LabelZonesToSet(v)
			if err != nil {
				klog.Warningf("Failed to parse label for %q: %q. Ignoring the label. err=%v. ", k, v, err)
				continue
			}

			if !volumeVSet.Has(nodeV) {
				klog.V(10).Infof("Won't schedule pod %q onto node %q due to volume %q (mismatch on %q)", pod.Name, node.Name, pvName, k)
				return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict)
			}
		}
	}
	return nil
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	pvLister := informerFactory.Core().V1().PersistentVolumes().Lister()
	pvcLister := informerFactory.Core().V1().PersistentVolumeClaims().Lister()
	scLister := informerFactory.Storage().V1().StorageClasses().Lister()
	return &VolumeZone{
		pvLister,
		pvcLister,
		scLister,
	}, nil
}
