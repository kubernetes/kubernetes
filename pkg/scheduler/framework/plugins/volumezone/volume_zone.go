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
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	storagehelpers "k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// VolumeZone is a plugin that checks volume zone.
type VolumeZone struct {
	pvLister  corelisters.PersistentVolumeLister
	pvcLister corelisters.PersistentVolumeClaimLister
	scLister  storagelisters.StorageClassLister
}

var _ framework.FilterPlugin = &VolumeZone{}
var _ framework.PreFilterPlugin = &VolumeZone{}
var _ framework.EnqueueExtensions = &VolumeZone{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.VolumeZone

	preFilterStateKey framework.StateKey = "PreFilter" + Name

	// ErrReasonConflict is used for NoVolumeZoneConflict predicate error.
	ErrReasonConflict = "node(s) had no available volume zone"
)

// pvTopology holds the value of a pv's topologyLabel
type pvTopology struct {
	pvName string
	key    string
	values sets.Set[string]
}

// the state is initialized in PreFilter phase. because we save the pointer in
// framework.CycleState, in the later phases we don't need to call Write method
// to update the value
type stateData struct {
	// podPVTopologies holds the pv information we need
	// it's initialized in the PreFilter phase
	podPVTopologies []pvTopology
}

func (d *stateData) Clone() framework.StateData {
	return d
}

var topologyLabels = []string{
	v1.LabelFailureDomainBetaZone,
	v1.LabelFailureDomainBetaRegion,
	v1.LabelTopologyZone,
	v1.LabelTopologyRegion,
}

func translateToGALabel(label string) string {
	if label == v1.LabelFailureDomainBetaRegion {
		return v1.LabelTopologyRegion
	}
	if label == v1.LabelFailureDomainBetaZone {
		return v1.LabelTopologyZone
	}
	return label
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *VolumeZone) Name() string {
	return Name
}

// PreFilter invoked at the prefilter extension point
//
// # It finds the topology of the PersistentVolumes corresponding to the volumes a pod requests
//
// Currently, this is only supported with PersistentVolumeClaims,
// and only looks for the bound PersistentVolume.
func (pl *VolumeZone) PreFilter(ctx context.Context, cs *framework.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	logger := klog.FromContext(ctx)
	podPVTopologies, status := pl.getPVbyPod(logger, pod)
	if !status.IsSuccess() {
		return nil, status
	}
	if len(podPVTopologies) == 0 {
		return nil, framework.NewStatus(framework.Skip)
	}
	cs.Write(preFilterStateKey, &stateData{podPVTopologies: podPVTopologies})
	return nil, nil
}

// getPVbyPod gets PVTopology from pod
func (pl *VolumeZone) getPVbyPod(logger klog.Logger, pod *v1.Pod) ([]pvTopology, *framework.Status) {
	podPVTopologies := make([]pvTopology, 0)

	pvcNames := pl.getPersistentVolumeClaimNameFromPod(pod)
	for _, pvcName := range pvcNames {
		if pvcName == "" {
			return nil, framework.NewStatus(framework.UnschedulableAndUnresolvable, "PersistentVolumeClaim had no name")
		}
		pvc, err := pl.pvcLister.PersistentVolumeClaims(pod.Namespace).Get(pvcName)
		if s := getErrorAsStatus(err); !s.IsSuccess() {
			return nil, s
		}

		pvName := pvc.Spec.VolumeName
		if pvName == "" {
			scName := storagehelpers.GetPersistentVolumeClaimClass(pvc)
			if len(scName) == 0 {
				return nil, framework.NewStatus(framework.UnschedulableAndUnresolvable, "PersistentVolumeClaim had no pv name and storageClass name")
			}

			class, err := pl.scLister.Get(scName)
			if s := getErrorAsStatus(err); !s.IsSuccess() {
				return nil, s
			}
			if class.VolumeBindingMode == nil {
				return nil, framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("VolumeBindingMode not set for StorageClass %q", scName))
			}
			if *class.VolumeBindingMode == storage.VolumeBindingWaitForFirstConsumer {
				// Skip unbound volumes
				continue
			}

			return nil, framework.NewStatus(framework.UnschedulableAndUnresolvable, "PersistentVolume had no name")
		}

		pv, err := pl.pvLister.Get(pvName)
		if s := getErrorAsStatus(err); !s.IsSuccess() {
			return nil, s
		}

		for _, key := range topologyLabels {
			if value, ok := pv.ObjectMeta.Labels[key]; ok {
				volumeVSet, err := volumehelpers.LabelZonesToSet(value)
				if err != nil {
					logger.Info("Failed to parse label, ignoring the label", "label", fmt.Sprintf("%s:%s", key, value), "err", err)
					continue
				}
				podPVTopologies = append(podPVTopologies, pvTopology{
					pvName: pv.Name,
					key:    key,
					values: sets.Set[string](volumeVSet),
				})
			}
		}
	}
	return podPVTopologies, nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *VolumeZone) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
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
func (pl *VolumeZone) Filter(ctx context.Context, cs *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	logger := klog.FromContext(ctx)
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return nil
	}
	var podPVTopologies []pvTopology
	state, err := getStateData(cs)
	if err != nil {
		// Fallback to calculate pv list here
		var status *framework.Status
		podPVTopologies, status = pl.getPVbyPod(logger, pod)
		if !status.IsSuccess() {
			return status
		}
	} else {
		podPVTopologies = state.podPVTopologies
	}

	node := nodeInfo.Node()
	hasAnyNodeConstraint := false
	for _, topologyLabel := range topologyLabels {
		if _, ok := node.Labels[topologyLabel]; ok {
			hasAnyNodeConstraint = true
			break
		}
	}

	if !hasAnyNodeConstraint {
		// The node has no zone constraints, so we're OK to schedule.
		// This is to handle a single-zone cluster scenario where the node may not have any topology labels.
		return nil
	}

	for _, pvTopology := range podPVTopologies {
		v, ok := node.Labels[pvTopology.key]
		if !ok {
			// if we can't match the beta label, try to match pv's beta label with node's ga label
			v, ok = node.Labels[translateToGALabel(pvTopology.key)]
		}
		if !ok || !pvTopology.values.Has(v) {
			logger.V(10).Info("Won't schedule pod onto node due to volume (mismatch on label key)", "pod", klog.KObj(pod), "node", klog.KObj(node), "PV", klog.KRef("", pvTopology.pvName), "PVLabelKey", pvTopology.key)
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict)
		}
	}

	return nil
}

func getStateData(cs *framework.CycleState) (*stateData, error) {
	state, err := cs.Read(preFilterStateKey)
	if err != nil {
		return nil, err
	}
	s, ok := state.(*stateData)
	if !ok {
		return nil, errors.New("unable to convert state into stateData")
	}
	return s, nil
}

func getErrorAsStatus(err error) *framework.Status {
	if err != nil {
		if apierrors.IsNotFound(err) {
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, err.Error())
		}
		return framework.AsStatus(err)
	}
	return nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *VolumeZone) EventsToRegister() []framework.ClusterEventWithHint {
	return []framework.ClusterEventWithHint{
		// New storageClass with bind mode `VolumeBindingWaitForFirstConsumer` will make a pod schedulable.
		// Due to immutable field `storageClass.volumeBindingMode`, storageClass update events are ignored.
		{Event: framework.ClusterEvent{Resource: framework.StorageClass, ActionType: framework.Add}},
		// A new node or updating a node's volume zone labels may make a pod schedulable.
		//
		// A note about UpdateNodeTaint event:
		// NodeAdd QueueingHint isn't always called because of the internal feature called preCheck.
		// As a common problematic scenario,
		// when a node is added but not ready, NodeAdd event is filtered out by preCheck and doesn't arrive.
		// In such cases, this plugin may miss some events that actually make pods schedulable.
		// As a workaround, we add UpdateNodeTaint event to catch the case.
		// We can remove UpdateNodeTaint when we remove the preCheck feature.
		// See: https://github.com/kubernetes/kubernetes/issues/110175
		{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeLabel | framework.UpdateNodeTaint}},
		// A new pvc may make a pod schedulable.
		// Also, if pvc's VolumeName is filled, that also could make a pod schedulable.
		{Event: framework.ClusterEvent{Resource: framework.PersistentVolumeClaim, ActionType: framework.Add | framework.Update}, QueueingHintFn: pl.isSchedulableAfterPersistentVolumeClaimChange},
		// A new pv or updating a pv's volume zone labels may make a pod schedulable.
		{Event: framework.ClusterEvent{Resource: framework.PersistentVolume, ActionType: framework.Add | framework.Update}},
	}
}

// getPersistentVolumeClaimNameFromPod gets pvc names bound to a pod.
func (pl *VolumeZone) getPersistentVolumeClaimNameFromPod(pod *v1.Pod) []string {
	var pvcNames []string
	for i := range pod.Spec.Volumes {
		volume := pod.Spec.Volumes[i]
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		pvcName := volume.PersistentVolumeClaim.ClaimName
		pvcNames = append(pvcNames, pvcName)
	}
	return pvcNames
}

// isSchedulableAfterPersistentVolumeClaimChange is invoked whenever a PersistentVolumeClaim added or updated.
// It checks whether the change of PVC has made a previously unschedulable pod schedulable.
// A PVC becoming bound or using a WaitForFirstConsumer storageclass can cause the pod to become schedulable.
func (pl *VolumeZone) isSchedulableAfterPersistentVolumeClaimChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	_, modifiedPVC, err := util.As[*v1.PersistentVolumeClaim](oldObj, newObj)
	if err != nil {
		return framework.Queue, fmt.Errorf("unexpected objects in isSchedulableAfterPersistentVolumeClaimChange: %w", err)
	}
	if pl.IsPVCRequestedFromPod(logger, modifiedPVC, pod) {
		logger.V(5).Info("PVC was created or updated and it might make this pod schedulable. PVC is binding to the pod.", "pod", klog.KObj(pod), "PVC", klog.KObj(modifiedPVC))
		return framework.Queue, nil
	}

	logger.V(5).Info("PVC irrelevant to the Pod was created or updated, which doesn't make this pod schedulable", "pod", klog.KObj(pod), "PVC", klog.KObj(modifiedPVC))
	return framework.QueueSkip, nil
}

// IsPVCRequestedFromPod verifies if the PVC is bound to PV of a given Pod.
func (pl *VolumeZone) IsPVCRequestedFromPod(logger klog.Logger, pvc *v1.PersistentVolumeClaim, pod *v1.Pod) bool {
	if (pvc == nil) || (pod.Namespace != pvc.Namespace) {
		return false
	}
	pvcNames := pl.getPersistentVolumeClaimNameFromPod(pod)
	for _, pvcName := range pvcNames {
		if pvc.Name == pvcName {
			logger.V(5).Info("PVC matches the pod's PVC", "pod", klog.KObj(pod), "PVC", klog.KObj(pvc))
			return true
		}
	}
	logger.V(5).Info("PVC doesn't match the pod's PVC", "pod", klog.KObj(pod), "PVC", klog.KObj(pvc))
	return false
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, handle framework.Handle) (framework.Plugin, error) {
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
