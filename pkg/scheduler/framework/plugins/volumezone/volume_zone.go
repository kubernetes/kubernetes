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
	"reflect"

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
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// VolumeZone is a plugin that checks volume zone.
type VolumeZone struct {
	pvLister                  corelisters.PersistentVolumeLister
	pvcLister                 corelisters.PersistentVolumeClaimLister
	scLister                  storagelisters.StorageClassLister
	enableSchedulingQueueHint bool
}

var _ framework.FilterPlugin = &VolumeZone{}
var _ framework.PreFilterPlugin = &VolumeZone{}
var _ framework.EnqueueExtensions = &VolumeZone{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.VolumeZone

	preFilterStateKey fwk.StateKey = "PreFilter" + Name

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
// fwk.CycleState, in the later phases we don't need to call Write method
// to update the value
type stateData struct {
	// podPVTopologies holds the pv information we need
	// it's initialized in the PreFilter phase
	podPVTopologies []pvTopology
}

func (d *stateData) Clone() fwk.StateData {
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
func (pl *VolumeZone) PreFilter(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {
	logger := klog.FromContext(ctx)
	podPVTopologies, status := pl.getPVbyPod(logger, pod)
	if !status.IsSuccess() {
		return nil, status
	}
	if len(podPVTopologies) == 0 {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	cs.Write(preFilterStateKey, &stateData{podPVTopologies: podPVTopologies})
	return nil, nil
}

// getPVbyPod gets PVTopology from pod
func (pl *VolumeZone) getPVbyPod(logger klog.Logger, pod *v1.Pod) ([]pvTopology, *fwk.Status) {
	podPVTopologies := make([]pvTopology, 0)

	pvcNames := pl.getPersistentVolumeClaimNameFromPod(pod)
	for _, pvcName := range pvcNames {
		if pvcName == "" {
			return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "PersistentVolumeClaim had no name")
		}
		pvc, err := pl.pvcLister.PersistentVolumeClaims(pod.Namespace).Get(pvcName)
		if s := getErrorAsStatus(err); !s.IsSuccess() {
			return nil, s
		}

		pvName := pvc.Spec.VolumeName
		if pvName == "" {
			scName := storagehelpers.GetPersistentVolumeClaimClass(pvc)
			if len(scName) == 0 {
				return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "PersistentVolumeClaim had no pv name and storageClass name")
			}

			class, err := pl.scLister.Get(scName)
			if s := getErrorAsStatus(err); !s.IsSuccess() {
				return nil, s
			}
			if class.VolumeBindingMode == nil {
				return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("VolumeBindingMode not set for StorageClass %q", scName))
			}
			if *class.VolumeBindingMode == storage.VolumeBindingWaitForFirstConsumer {
				// Skip unbound volumes
				continue
			}

			return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "PersistentVolume had no name")
		}

		pv, err := pl.pvLister.Get(pvName)
		if s := getErrorAsStatus(err); !s.IsSuccess() {
			return nil, s
		}
		podPVTopologies = append(podPVTopologies, pl.getPVTopologies(logger, pv)...)
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
func (pl *VolumeZone) Filter(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
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
		var status *fwk.Status
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
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonConflict)
		}
	}

	return nil
}

func getStateData(cs fwk.CycleState) (*stateData, error) {
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

func getErrorAsStatus(err error) *fwk.Status {
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, err.Error())
		}
		return fwk.AsStatus(err)
	}
	return nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *VolumeZone) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	// A new node or updating a node's volume zone labels may make a pod schedulable.
	// A note about UpdateNodeTaint event:
	// Ideally, it's supposed to register only Add | UpdateNodeLabel because UpdateNodeTaint will never change the result from this plugin.
	// But, we may miss Node/Add event due to preCheck, and we decided to register UpdateNodeTaint | UpdateNodeLabel for all plugins registering Node/Add.
	// See: https://github.com/kubernetes/kubernetes/issues/109437
	nodeActionType := fwk.Add | fwk.UpdateNodeLabel | fwk.UpdateNodeTaint
	if pl.enableSchedulingQueueHint {
		// preCheck is not used when QHint is enabled.
		nodeActionType = fwk.Add | fwk.UpdateNodeLabel
	}

	return []fwk.ClusterEventWithHint{
		// New storageClass with bind mode `VolumeBindingWaitForFirstConsumer` will make a pod schedulable.
		// Due to immutable field `storageClass.volumeBindingMode`, storageClass update events are ignored.
		{Event: fwk.ClusterEvent{Resource: fwk.StorageClass, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterStorageClassAdded},
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: nodeActionType}},
		// A new pvc may make a pod schedulable.
		// Also, if pvc's VolumeName is filled, that also could make a pod schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.PersistentVolumeClaim, ActionType: fwk.Add | fwk.Update}, QueueingHintFn: pl.isSchedulableAfterPersistentVolumeClaimChange},
		// A new pv or updating a pv's volume zone labels may make a pod schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.PersistentVolume, ActionType: fwk.Add | fwk.Update}, QueueingHintFn: pl.isSchedulableAfterPersistentVolumeChange},
	}, nil
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
func (pl *VolumeZone) isSchedulableAfterPersistentVolumeClaimChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, modifiedPVC, err := util.As[*v1.PersistentVolumeClaim](oldObj, newObj)
	if err != nil {
		return fwk.Queue, fmt.Errorf("unexpected objects in isSchedulableAfterPersistentVolumeClaimChange: %w", err)
	}
	if pl.isPVCRequestedFromPod(logger, modifiedPVC, pod) {
		logger.V(5).Info("PVC that is referred from the pod was created or updated, which might make this pod schedulable", "pod", klog.KObj(pod), "PVC", klog.KObj(modifiedPVC))
		return fwk.Queue, nil
	}

	logger.V(5).Info("PVC irrelevant to the Pod was created or updated, which doesn't make this pod schedulable", "pod", klog.KObj(pod), "PVC", klog.KObj(modifiedPVC))
	return fwk.QueueSkip, nil
}

// isPVCRequestedFromPod verifies if the PVC is requested from a given Pod.
func (pl *VolumeZone) isPVCRequestedFromPod(logger klog.Logger, pvc *v1.PersistentVolumeClaim, pod *v1.Pod) bool {
	if (pvc == nil) || (pod.Namespace != pvc.Namespace) {
		return false
	}
	pvcNames := pl.getPersistentVolumeClaimNameFromPod(pod)
	for _, pvcName := range pvcNames {
		if pvc.Name == pvcName {
			logger.V(5).Info("PVC is referred from the pod", "pod", klog.KObj(pod), "PVC", klog.KObj(pvc))
			return true
		}
	}
	logger.V(5).Info("PVC is not referred from the pod", "pod", klog.KObj(pod), "PVC", klog.KObj(pvc))
	return false
}

// isSchedulableAfterStorageClassAdded is invoked whenever a StorageClass is added.
// It checks whether the addition of StorageClass has made a previously unschedulable pod schedulable.
// Only a new StorageClass with WaitForFirstConsumer will cause a pod to become schedulable.
func (pl *VolumeZone) isSchedulableAfterStorageClassAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedStorageClass, err := util.As[*storage.StorageClass](nil, newObj)
	if err != nil {
		return fwk.Queue, fmt.Errorf("unexpected objects in isSchedulableAfterStorageClassAdded: %w", err)
	}
	if (addedStorageClass.VolumeBindingMode == nil) || (*addedStorageClass.VolumeBindingMode != storage.VolumeBindingWaitForFirstConsumer) {
		logger.V(5).Info("StorageClass is created, but its VolumeBindingMode is not waitForFirstConsumer, which doesn't make the pod schedulable", "storageClass", klog.KObj(addedStorageClass), "pod", klog.KObj(pod))
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("StorageClass with waitForFirstConsumer mode was created and it might make this pod schedulable", "pod", klog.KObj(pod), "StorageClass", klog.KObj(addedStorageClass))
	return fwk.Queue, nil
}

// isSchedulableAfterPersistentVolumeChange is invoked whenever a PersistentVolume added or updated.
// It checks whether the change of PV has made a previously unschedulable pod schedulable.
// Changing the PV topology labels could cause the pod to become schedulable.
func (pl *VolumeZone) isSchedulableAfterPersistentVolumeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalPV, modifiedPV, err := util.As[*v1.PersistentVolume](oldObj, newObj)
	if err != nil {
		return fwk.Queue, fmt.Errorf("unexpected objects in isSchedulableAfterPersistentVolumeChange: %w", err)
	}
	if originalPV == nil {
		logger.V(5).Info("PV is newly created, which might make the pod schedulable")
		return fwk.Queue, nil
	}
	originalPVTopologies := pl.getPVTopologies(logger, originalPV)
	modifiedPVTopologies := pl.getPVTopologies(logger, modifiedPV)
	if !reflect.DeepEqual(originalPVTopologies, modifiedPVTopologies) {
		logger.V(5).Info("PV's topology was updated, which might make the pod schedulable.", "pod", klog.KObj(pod), "PV", klog.KObj(modifiedPV))
		return fwk.Queue, nil
	}

	logger.V(5).Info("PV was updated, but the topology is unchanged, which it doesn't make the pod schedulable", "pod", klog.KObj(pod), "PV", klog.KObj(modifiedPV))
	return fwk.QueueSkip, nil
}

// getPVTopologies retrieves pvTopology from a given PV and returns the array
// This function doesn't check spec.nodeAffinity
// because it's read-only after creation and thus cannot be updated
// and nodeAffinity is being handled in node affinity plugin
func (pl *VolumeZone) getPVTopologies(logger klog.Logger, pv *v1.PersistentVolume) []pvTopology {
	podPVTopologies := make([]pvTopology, 0)
	for _, key := range topologyLabels {
		if value, ok := pv.ObjectMeta.Labels[key]; ok {
			labelZonesSet, err := volumehelpers.LabelZonesToSet(value)
			if err != nil {
				logger.V(5).Info("failed to parse PV's topology label, ignoring the label", "label", fmt.Sprintf("%s:%s", key, value), "err", err)
				continue
			}
			podPVTopologies = append(podPVTopologies, pvTopology{
				pvName: pv.Name,
				key:    key,
				values: sets.Set[string](labelZonesSet),
			})
		}
	}
	return podPVTopologies
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, handle framework.Handle, fts feature.Features) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	pvLister := informerFactory.Core().V1().PersistentVolumes().Lister()
	pvcLister := informerFactory.Core().V1().PersistentVolumeClaims().Lister()
	scLister := informerFactory.Storage().V1().StorageClasses().Lister()
	return &VolumeZone{
		pvLister:                  pvLister,
		pvcLister:                 pvcLister,
		scLister:                  scLister,
		enableSchedulingQueueHint: fts.EnableSchedulingQueueHint,
	}, nil
}
