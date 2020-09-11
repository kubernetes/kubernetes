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

package volumebinding

import (
	"context"
	"errors"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

const (
	// DefaultBindTimeoutSeconds defines the default bind timeout in seconds
	DefaultBindTimeoutSeconds = 600

	stateKey framework.StateKey = Name
)

// the state is initialized in PreFilter phase. because we save the pointer in
// framework.CycleState, in the later phases we don't need to call Write method
// to update the value
type stateData struct {
	skip         bool // set true if pod does not have PVCs
	boundClaims  []*v1.PersistentVolumeClaim
	claimsToBind []*v1.PersistentVolumeClaim
	allBound     bool
	// podVolumesByNode holds the pod's volume information found in the Filter
	// phase for each node
	// it's initialized in the PreFilter phase
	podVolumesByNode map[string]*scheduling.PodVolumes
}

func (d *stateData) Clone() framework.StateData {
	return d
}

// VolumeBinding is a plugin that binds pod volumes in scheduling.
// In the Filter phase, pod binding cache is created for the pod and used in
// Reserve and PreBind phases.
type VolumeBinding struct {
	Binder                               scheduling.SchedulerVolumeBinder
	PVCLister                            corelisters.PersistentVolumeClaimLister
	GenericEphemeralVolumeFeatureEnabled bool
}

var _ framework.PreFilterPlugin = &VolumeBinding{}
var _ framework.FilterPlugin = &VolumeBinding{}
var _ framework.ReservePlugin = &VolumeBinding{}
var _ framework.PreBindPlugin = &VolumeBinding{}

// Name is the name of the plugin used in Registry and configurations.
const Name = "VolumeBinding"

// Name returns name of the plugin. It is used in logs, etc.
func (pl *VolumeBinding) Name() string {
	return Name
}

// podHasPVCs returns 2 values:
// - the first one to denote if the given "pod" has any PVC defined.
// - the second one to return any error if the requested PVC is illegal.
func (pl *VolumeBinding) podHasPVCs(pod *v1.Pod) (bool, error) {
	hasPVC := false
	for _, vol := range pod.Spec.Volumes {
		var pvcName string
		ephemeral := false
		switch {
		case vol.PersistentVolumeClaim != nil:
			pvcName = vol.PersistentVolumeClaim.ClaimName
		case vol.Ephemeral != nil && pl.GenericEphemeralVolumeFeatureEnabled:
			pvcName = pod.Name + "-" + vol.Name
			ephemeral = true
		default:
			// Volume is not using a PVC, ignore
			continue
		}
		hasPVC = true
		pvc, err := pl.PVCLister.PersistentVolumeClaims(pod.Namespace).Get(pvcName)
		if err != nil {
			// The error has already enough context ("persistentvolumeclaim "myclaim" not found")
			return hasPVC, err
		}

		if pvc.DeletionTimestamp != nil {
			return hasPVC, fmt.Errorf("persistentvolumeclaim %q is being deleted", pvc.Name)
		}

		if ephemeral && !metav1.IsControlledBy(pvc, pod) {
			return hasPVC, fmt.Errorf("persistentvolumeclaim %q was not created for the pod", pvc.Name)
		}
	}
	return hasPVC, nil
}

// PreFilter invoked at the prefilter extension point to check if pod has all
// immediate PVCs bound. If not all immediate PVCs are bound, an
// UnschedulableAndUnresolvable is returned.
func (pl *VolumeBinding) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) *framework.Status {
	// If pod does not reference any PVC, we don't need to do anything.
	if hasPVC, err := pl.podHasPVCs(pod); err != nil {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, err.Error())
	} else if !hasPVC {
		state.Write(stateKey, &stateData{skip: true})
		return nil
	}
	boundClaims, claimsToBind, unboundClaimsImmediate, err := pl.Binder.GetPodVolumes(pod)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	if len(unboundClaimsImmediate) > 0 {
		// Return UnschedulableAndUnresolvable error if immediate claims are
		// not bound. Pod will be moved to active/backoff queues once these
		// claims are bound by PV controller.
		status := framework.NewStatus(framework.UnschedulableAndUnresolvable)
		status.AppendReason("pod has unbound immediate PersistentVolumeClaims")
		return status
	}
	state.Write(stateKey, &stateData{boundClaims: boundClaims, claimsToBind: claimsToBind, podVolumesByNode: make(map[string]*scheduling.PodVolumes)})
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *VolumeBinding) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func getStateData(cs *framework.CycleState) (*stateData, error) {
	state, err := cs.Read(stateKey)
	if err != nil {
		return nil, err
	}
	s, ok := state.(*stateData)
	if !ok {
		return nil, errors.New("unable to convert state into stateData")
	}
	return s, nil
}

// Filter invoked at the filter extension point.
// It evaluates if a pod can fit due to the volumes it requests,
// for both bound and unbound PVCs.
//
// For PVCs that are bound, then it checks that the corresponding PV's node affinity is
// satisfied by the given node.
//
// For PVCs that are unbound, it tries to find available PVs that can satisfy the PVC requirements
// and that the PV node affinity is satisfied by the given node.
//
// If storage capacity tracking is enabled, then enough space has to be available
// for the node and volumes that still need to be created.
//
// The predicate returns true if all bound PVCs have compatible PVs with the node, and if all unbound
// PVCs can be matched with an available and node-compatible PV.
func (pl *VolumeBinding) Filter(ctx context.Context, cs *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}

	state, err := getStateData(cs)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	if state.skip {
		return nil
	}

	podVolumes, reasons, err := pl.Binder.FindPodVolumes(pod, state.boundClaims, state.claimsToBind, node)

	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	if len(reasons) > 0 {
		status := framework.NewStatus(framework.UnschedulableAndUnresolvable)
		for _, reason := range reasons {
			status.AppendReason(string(reason))
		}
		return status
	}

	cs.Lock()
	state.podVolumesByNode[node.Name] = podVolumes
	cs.Unlock()
	return nil
}

// Reserve reserves volumes of pod and saves binding status in cycle state.
func (pl *VolumeBinding) Reserve(ctx context.Context, cs *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	state, err := getStateData(cs)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	// we don't need to hold the lock as only one node will be reserved for the given pod
	podVolumes, ok := state.podVolumesByNode[nodeName]
	if ok {
		allBound, err := pl.Binder.AssumePodVolumes(pod, nodeName, podVolumes)
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}
		state.allBound = allBound
	} else {
		// may not exist if the pod does not reference any PVC
		state.allBound = true
	}
	return nil
}

// PreBind will make the API update with the assumed bindings and wait until
// the PV controller has completely finished the binding operation.
//
// If binding errors, times out or gets undone, then an error will be returned to
// retry scheduling.
func (pl *VolumeBinding) PreBind(ctx context.Context, cs *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	s, err := getStateData(cs)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	if s.allBound {
		// no need to bind volumes
		return nil
	}
	// we don't need to hold the lock as only one node will be pre-bound for the given pod
	podVolumes, ok := s.podVolumesByNode[nodeName]
	if !ok {
		return framework.NewStatus(framework.Error, fmt.Sprintf("no pod volumes found for node %q", nodeName))
	}
	klog.V(5).Infof("Trying to bind volumes for pod \"%v/%v\"", pod.Namespace, pod.Name)
	err = pl.Binder.BindPodVolumes(pod, podVolumes)
	if err != nil {
		klog.V(1).Infof("Failed to bind volumes for pod \"%v/%v\": %v", pod.Namespace, pod.Name, err)
		return framework.NewStatus(framework.Error, err.Error())
	}
	klog.V(5).Infof("Success binding volumes for pod \"%v/%v\"", pod.Namespace, pod.Name)
	return nil
}

// Unreserve clears assumed PV and PVC cache.
// It's idempotent, and does nothing if no cache found for the given pod.
func (pl *VolumeBinding) Unreserve(ctx context.Context, cs *framework.CycleState, pod *v1.Pod, nodeName string) {
	s, err := getStateData(cs)
	if err != nil {
		return
	}
	// we don't need to hold the lock as only one node may be unreserved
	podVolumes, ok := s.podVolumesByNode[nodeName]
	if !ok {
		return
	}
	pl.Binder.RevertAssumedPodVolumes(podVolumes)
	return
}

// New initializes a new plugin and returns it.
func New(plArgs runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
	args, ok := plArgs.(*config.VolumeBindingArgs)
	if !ok {
		return nil, fmt.Errorf("want args to be of type VolumeBindingArgs, got %T", plArgs)
	}
	if err := validateArgs(args); err != nil {
		return nil, err
	}
	podInformer := fh.SharedInformerFactory().Core().V1().Pods()
	nodeInformer := fh.SharedInformerFactory().Core().V1().Nodes()
	pvcInformer := fh.SharedInformerFactory().Core().V1().PersistentVolumeClaims()
	pvInformer := fh.SharedInformerFactory().Core().V1().PersistentVolumes()
	storageClassInformer := fh.SharedInformerFactory().Storage().V1().StorageClasses()
	csiNodeInformer := fh.SharedInformerFactory().Storage().V1().CSINodes()
	var capacityCheck *scheduling.CapacityCheck
	if utilfeature.DefaultFeatureGate.Enabled(features.CSIStorageCapacity) {
		capacityCheck = &scheduling.CapacityCheck{
			CSIDriverInformer:          fh.SharedInformerFactory().Storage().V1().CSIDrivers(),
			CSIStorageCapacityInformer: fh.SharedInformerFactory().Storage().V1alpha1().CSIStorageCapacities(),
		}
	}
	binder := scheduling.NewVolumeBinder(fh.ClientSet(), podInformer, nodeInformer, csiNodeInformer, pvcInformer, pvInformer, storageClassInformer, capacityCheck, time.Duration(args.BindTimeoutSeconds)*time.Second)
	return &VolumeBinding{
		Binder:                               binder,
		PVCLister:                            pvcInformer.Lister(),
		GenericEphemeralVolumeFeatureEnabled: utilfeature.DefaultFeatureGate.Enabled(features.GenericEphemeralVolume),
	}, nil
}

func validateArgs(args *config.VolumeBindingArgs) error {
	if args.BindTimeoutSeconds <= 0 {
		return fmt.Errorf("invalid BindTimeoutSeconds: %d, must be positive integer", args.BindTimeoutSeconds)
	}
	return nil
}
