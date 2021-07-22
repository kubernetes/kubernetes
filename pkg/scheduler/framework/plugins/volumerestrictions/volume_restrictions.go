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

package volumerestrictions

import (
	"context"
	"fmt"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/internal/parallelize"
	"strings"
)

// VolumeRestrictions is a plugin that checks volume restrictions.
type VolumeRestrictions struct {
	parallelizer           parallelize.Parallelizer
	pvcLister              corelisters.PersistentVolumeClaimLister
	nodeInfoLister         framework.SharedLister
	enableReadWriteOncePod bool
}

var _ framework.PreFilterPlugin = &VolumeRestrictions{}
var _ framework.FilterPlugin = &VolumeRestrictions{}
var _ framework.EnqueueExtensions = &VolumeRestrictions{}

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = names.VolumeRestrictions

const (
	preFilterStateKey = "PreFilter" + Name
	// ErrReasonDiskConflict is used for NoDiskConflict predicate error.
	ErrReasonDiskConflict = "node(s) had no available disk"
	// ErrReasonReadWriteOncePodConflict is used when a pod is found using the same PVC with the ReadWriteOncePod access mode.
	ErrReasonReadWriteOncePodConflict = "node has pod using PersistentVolumeClaim with the same name and ReadWriteOncePod access mode"
	// InfoReasonReadWriteOncePodSame is used when a pod can be scheduled pods that use the same pvc on the same node
	InfoReasonReadWriteOncePodSame = "Can be scheduled to the same node"
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *VolumeRestrictions) Name() string {
	return Name
}

func isVolumeConflict(volume *v1.Volume, pod *v1.Pod) bool {
	for _, existingVolume := range pod.Spec.Volumes {
		// Same GCE disk mounted by multiple pods conflicts unless all pods mount it read-only.
		if volume.GCEPersistentDisk != nil && existingVolume.GCEPersistentDisk != nil {
			disk, existingDisk := volume.GCEPersistentDisk, existingVolume.GCEPersistentDisk
			if disk.PDName == existingDisk.PDName && !(disk.ReadOnly && existingDisk.ReadOnly) {
				return true
			}
		}

		if volume.AWSElasticBlockStore != nil && existingVolume.AWSElasticBlockStore != nil {
			if volume.AWSElasticBlockStore.VolumeID == existingVolume.AWSElasticBlockStore.VolumeID {
				return true
			}
		}

		if volume.ISCSI != nil && existingVolume.ISCSI != nil {
			iqn := volume.ISCSI.IQN
			eiqn := existingVolume.ISCSI.IQN
			// two ISCSI volumes are same, if they share the same iqn. As iscsi volumes are of type
			// RWO or ROX, we could permit only one RW mount. Same iscsi volume mounted by multiple Pods
			// conflict unless all other pods mount as read only.
			if iqn == eiqn && !(volume.ISCSI.ReadOnly && existingVolume.ISCSI.ReadOnly) {
				return true
			}
		}

		if volume.RBD != nil && existingVolume.RBD != nil {
			mon, pool, image := volume.RBD.CephMonitors, volume.RBD.RBDPool, volume.RBD.RBDImage
			emon, epool, eimage := existingVolume.RBD.CephMonitors, existingVolume.RBD.RBDPool, existingVolume.RBD.RBDImage
			// two RBDs images are the same if they share the same Ceph monitor, are in the same RADOS Pool, and have the same image name
			// only one read-write mount is permitted for the same RBD image.
			// same RBD image mounted by multiple Pods conflicts unless all Pods mount the image read-only
			if haveOverlap(mon, emon) && pool == epool && image == eimage && !(volume.RBD.ReadOnly && existingVolume.RBD.ReadOnly) {
				return true
			}
		}
	}

	return false
}

// haveOverlap searches two arrays and returns true if they have at least one common element; returns false otherwise.
func haveOverlap(a1, a2 []string) bool {
	if len(a1) > len(a2) {
		a1, a2 = a2, a1
	}
	m := make(sets.String)

	for _, val := range a1 {
		m.Insert(val)
	}
	for _, val := range a2 {
		if _, ok := m[val]; ok {
			return true
		}
	}

	return false
}

func (pl *VolumeRestrictions) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	if pl.enableReadWriteOncePod {
		return pl.isReadWriteOncePodAccessModeConflict(cycleState, pod)
	}
	return framework.NewStatus(framework.Success)
}

// isReadWriteOncePodAccessModeConflict checks if a pod uses a PVC with the ReadWriteOncePod access mode.
// This access mode restricts volume access to a single pod on a single node. Since only a single pod can
// use a ReadWriteOncePod PVC, mark any other pods attempting to use this PVC as UnschedulableAndUnresolvable.
func (pl *VolumeRestrictions) isReadWriteOncePodAccessModeConflict(cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	nodeInfos, err := pl.nodeInfoLister.NodeInfos().List()
	if err != nil {
		return framework.NewStatus(framework.Error, "error while getting node info")
	}
	pvcKeys, status := pl.GetPvcKeys(pod)
	if status != nil {
		return status
	}
	nodes := make([]*v1.Node, 0)
	processNode := func(i int) {
		nodeInfo := nodeInfos[i]
		pvcKeysNumber := len(pvcKeys)
		for key := range pvcKeys {
			refCount := nodeInfo.PVCRefCounts[key]
			if refCount != 0 {
				pvcKeysNumber--
			}
		}
		if pvcKeysNumber == 0 {
			nodes = append(nodes, nodeInfo.Node())
		}
	}
	pl.parallelizer.Until(context.Background(), len(nodeInfos), processNode)
	// Enforce ReadWriteOncePod access mode. This is also enforced during volume mount in kubelet.
	//if conflicts > 0 {
	//	return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonReadWriteOncePodConflict)
	//}

	// update cyclestate
	p := &preFilterState{
		nodes: nodes,
		pvck:  pvcKeys,
	}
	cycleState.Write(preFilterStateKey, p)
	return nil
}

func (pl *VolumeRestrictions) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func (pl *VolumeRestrictions) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToAdd *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	pvcKeys, status := pl.GetPvcKeys(podToSchedule)
	if status.IsUnschedulable() {
		return status
	}
	state.updateWithPod(nodeInfo, pvcKeys, 1)
	return nil
}

func (pl *VolumeRestrictions) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToRemove *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}
	pvcKeys, status := pl.GetPvcKeys(podToSchedule)
	if status.IsUnschedulable() {
		return status
	}
	state.updateWithPod(nodeInfo, pvcKeys, -1)
	return nil
}

func (pl *VolumeRestrictions) GetPvcKeys(pod *v1.Pod) (sets.String, *framework.Status) {
	var pvcKeys []string
	for _, volume := range pod.Spec.Volumes {
		if volume.PersistentVolumeClaim == nil {
			continue
		}

		pvc, err := pl.pvcLister.PersistentVolumeClaims(pod.Namespace).Get(volume.PersistentVolumeClaim.ClaimName)
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil, framework.NewStatus(framework.UnschedulableAndUnresolvable, err.Error())
			}
			return nil, framework.AsStatus(err)
		}

		if !v1helper.ContainsAccessMode(pvc.Spec.AccessModes, v1.ReadWriteOncePod) {
			continue
		}

		key := pod.Namespace + "/" + volume.PersistentVolumeClaim.ClaimName
		pvcKeys = append(pvcKeys, key)
	}
	return sets.NewString(pvcKeys...), nil
}

// Filter invoked at the filter extension point.
// It evaluates if a pod can fit due to the volumes it requests, and those that
// are already mounted. If there is already a volume mounted on that node,
// another pod can use the same volume by preempting
// This is GCE, Amazon EBS, ISCSI and Ceph RBD specific for now:
// - GCE PD allows multiple mounts as long as they're all read-only
// - AWS EBS forbids any two pods mounting the same volume ID
// - Ceph RBD forbids if any two pods share at least same monitor, and match pool and image, and the image is read-only
// - ISCSI forbids if any two pods share at least same IQN and ISCSI volume is read-only
func (pl *VolumeRestrictions) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}
	for i := range pod.Spec.Volumes {
		v := &pod.Spec.Volumes[i]
		// fast path if there is no conflict checking targets.
		if v.GCEPersistentDisk == nil && v.AWSElasticBlockStore == nil && v.RBD == nil && v.ISCSI == nil {
			continue
		}
		for _, ev := range nodeInfo.Pods {
			if isVolumeConflict(v, ev.Pod) {
				return framework.NewStatus(framework.Unschedulable, ErrReasonDiskConflict)
			}
		}
	}
	// Can be Scheduled in one node
	for i := 0; i < len(state.nodes); i++ {
		if nodeInfo.Node().Name == state.nodes[i].Name{
			return framework.NewStatus(framework.Success,InfoReasonReadWriteOncePodSame)
		}
	}
	return nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *VolumeRestrictions) EventsToRegister() []framework.ClusterEvent {
	return []framework.ClusterEvent{
		// Add/update/delete may change the state of the node pod.Spec.Volumes,
		// thereby changing the state of whether the pod can be preempted
		{Resource: framework.Pod, ActionType: framework.Delete | framework.Add | framework.Update},
		// A new Node may make a pod schedulable.
		{Resource: framework.Node, ActionType: framework.Add},
		// Pods may fail to schedule because the PVC it uses has not yet been created.
		// This PVC is required to exist to check its access modes.
		{Resource: framework.PersistentVolumeClaim, ActionType: framework.Add | framework.Update},
	}
}

// New initializes a new plugin and returns it.
func New(_ runtime.Object, handle framework.Handle, fts feature.Features) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	pvcLister := informerFactory.Core().V1().PersistentVolumeClaims().Lister()
	nodeInfoLister := handle.SnapshotSharedLister()

	return &VolumeRestrictions{
		parallelizer:           handle.Parallelizer(),
		pvcLister:              pvcLister,
		nodeInfoLister:         nodeInfoLister,
		enableReadWriteOncePod: fts.EnableReadWriteOncePod,
	}, nil
}

// canBePreempt only verifies the volume, not the resource/affinity..
func canBePreempt(nodeInfo *framework.NodeInfo, key string) bool {
	namespace, pvcName := strings.Split(key, "/")[0], strings.Split(key, "/")[1]
	anotherNodeInfo := nodeInfo.Clone()
	for _, podInfo := range anotherNodeInfo.Pods {
		pod := podInfo.Pod
		if pod.Namespace != namespace {
			continue
		}
		for _, vol := range pod.Spec.Volumes {
			if vol.PersistentVolumeClaim.ClaimName == pvcName {
				return true
			}
		}
	}
	return false
}

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	nodes []*v1.Node
	// If the AccessModes of the pvc used by a pod is ReadWriteOncePod,
	// pod's namespace and pvcName are added to pvck
	pvck sets.String
}

func (s *preFilterState) Clone() framework.StateData {
	return s
}

func getPreFilterState(cycleState *framework.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// preFilterState doesn't exist, likely PreFilter wasn't invoked.
		return nil, fmt.Errorf("error reading %q from cycleState: %w", Name, err)
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to volumerestrictions.state error", c)
	}
	return s, nil
}

func (s *preFilterState) updateWithPod(nodeInfo *framework.NodeInfo, pvcKeys sets.String, multiplier int64) {
	if multiplier == 1 {
		// update nodes
		pvcKeysNumber := len(pvcKeys)
		for key := range pvcKeys {
			refCount := nodeInfo.PVCRefCounts[key]
			if refCount != 0 {
				pvcKeysNumber--
			}
		}
		if pvcKeysNumber == 0 {
			s.nodes = append(s.nodes, nodeInfo.Node())
		}
		s.pvck = pvcKeys.Union(s.pvck)
	} else {
		flag := false
		for key := range pvcKeys {
			refCount := nodeInfo.PVCRefCounts[key]
			if refCount != 0 {
				flag = true
				break
			}
		}
		if flag {
			for i := range s.nodes {
				if s.nodes[i].Name == nodeInfo.Node().Name {
					s.nodes = append(s.nodes[:i], s.nodes[i+1:]...)
				}
			}
		}
		s.pvck = s.pvck.Difference(pvcKeys)
	}
}
