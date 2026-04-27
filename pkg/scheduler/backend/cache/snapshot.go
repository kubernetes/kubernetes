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

package cache

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// placementNodes stores nodes that are present in the current placement.
// Placement is a limited set of nodes that is used in the pod group scheduling cycle.
type placementNodes struct {
	// nodeInfoList contains the list of nodes in the placement.
	// This is useful for quickly returning the entire list to the caller.
	nodeInfoList []fwk.NodeInfo
	// nodeInfoSet contains the set of nodes in the placement.
	// This is useful for quickly checking if a node belongs the the placement.
	nodeInfoSet sets.Set[string]
}

// Snapshot is a snapshot of cache NodeInfo and NodeTree order. The scheduler takes a
// snapshot at the beginning of each scheduling cycle and uses it for its operations in that cycle.
type Snapshot struct {
	// nodeInfoMap a map of node name to a snapshot of its NodeInfo.
	nodeInfoMap map[string]*framework.NodeInfo
	// nodeInfoList is the list of nodes as ordered in the cache's nodeTree.
	nodeInfoList []fwk.NodeInfo
	// havePodsWithAffinityNodeInfoList is the list of nodes with at least one pod declaring affinity terms.
	havePodsWithAffinityNodeInfoList []fwk.NodeInfo
	// havePodsWithRequiredAntiAffinityNodeInfoList is the list of nodes with at least one pod declaring
	// required anti-affinity terms.
	havePodsWithRequiredAntiAffinityNodeInfoList []fwk.NodeInfo
	// usedPVCSet contains a set of PVC names that have one or more scheduled pods using them,
	// keyed in the format "namespace/name".
	usedPVCSet sets.Set[string]
	generation int64
	// assumedPods maps a pod key to an assumed pod object during a single pod group scheduling cycle.
	// This map should be emptied before the next cycle starts.
	assumedPods map[string]*v1.Pod
	// podGroupStates maps a pod group key to a snapshot of its state, used during a pod group scheduling cycle.
	podGroupStates map[podGroupKey]*podGroupStateSnapshot
	// placementNodes stores nodes that are present in the current placement.
	// If placement is not set, this is nil.
	// It should only be set in the pod group scheduling cycle, when checking if pod group can be scheduled within the placement.
	// This field should be cleared once the pod group has been checked for the placement.
	placementNodes *placementNodes
	// genericWorkloadEnabled stores the GenericWorkload feature gate value.
	genericWorkloadEnabled bool
	// hasBackup holds information whether backup was performed and
	// restore was not performed yet.
	hasBackup bool
}

var _ fwk.SharedLister = &Snapshot{}

// NewEmptySnapshot initializes a Snapshot struct and returns it.
func NewEmptySnapshot() *Snapshot {
	return &Snapshot{
		nodeInfoMap:            make(map[string]*framework.NodeInfo),
		usedPVCSet:             sets.New[string](),
		assumedPods:            make(map[string]*v1.Pod),
		podGroupStates:         make(map[podGroupKey]*podGroupStateSnapshot),
		genericWorkloadEnabled: utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload),
	}
}

// NewSnapshot initializes a Snapshot struct and returns it.
func NewSnapshot(pods []*v1.Pod, nodes []*v1.Node) *Snapshot {
	nodeInfoMap := createNodeInfoMap(pods, nodes)
	nodeInfoList := make([]fwk.NodeInfo, 0, len(nodeInfoMap))
	havePodsWithAffinityNodeInfoList := make([]fwk.NodeInfo, 0, len(nodeInfoMap))
	havePodsWithRequiredAntiAffinityNodeInfoList := make([]fwk.NodeInfo, 0, len(nodeInfoMap))
	for _, v := range nodeInfoMap {
		nodeInfoList = append(nodeInfoList, v)
		if len(v.PodsWithAffinity) > 0 {
			havePodsWithAffinityNodeInfoList = append(havePodsWithAffinityNodeInfoList, v)
		}
		if len(v.PodsWithRequiredAntiAffinity) > 0 {
			havePodsWithRequiredAntiAffinityNodeInfoList = append(havePodsWithRequiredAntiAffinityNodeInfoList, v)
		}
	}

	s := NewEmptySnapshot()
	s.nodeInfoMap = nodeInfoMap
	s.nodeInfoList = nodeInfoList
	s.havePodsWithAffinityNodeInfoList = havePodsWithAffinityNodeInfoList
	s.havePodsWithRequiredAntiAffinityNodeInfoList = havePodsWithRequiredAntiAffinityNodeInfoList
	s.usedPVCSet = createUsedPVCSet(pods)
	if s.genericWorkloadEnabled {
		s.podGroupStates = createPodGroupStates(pods)
	}

	return s
}

// createPodGroupStates builds the initial pod group state snapshot map from a list of pods.
func createPodGroupStates(pods []*v1.Pod) map[podGroupKey]*podGroupStateSnapshot {
	podGroupStates := make(map[podGroupKey]*podGroupStateSnapshot)
	for _, pod := range pods {
		if pod.Spec.SchedulingGroup == nil {
			continue
		}
		key := newPodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
		pgs, ok := podGroupStates[key]
		if !ok {
			pgs = &podGroupStateSnapshot{podGroupStateData: newPodGroupStateData()}
			podGroupStates[key] = pgs
		}
		pgs.addPod(pod)
	}
	return podGroupStates
}

// RestoreSnapshot is a function that can be used to restore the snapshot to the state
// before the backup was taken.
type RestoreSnapshot func()

// BackupSnapshot provides a way to temporarily backup the snapshot's state
// and returns a restore function. This is primarily used in workload-aware
// preemption to simulate pod group preemption by mutating deep copies of NodeInfos.
// Backups cannot be stacked, i.e., only one backup can be made without restoring
// the snapshot first.
// Restoring backup when the placement is set is not supported and can lead to
// undefined behavior.
func (s *Snapshot) BackupSnapshot() (RestoreSnapshot, error) {
	if s.hasBackup {
		return nil, fmt.Errorf("cannot stack backups")
	}
	origNodeInfoMap := s.nodeInfoMap
	origNodeInfoList := s.nodeInfoList
	origHavePodsWithAffinityNodeInfoList := s.havePodsWithAffinityNodeInfoList
	origHavePodsWithRequiredAntiAffinityNodeInfoList := s.havePodsWithRequiredAntiAffinityNodeInfoList

	clonedNodeInfoMap := make(map[string]*framework.NodeInfo, len(s.nodeInfoMap))
	for k, v := range s.nodeInfoMap {
		clonedNodeInfoMap[k] = v.Snapshot().(*framework.NodeInfo)
	}

	clonedNodeInfoList := make([]fwk.NodeInfo, 0, len(clonedNodeInfoMap))
	clonedHavePodsWithAffinityNodeInfoList := make([]fwk.NodeInfo, 0, len(clonedNodeInfoMap))
	clonedHavePodsWithRequiredAntiAffinityNodeInfoList := make([]fwk.NodeInfo, 0, len(clonedNodeInfoMap))

	for _, v := range s.nodeInfoList {
		clonedNode := clonedNodeInfoMap[v.Node().Name]
		clonedNodeInfoList = append(clonedNodeInfoList, clonedNode)
		if len(clonedNode.PodsWithAffinity) > 0 {
			clonedHavePodsWithAffinityNodeInfoList = append(clonedHavePodsWithAffinityNodeInfoList, clonedNode)
		}
		if len(clonedNode.PodsWithRequiredAntiAffinity) > 0 {
			clonedHavePodsWithRequiredAntiAffinityNodeInfoList = append(clonedHavePodsWithRequiredAntiAffinityNodeInfoList, clonedNode)
		}
	}

	s.hasBackup = true
	s.nodeInfoMap = clonedNodeInfoMap
	s.nodeInfoList = clonedNodeInfoList
	s.havePodsWithAffinityNodeInfoList = clonedHavePodsWithAffinityNodeInfoList
	s.havePodsWithRequiredAntiAffinityNodeInfoList = clonedHavePodsWithRequiredAntiAffinityNodeInfoList

	return func() {
		s.hasBackup = false
		s.nodeInfoMap = origNodeInfoMap
		s.nodeInfoList = origNodeInfoList
		s.havePodsWithAffinityNodeInfoList = origHavePodsWithAffinityNodeInfoList
		s.havePodsWithRequiredAntiAffinityNodeInfoList = origHavePodsWithRequiredAntiAffinityNodeInfoList
	}, nil
}

// createNodeInfoMap obtains a list of pods and pivots that list into a map
// where the keys are node names and the values are the aggregated information
// for that node.
func createNodeInfoMap(pods []*v1.Pod, nodes []*v1.Node) map[string]*framework.NodeInfo {
	nodeNameToInfo := make(map[string]*framework.NodeInfo)
	for _, pod := range pods {
		nodeName := pod.Spec.NodeName
		if _, ok := nodeNameToInfo[nodeName]; !ok {
			nodeNameToInfo[nodeName] = framework.NewNodeInfo()
		}
		nodeNameToInfo[nodeName].AddPod(pod)
	}
	imageExistenceMap := createImageExistenceMap(nodes)

	for _, node := range nodes {
		if _, ok := nodeNameToInfo[node.Name]; !ok {
			nodeNameToInfo[node.Name] = framework.NewNodeInfo()
		}
		nodeInfo := nodeNameToInfo[node.Name]
		nodeInfo.SetNode(node)
		nodeInfo.ImageStates = getNodeImageStates(node, imageExistenceMap)
	}
	return nodeNameToInfo
}

func createUsedPVCSet(pods []*v1.Pod) sets.Set[string] {
	usedPVCSet := sets.New[string]()
	for _, pod := range pods {
		if pod.Spec.NodeName == "" {
			continue
		}

		for _, v := range pod.Spec.Volumes {
			if v.PersistentVolumeClaim == nil {
				continue
			}

			key := framework.GetNamespacedName(pod.Namespace, v.PersistentVolumeClaim.ClaimName)
			usedPVCSet.Insert(key)
		}
	}
	return usedPVCSet
}

// getNodeImageStates returns the given node's image states based on the given imageExistence map.
func getNodeImageStates(node *v1.Node, imageExistenceMap map[string]sets.Set[string]) map[string]*fwk.ImageStateSummary {
	imageStates := make(map[string]*fwk.ImageStateSummary)

	for _, image := range node.Status.Images {
		for _, name := range image.Names {
			imageStates[name] = &fwk.ImageStateSummary{
				Size:     image.SizeBytes,
				NumNodes: imageExistenceMap[name].Len(),
			}
		}
	}
	return imageStates
}

// createImageExistenceMap returns a map recording on which nodes the images exist, keyed by the images' names.
func createImageExistenceMap(nodes []*v1.Node) map[string]sets.Set[string] {
	imageExistenceMap := make(map[string]sets.Set[string])
	for _, node := range nodes {
		for _, image := range node.Status.Images {
			for _, name := range image.Names {
				if _, ok := imageExistenceMap[name]; !ok {
					imageExistenceMap[name] = sets.New(node.Name)
				} else {
					imageExistenceMap[name].Insert(node.Name)
				}
			}
		}
	}
	return imageExistenceMap
}

// NodeInfos returns a NodeInfoLister.
func (s *Snapshot) NodeInfos() fwk.NodeInfoLister {
	return s
}

// StorageInfos returns a StorageInfoLister.
func (s *Snapshot) StorageInfos() fwk.StorageInfoLister {
	return s
}

// PodGroupStates returns a PodGroupStateLister.
func (s *Snapshot) PodGroupStates() fwk.PodGroupStateLister {
	return &podGroupStateSnapshotLister{podGroupStates: s.podGroupStates}
}

var _ fwk.PodGroupStateLister = &podGroupStateSnapshotLister{}

type podGroupStateSnapshotLister struct {
	podGroupStates map[podGroupKey]*podGroupStateSnapshot
}

// Get returns the pod group state from the snapshot for the given pod group.
func (l *podGroupStateSnapshotLister) Get(namespace string, podGroupName string) (fwk.PodGroupState, error) {
	key := newPodGroupKey(namespace, podGroupName)
	state, ok := l.podGroupStates[key]
	if !ok {
		return nil, fmt.Errorf("pod group state not found for pod group %s", key)
	}
	return state, nil
}

// NumNodesInPlacement returns the number of nodes in the snapshot for the current placement.
// If no placement is set, it returns the number of nodes in the snapshot.
// This function is not thread safe so it should be executed when no other routines can write to the snapshot.
func (s *Snapshot) NumNodesInPlacement() int {
	if s.placementNodes != nil {
		return len(s.placementNodes.nodeInfoList)
	}
	return len(s.nodeInfoList)
}

// List returns the list of nodes in the snapshot.
func (s *Snapshot) List() ([]fwk.NodeInfo, error) {
	return s.nodeInfoList, nil
}

// HavePodsWithAffinityList returns the list of nodes with at least one pod with inter-pod affinity
func (s *Snapshot) HavePodsWithAffinityList() ([]fwk.NodeInfo, error) {
	return s.havePodsWithAffinityNodeInfoList, nil
}

// HavePodsWithRequiredAntiAffinityList returns the list of nodes with at least one pod with
// required inter-pod anti-affinity
func (s *Snapshot) HavePodsWithRequiredAntiAffinityList() ([]fwk.NodeInfo, error) {
	return s.havePodsWithRequiredAntiAffinityNodeInfoList, nil
}

// Get returns the NodeInfo of the given node name.
func (s *Snapshot) Get(nodeName string) (fwk.NodeInfo, error) {
	if v, ok := s.nodeInfoMap[nodeName]; ok && v.Node() != nil {
		return v, nil
	}
	return nil, fmt.Errorf("nodeinfo not found for node name %q", nodeName)
}

func (s *Snapshot) IsPVCUsedByPods(key string) bool {
	return s.usedPVCSet.Has(key)
}

// AssumePod assumes a given pod in the snapshot.
// ForgetPod should be called on the snapshot before syncing it with the cache.
// This function is not thread safe, so it should be executed when no other routines can write/read from the snapshot.
func (s *Snapshot) AssumePod(podInfo *framework.PodInfo) error {
	pod := podInfo.Pod
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}
	nodeInfo, ok := s.nodeInfoMap[pod.Spec.NodeName]
	if !ok {
		nodeInfo = framework.NewNodeInfo()
		s.nodeInfoMap[pod.Spec.NodeName] = nodeInfo
	}
	// Calling AddPodInfo increases the Generation number of the nodeInfo.
	// Since this operation only affects the snapshot,
	// we should keep the old number to remain consistent with the cached value.
	oldGeneration := nodeInfo.Generation
	nodeInfo.AddPodInfo(podInfo)
	nodeInfo.Generation = oldGeneration
	s.assumedPods[key] = pod
	// Update the pod group state in the snapshot if the pod belongs to a pod group.
	if !s.genericWorkloadEnabled || pod.Spec.SchedulingGroup == nil {
		return nil
	}
	pgKey := newPodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	if pgs, ok := s.podGroupStates[pgKey]; ok {
		pgs.assumePod(pod)
	}
	return nil
}

// ForgetPod forgets a given pod from the snapshot.
// This function is not thread safe, so it should be executed when no other routines can write/read from the snapshot.
func (s *Snapshot) ForgetPod(logger klog.Logger, pod *v1.Pod) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}
	assumedPod, ok := s.assumedPods[key]
	if !ok {
		return fmt.Errorf("assumed pod %q not found in the snapshot", key)
	}
	delete(s.assumedPods, key)
	nodeName := assumedPod.Spec.NodeName
	if nodeInfo, ok := s.nodeInfoMap[nodeName]; ok {
		// Calling RemovePod increases the Generation number of the nodeInfo.
		// Since this operation only affects the snapshot,
		// we should keep the old number to remain consistent with the cached value.
		oldGeneration := nodeInfo.Generation
		err := nodeInfo.RemovePod(logger, pod)
		if err != nil {
			return err
		}
		nodeInfo.Generation = oldGeneration
		if len(nodeInfo.Pods) == 0 && nodeInfo.Node() == nil {
			delete(s.nodeInfoMap, nodeName)
		}
	}
	// Update the pod group state in the snapshot if the pod belongs to a pod group.
	if !s.genericWorkloadEnabled || pod.Spec.SchedulingGroup == nil {
		return nil
	}
	pgKey := newPodGroupKey(assumedPod.Namespace, *assumedPod.Spec.SchedulingGroup.PodGroupName)
	if pgs, ok := s.podGroupStates[pgKey]; ok {
		pgs.forgetPod(assumedPod.UID)
	}
	return nil
}

// forgetAllAssumedPods forgets all assumed pods from the snapshot.
// This function is not thread safe, so it should be executed when no other routines can write/read from the snapshot.
func (s *Snapshot) forgetAllAssumedPods(logger klog.Logger) {
	if len(s.assumedPods) == 0 {
		return
	}
	for _, pod := range s.assumedPods {
		err := s.ForgetPod(logger, pod)
		if err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Failed to forget assumed pod")
		}
	}
	logger.Error(nil, "Found assumed pods in the snapshot that were not forgotten", "assumedPodsCount", len(s.assumedPods))
}

// AssumePlacement sets placement context in the snapshot.
// The snapshot should not be updated if a placement is assumed.
// The placement should be unset with ForgetPlacement once it's no longer needed.
// This function should only be used by the scheduler to limit the node candidates for scheduling.
// This function is not thread safe, so it should be executed when no other routines can write/read from the snapshot.
func (s *Snapshot) AssumePlacement(placement *fwk.Placement) error {
	if len(placement.Nodes) == len(s.nodeInfoList) {
		// All nodes in placement, meaning we can treat it the same as no placement and avoid copying the buffer.
		s.ForgetPlacement()
		return nil
	}
	s.placementNodes = &placementNodes{
		nodeInfoList: placement.Nodes,
		nodeInfoSet:  sets.New[string](),
	}
	for _, node := range placement.Nodes {
		snapshotNode, ok := s.nodeInfoMap[node.Node().Name]
		if !ok {
			s.ForgetPlacement()
			return fmt.Errorf("node %s in placement is not present in snapshot", node.Node().Name)
		}
		if snapshotNode != node {
			s.ForgetPlacement()
			return fmt.Errorf("node %s in placement is not the same instance as in the snapshot", node.Node().Name)
		}
		s.placementNodes.nodeInfoSet.Insert(node.Node().Name)
	}
	return nil
}

// ForgetPlacement clears placement.
// This function should only be used by the scheduler once the pods have been considered for that placement.
// This function is not thread safe, so it should be executed when no other routines can write/read from the snapshot.
func (s *Snapshot) ForgetPlacement() {
	s.placementNodes = nil
}

// GetNodeInPlacement returns the NodeInfo of the given node name within the current placement.
// If no placement is set, this is equivalent to Get.
// Placement is typically set in the pod group scheduling cycle.
// This function should only be used by the scheduler to limit the node candidates for scheduling.
// Plugins normally do not need this information.
// This function is not thread safe so it should be executed when no other routines can write to the snapshot.
func (s *Snapshot) GetNodeInPlacement(nodeName string) (fwk.NodeInfo, error) {
	if s.placementNodes == nil || s.placementNodes.nodeInfoSet.Has(nodeName) {
		return s.Get(nodeName)
	}
	return nil, fmt.Errorf("node %q not found in placement", nodeName)
}

// ListNodesInPlacement returns the list of nodes in the snapshot within the current placement.
// If no placement is set, this is equivalent to List.
// Placement is typically set in the pod group scheduling cycle.
// This function should only be used by the scheduler to limit the node candidates for scheduling.
// Plugins normally do not need this information.
// This function is not thread safe so it should be executed when no other routines can write to the snapshot.
func (s *Snapshot) ListNodesInPlacement() ([]fwk.NodeInfo, error) {
	if s.placementNodes == nil {
		return s.List()
	}
	return s.placementNodes.nodeInfoList, nil
}
