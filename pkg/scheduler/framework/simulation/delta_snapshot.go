package simulation

import (
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	schedulerinterface "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

var errNodeNotFound = errors.New("node not found")

// DeltaSnapshotStore is an implementation of ClusterSnapshotStore optimized for typical Cluster Autoscaler usage - (fork, add stuff, revert), repeated many times per loop.
//
// Complexity of some notable operations:
//
//	fork - O(1)
//	revert - O(1)
//	commit - O(n)
//	list all pods (no filtering) - O(n), cached
//	list all pods (with filtering) - O(n)
//	list node infos - O(n), cached
//
// Watch out for:
//
// * Node deletions, pod additions & deletions - invalidates cache of current snapshot
// (when forked affects delta, but not base.)
//
// * Pod affinity - causes scheduler framework to list pods with non-empty selector,
// so basic caching doesn't help.
//
// * DRA objects are tracked in the separate snapshot and while they don't exactly share
// memory and time complexities of DeltaSnapshotStore - they are optimized for
// cluster autoscaler operations
type DeltaSnapshotStore struct {
	data        *internalDeltaSnapshotData
	parallelism int
}

var _ fwk.SharedLister = &DeltaSnapshotStore{}

type deltaSnapshotStoreNodeLister DeltaSnapshotStore
type deltaSnapshotStoreStorageLister DeltaSnapshotStore

type internalDeltaSnapshotData struct {
	baseData *internalDeltaSnapshotData

	addedNodeInfoMap    map[string]schedulerinterface.NodeInfo
	modifiedNodeInfoMap map[string]schedulerinterface.NodeInfo
	deletedNodeInfos    map[string]bool

	nodeInfoList                     []schedulerinterface.NodeInfo
	havePodsWithAffinity             []schedulerinterface.NodeInfo
	havePodsWithRequiredAntiAffinity []schedulerinterface.NodeInfo
	pvcNamespaceMap                  map[string]int
}

func newInternalDeltaSnapshotData() *internalDeltaSnapshotData {
	return &internalDeltaSnapshotData{
		addedNodeInfoMap:    make(map[string]schedulerinterface.NodeInfo),
		modifiedNodeInfoMap: make(map[string]schedulerinterface.NodeInfo),
		deletedNodeInfos:    make(map[string]bool),
	}
}

func (data *internalDeltaSnapshotData) getNodeInfo(name string) (schedulerinterface.NodeInfo, bool) {
	if data == nil {
		return nil, false
	}
	if nodeInfo, found := data.getNodeInfoLocal(name); found {
		return nodeInfo, found
	}
	if data.deletedNodeInfos[name] {
		return nil, false
	}
	return data.baseData.getNodeInfo(name)
}

func (data *internalDeltaSnapshotData) getNodeInfoLocal(name string) (schedulerinterface.NodeInfo, bool) {
	if data == nil {
		return nil, false
	}
	if nodeInfo, found := data.addedNodeInfoMap[name]; found {
		return nodeInfo, true
	}
	if nodeInfo, found := data.modifiedNodeInfoMap[name]; found {
		return nodeInfo, true
	}
	return nil, false
}

func (data *internalDeltaSnapshotData) getNodeInfoList() []schedulerinterface.NodeInfo {
	if data == nil {
		return nil
	}
	if data.nodeInfoList == nil {
		data.nodeInfoList = data.buildNodeInfoList()
	}
	return data.nodeInfoList
}

// Contains costly copying throughout the struct chain. Use wisely.
func (data *internalDeltaSnapshotData) buildNodeInfoList() []schedulerinterface.NodeInfo {
	baseList := data.baseData.getNodeInfoList()
	totalLen := len(baseList) + len(data.addedNodeInfoMap)
	var nodeInfoList []schedulerinterface.NodeInfo

	if len(data.deletedNodeInfos) > 0 || len(data.modifiedNodeInfoMap) > 0 {
		nodeInfoList = make([]schedulerinterface.NodeInfo, 0, totalLen)
		for _, bni := range baseList {
			if data.deletedNodeInfos[bni.Node().Name] {
				continue
			}
			if mni, found := data.modifiedNodeInfoMap[bni.Node().Name]; found {
				nodeInfoList = append(nodeInfoList, mni)
				continue
			}
			nodeInfoList = append(nodeInfoList, bni)
		}
	} else {
		nodeInfoList = make([]schedulerinterface.NodeInfo, len(baseList), totalLen)
		copy(nodeInfoList, baseList)
	}

	for _, ani := range data.addedNodeInfoMap {
		nodeInfoList = append(nodeInfoList, ani)
	}

	return nodeInfoList
}

func (data *internalDeltaSnapshotData) addNodeInfo(nodeInfo schedulerinterface.NodeInfo) error {
	if _, found := data.getNodeInfo(nodeInfo.Node().Name); found {
		return fmt.Errorf("node %s already in snapshot", nodeInfo.Node().Name)
	}

	if _, found := data.deletedNodeInfos[nodeInfo.Node().Name]; found {
		delete(data.deletedNodeInfos, nodeInfo.Node().Name)
		data.modifiedNodeInfoMap[nodeInfo.Node().Name] = nodeInfo
	} else {
		data.addedNodeInfoMap[nodeInfo.Node().Name] = nodeInfo
	}

	if data.nodeInfoList != nil {
		data.nodeInfoList = append(data.nodeInfoList, nodeInfo)
	}

	if len(nodeInfo.GetPods()) > 0 {
		data.clearPodCaches()
	}

	return nil
}

func (data *internalDeltaSnapshotData) clearCaches() {
	data.nodeInfoList = nil
	data.clearPodCaches()
}

func (data *internalDeltaSnapshotData) clearPodCaches() {
	data.havePodsWithAffinity = nil
	data.havePodsWithRequiredAntiAffinity = nil
	// TODO: update the cache when adding/removing pods instead of invalidating the whole cache
	data.pvcNamespaceMap = nil
}

func (data *internalDeltaSnapshotData) removeNodeInfo(nodeName string) error {
	_, foundInDelta := data.addedNodeInfoMap[nodeName]
	if foundInDelta {
		// If node was added within this delta, delete this change.
		delete(data.addedNodeInfoMap, nodeName)
	}

	if _, modified := data.modifiedNodeInfoMap[nodeName]; modified {
		// If node was modified within this delta, delete this change.
		delete(data.modifiedNodeInfoMap, nodeName)
	}

	if _, deleted := data.deletedNodeInfos[nodeName]; deleted {
		// If node was deleted within this delta, fail with error.
		return errNodeNotFound
	}

	_, foundInBase := data.baseData.getNodeInfo(nodeName)
	if foundInBase {
		// If node was found in the underlying data, mark it as deleted in delta.
		data.deletedNodeInfos[nodeName] = true
	}

	if !foundInBase && !foundInDelta {
		// Node not found in the chain.
		return errNodeNotFound
	}

	// Maybe consider deleting from the lists instead. Maybe not.
	data.clearCaches()
	return nil
}

func (data *internalDeltaSnapshotData) nodeInfoToModify(nodeName string) (schedulerinterface.NodeInfo, bool) {
	dni, found := data.getNodeInfoLocal(nodeName)
	if !found {
		if _, found := data.deletedNodeInfos[nodeName]; found {
			return nil, false
		}
		bni, found := data.baseData.getNodeInfo(nodeName)
		if !found {
			return nil, false
		}
		dni = bni.Snapshot()
		data.modifiedNodeInfoMap[nodeName] = dni
		data.clearCaches()
	}
	return dni, true
}

func (data *internalDeltaSnapshotData) addPodInfo(podInfo schedulerinterface.PodInfo, nodeName string) error {
	ni, found := data.nodeInfoToModify(nodeName)
	if !found {
		return errNodeNotFound
	}

	ni.AddPodInfo(podInfo)

	// Maybe consider deleting from the list in the future. Maybe not.
	data.clearCaches()
	return nil
}

func (data *internalDeltaSnapshotData) removePod(namespace, name, nodeName string) error {
	// This always clones node info, even if the pod is actually missing.
	// Not sure if we mind, since removing non-existent pod
	// probably means things are very bad anyway.
	ni, found := data.nodeInfoToModify(nodeName)
	if !found {
		return errNodeNotFound
	}

	podFound := false
	logger := klog.Background()
	for _, podInfo := range ni.GetPods() {
		if podInfo.GetPod().Namespace == namespace && podInfo.GetPod().Name == name {
			if err := ni.RemovePod(logger, podInfo.GetPod()); err != nil {
				return fmt.Errorf("cannot remove pod; %v", err)
			}
			podFound = true
			break
		}
	}
	if !podFound {
		return fmt.Errorf("pod %s/%s not in snapshot", namespace, name)
	}

	// Maybe consider deleting from the list in the future. Maybe not.
	data.clearCaches()
	return nil
}

func (data *internalDeltaSnapshotData) isPVCUsedByPods(key string) bool {
	if data.pvcNamespaceMap != nil {
		return data.pvcNamespaceMap[key] > 0
	}
	nodeInfos := data.getNodeInfoList()
	pvcNamespaceMap := make(map[string]int)
	for _, v := range nodeInfos {
		for k, i := range v.GetPVCRefCounts() {
			pvcNamespaceMap[k] += i
		}
	}
	data.pvcNamespaceMap = pvcNamespaceMap
	return data.pvcNamespaceMap[key] > 0
}

func (data *internalDeltaSnapshotData) fork() *internalDeltaSnapshotData {
	forkedData := newInternalDeltaSnapshotData()
	forkedData.baseData = data
	return forkedData
}

func (data *internalDeltaSnapshotData) commit() (*internalDeltaSnapshotData, error) {
	if data.baseData == nil {
		// do nothing... as in basic snapshot.
		return data, nil
	}
	for node := range data.deletedNodeInfos {
		if err := data.baseData.removeNodeInfo(node); err != nil {
			return nil, err
		}
	}
	for _, node := range data.modifiedNodeInfoMap {
		if err := data.baseData.removeNodeInfo(node.Node().Name); err != nil {
			return nil, err
		}
		if err := data.baseData.addNodeInfo(node); err != nil {
			return nil, err
		}
	}
	for _, node := range data.addedNodeInfoMap {
		if err := data.baseData.addNodeInfo(node); err != nil {
			return nil, err
		}
	}

	return data.baseData, nil
}

// List returns list of all node infos.
func (snapshot *deltaSnapshotStoreNodeLister) List() ([]schedulerinterface.NodeInfo, error) {
	return snapshot.data.getNodeInfoList(), nil
}

// HavePodsWithAffinityList returns list of all node infos with pods that have affinity constrints.
func (snapshot *deltaSnapshotStoreNodeLister) HavePodsWithAffinityList() ([]schedulerinterface.NodeInfo, error) {
	data := snapshot.data
	if data.havePodsWithAffinity != nil {
		return data.havePodsWithAffinity, nil
	}

	nodeInfoList := snapshot.data.getNodeInfoList()
	havePodsWithAffinityList := make([]schedulerinterface.NodeInfo, 0, len(nodeInfoList))
	for _, node := range nodeInfoList {
		if len(node.GetPodsWithAffinity()) > 0 {
			havePodsWithAffinityList = append(havePodsWithAffinityList, node)
		}
	}
	data.havePodsWithAffinity = havePodsWithAffinityList
	return data.havePodsWithAffinity, nil
}

// HavePodsWithRequiredAntiAffinityList returns the list of NodeInfos of nodes with pods with required anti-affinity terms.
func (snapshot *deltaSnapshotStoreNodeLister) HavePodsWithRequiredAntiAffinityList() ([]schedulerinterface.NodeInfo, error) {
	data := snapshot.data
	if data.havePodsWithRequiredAntiAffinity != nil {
		return data.havePodsWithRequiredAntiAffinity, nil
	}

	nodeInfoList := snapshot.data.getNodeInfoList()
	havePodsWithRequiredAntiAffinityList := make([]schedulerinterface.NodeInfo, 0, len(nodeInfoList))
	for _, node := range nodeInfoList {
		if len(node.GetPodsWithRequiredAntiAffinity()) > 0 {
			havePodsWithRequiredAntiAffinityList = append(havePodsWithRequiredAntiAffinityList, node)
		}
	}
	data.havePodsWithRequiredAntiAffinity = havePodsWithRequiredAntiAffinityList
	return data.havePodsWithRequiredAntiAffinity, nil
}

// Get returns node info by node name.
func (snapshot *deltaSnapshotStoreNodeLister) Get(nodeName string) (schedulerinterface.NodeInfo, error) {
	return (*DeltaSnapshotStore)(snapshot).getNodeInfo(nodeName)
}

// IsPVCUsedByPods returns if PVC is used by pods
func (snapshot *deltaSnapshotStoreStorageLister) IsPVCUsedByPods(key string) bool {
	return (*DeltaSnapshotStore)(snapshot).IsPVCUsedByPods(key)
}

func (snapshot *DeltaSnapshotStore) getNodeInfo(nodeName string) (schedulerinterface.NodeInfo, error) {
	data := snapshot.data
	node, found := data.getNodeInfo(nodeName)
	if !found {
		return nil, errNodeNotFound
	}
	return node, nil
}

// NodeInfos returns node lister.
func (snapshot *DeltaSnapshotStore) NodeInfos() schedulerinterface.NodeInfoLister {
	return (*deltaSnapshotStoreNodeLister)(snapshot)
}

// StorageInfos returns storage lister
func (snapshot *DeltaSnapshotStore) StorageInfos() schedulerinterface.StorageInfoLister {
	return (*deltaSnapshotStoreStorageLister)(snapshot)
}

// NewDeltaSnapshotStore creates instances of DeltaSnapshotStore.
func NewDeltaSnapshotStore(parallelism int, pods []*v1.Pod, nodes []*v1.Node) *DeltaSnapshotStore {
	snapshot := &DeltaSnapshotStore{
		parallelism: parallelism,
	}
	snapshot.Clear()
	for _, node := range nodes {
		nodeInfo := framework.NewNodeInfo()
		nodeInfo.SetNode(node)
		if err := snapshot.StoreNodeInfo(nodeInfo); err != nil {
			klog.ErrorS(err, "Failed to store node info", "node", klog.KObj(node))
		}
	}
	for _, pod := range pods {
		podInfo, err := framework.NewPodInfo(pod)
		if err != nil {
			klog.ErrorS(err, "Failed to create pod info", "pod", klog.KObj(pod))
			continue
		}
		if err := snapshot.StorePodInfo(podInfo, pod.Spec.NodeName); err != nil {
			klog.ErrorS(err, "Failed to store pod info", "pod", klog.KObj(pod))
		}
	}
	return snapshot
}

// RemoveNodeInfo removes nodes (and pods scheduled to it) from the snapshot.
func (snapshot *DeltaSnapshotStore) RemoveNodeInfo(nodeName string) error {
	return snapshot.data.removeNodeInfo(nodeName)
}

// StoreNodeInfo adds the given *framework.NodeInfo to the snapshot without checking scheduler predicates.
func (snapshot *DeltaSnapshotStore) StoreNodeInfo(nodeInfo *framework.NodeInfo) error {
	return snapshot.data.addNodeInfo(nodeInfo)
}

// StorePodInfo adds pod to the snapshot and schedules it to given node.
func (snapshot *DeltaSnapshotStore) StorePodInfo(podInfo *framework.PodInfo, nodeName string) error {
	return snapshot.data.addPodInfo(podInfo, nodeName)
}

// RemovePodInfo removes pod from the snapshot.
func (snapshot *DeltaSnapshotStore) RemovePodInfo(namespace, podName, nodeName string) error {
	return snapshot.data.removePod(namespace, podName, nodeName)
}

// IsPVCUsedByPods returns if the pvc is used by any pod
func (snapshot *DeltaSnapshotStore) IsPVCUsedByPods(key string) bool {
	return snapshot.data.isPVCUsedByPods(key)
}

// Fork creates a fork of snapshot state. All modifications can later be reverted to moment of forking via Revert()
// Time: O(1)
func (snapshot *DeltaSnapshotStore) Fork() {
	snapshot.data = snapshot.data.fork()
}

// Revert reverts snapshot state to moment of forking.
// Time: O(1)
func (snapshot *DeltaSnapshotStore) Revert() {
	if snapshot.data.baseData != nil {
		snapshot.data = snapshot.data.baseData
	}
}

// Commit commits changes done after forking.
// Time: O(n), where n = size of delta (number of nodes added, modified or deleted since forking)
func (snapshot *DeltaSnapshotStore) Commit() error {
	newData, err := snapshot.data.commit()
	if err != nil {
		return err
	}
	snapshot.data = newData
	return nil
}

// Clear reset cluster snapshot to empty, unforked state
// Time: O(1)
func (snapshot *DeltaSnapshotStore) Clear() {
	snapshot.data = newInternalDeltaSnapshotData()
}

func (snapshot *DeltaSnapshotStore) PodGroupStates() fwk.PodGroupStateLister {
	return nil
}
