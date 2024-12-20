/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

var (
	cleanAssumedPeriod = 1 * time.Second
)

// New returns a Cache implementation.
// It automatically starts a go routine that manages expiration of assumed pods.
// "ttl" is how long the assumed pod will get expired.
// "ctx" is the context that would close the background goroutine.
func New(ctx context.Context, ttl time.Duration, apiDispatcher fwk.APIDispatcher) Cache {
	logger := klog.FromContext(ctx)
	cache := newCache(ctx, ttl, cleanAssumedPeriod, apiDispatcher)
	cache.run(logger)
	return cache
}

// nodeInfoListItem holds a NodeInfo pointer and acts as an item in a doubly
// linked list. When a NodeInfo is updated, it goes to the head of the list.
// The items closer to the head are the most recently updated items.
type nodeInfoListItem struct {
	info *framework.NodeInfo
	next *nodeInfoListItem
	prev *nodeInfoListItem
}

type cacheImpl struct {
	stop   <-chan struct{}
	ttl    time.Duration
	period time.Duration

	// This mutex guards all fields within this cache struct.
	mu sync.RWMutex
	// a set of assumed pod keys.
	// The key could further be used to get an entry in podStates.
	assumedPods sets.Set[string]
	// a map from pod key to podState.
	podStates map[string]*podState
	nodes     map[string]*nodeInfoListItem
	// headNode points to the most recently updated NodeInfo in "nodes". It is the
	// head of the linked list.
	headNode *nodeInfoListItem
	nodeTree *nodeTree
	// A map from image name to its ImageStateSummary.
	imageStates map[string]*fwk.ImageStateSummary

	// apiDispatcher is used for the methods that are expected to send API calls.
	// It's non-nil only if the SchedulerAsyncAPICalls feature gate is enabled.
	apiDispatcher fwk.APIDispatcher
}

type podState struct {
	pod *v1.Pod
	// Used by assumedPod to determinate expiration.
	// If deadline is nil, assumedPod will never expire.
	deadline *time.Time
	// Used to block cache from expiring assumedPod if binding still runs
	bindingFinished bool
}

func newCache(ctx context.Context, ttl, period time.Duration, apiDispatcher fwk.APIDispatcher) *cacheImpl {
	logger := klog.FromContext(ctx)
	return &cacheImpl{
		ttl:    ttl,
		period: period,
		stop:   ctx.Done(),

		nodes:         make(map[string]*nodeInfoListItem),
		nodeTree:      newNodeTree(logger, nil),
		assumedPods:   sets.New[string](),
		podStates:     make(map[string]*podState),
		imageStates:   make(map[string]*fwk.ImageStateSummary),
		apiDispatcher: apiDispatcher,
	}
}

// newNodeInfoListItem initializes a new nodeInfoListItem.
func newNodeInfoListItem(ni *framework.NodeInfo) *nodeInfoListItem {
	return &nodeInfoListItem{
		info: ni,
	}
}

// moveNodeInfoToHead moves a NodeInfo to the head of "cache.nodes" doubly
// linked list. The head is the most recently updated NodeInfo.
// We assume cache lock is already acquired.
func (cache *cacheImpl) moveNodeInfoToHead(logger klog.Logger, name string) {
	ni, ok := cache.nodes[name]
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "No node info with given name found in the cache", "node", klog.KRef("", name))
		return
	}
	// if the node info list item is already at the head, we are done.
	if ni == cache.headNode {
		return
	}

	if ni.prev != nil {
		ni.prev.next = ni.next
	}
	if ni.next != nil {
		ni.next.prev = ni.prev
	}
	if cache.headNode != nil {
		cache.headNode.prev = ni
	}
	ni.next = cache.headNode
	ni.prev = nil
	cache.headNode = ni
}

// removeNodeInfoFromList removes a NodeInfo from the "cache.nodes" doubly
// linked list.
// We assume cache lock is already acquired.
func (cache *cacheImpl) removeNodeInfoFromList(logger klog.Logger, name string) {
	ni, ok := cache.nodes[name]
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "No node info with given name found in the cache", "node", klog.KRef("", name))
		return
	}

	if ni.prev != nil {
		ni.prev.next = ni.next
	}
	if ni.next != nil {
		ni.next.prev = ni.prev
	}
	// if the removed item was at the head, we must update the head.
	if ni == cache.headNode {
		cache.headNode = ni.next
	}
	delete(cache.nodes, name)
}

// Dump produces a dump of the current scheduler cache. This is used for
// debugging purposes only and shouldn't be confused with UpdateSnapshot
// function.
// This method is expensive, and should be only used in non-critical path.
func (cache *cacheImpl) Dump() *Dump {
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	nodes := make(map[string]*framework.NodeInfo, len(cache.nodes))
	for k, v := range cache.nodes {
		nodes[k] = v.info.SnapshotConcrete()
	}

	return &Dump{
		Nodes:       nodes,
		AssumedPods: cache.assumedPods.Union(nil),
	}
}

// UpdateSnapshot takes a snapshot of cached NodeInfo map. This is called at
// beginning of every scheduling cycle.
// The snapshot only includes Nodes that are not deleted at the time this function is called.
// nodeInfo.Node() is guaranteed to be not nil for all the nodes in the snapshot.
// This function tracks generation number of NodeInfo and updates only the
// entries of an existing snapshot that have changed after the snapshot was taken.
func (cache *cacheImpl) UpdateSnapshot(logger klog.Logger, nodeSnapshot *Snapshot) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	// Get the last generation of the snapshot.
	snapshotGeneration := nodeSnapshot.generation

	// NodeInfoList and HavePodsWithAffinityNodeInfoList must be re-created if a node was added
	// or removed from the cache.
	updateAllLists := false
	// HavePodsWithAffinityNodeInfoList must be re-created if a node changed its
	// status from having pods with affinity to NOT having pods with affinity or the other
	// way around.
	updateNodesHavePodsWithAffinity := false
	// HavePodsWithRequiredAntiAffinityNodeInfoList must be re-created if a node changed its
	// status from having pods with required anti-affinity to NOT having pods with required
	// anti-affinity or the other way around.
	updateNodesHavePodsWithRequiredAntiAffinity := false
	// usedPVCSet must be re-created whenever the head node generation is greater than
	// last snapshot generation.
	updateUsedPVCSet := false

	// Start from the head of the NodeInfo doubly linked list and update snapshot
	// of NodeInfos updated after the last snapshot.
	for node := cache.headNode; node != nil; node = node.next {
		if node.info.Generation <= snapshotGeneration {
			// all the nodes are updated before the existing snapshot. We are done.
			break
		}
		if np := node.info.Node(); np != nil {
			existing, ok := nodeSnapshot.nodeInfoMap[np.Name]
			if !ok {
				updateAllLists = true
				existing = &framework.NodeInfo{}
				nodeSnapshot.nodeInfoMap[np.Name] = existing
			}
			clone := node.info.SnapshotConcrete()
			// We track nodes that have pods with affinity, here we check if this node changed its
			// status from having pods with affinity to NOT having pods with affinity or the other
			// way around.
			if (len(existing.PodsWithAffinity) > 0) != (len(clone.PodsWithAffinity) > 0) {
				updateNodesHavePodsWithAffinity = true
			}
			if (len(existing.PodsWithRequiredAntiAffinity) > 0) != (len(clone.PodsWithRequiredAntiAffinity) > 0) {
				updateNodesHavePodsWithRequiredAntiAffinity = true
			}
			if !updateUsedPVCSet {
				if len(existing.PVCRefCounts) != len(clone.PVCRefCounts) {
					updateUsedPVCSet = true
				} else {
					for pvcKey := range clone.PVCRefCounts {
						if _, found := existing.PVCRefCounts[pvcKey]; !found {
							updateUsedPVCSet = true
							break
						}
					}
				}
			}
			// We need to preserve the original pointer of the NodeInfo struct since it
			// is used in the NodeInfoList, which we may not update.
			*existing = *clone
		}
	}
	// Update the snapshot generation with the latest NodeInfo generation.
	if cache.headNode != nil {
		nodeSnapshot.generation = cache.headNode.info.Generation
	}

	// Comparing to pods in nodeTree.
	// Deleted nodes get removed from the tree, but they might remain in the nodes map
	// if they still have non-deleted Pods.
	if len(nodeSnapshot.nodeInfoMap) > cache.nodeTree.numNodes {
		cache.removeDeletedNodesFromSnapshot(nodeSnapshot)
		updateAllLists = true
	}

	if updateAllLists || updateNodesHavePodsWithAffinity || updateNodesHavePodsWithRequiredAntiAffinity || updateUsedPVCSet {
		cache.updateNodeInfoSnapshotList(logger, nodeSnapshot, updateAllLists)
	}

	if len(nodeSnapshot.nodeInfoList) != cache.nodeTree.numNodes {
		errMsg := fmt.Sprintf("snapshot state is not consistent, length of NodeInfoList=%v not equal to length of nodes in tree=%v "+
			", length of NodeInfoMap=%v, length of nodes in cache=%v"+
			", trying to recover",
			len(nodeSnapshot.nodeInfoList), cache.nodeTree.numNodes,
			len(nodeSnapshot.nodeInfoMap), len(cache.nodes))
		logger.Error(nil, errMsg)
		// We will try to recover by re-creating the lists for the next scheduling cycle, but still return an
		// error to surface the problem, the error will likely cause a failure to the current scheduling cycle.
		cache.updateNodeInfoSnapshotList(logger, nodeSnapshot, true)
		return errors.New(errMsg)
	}

	return nil
}

func (cache *cacheImpl) updateNodeInfoSnapshotList(logger klog.Logger, snapshot *Snapshot, updateAll bool) {
	snapshot.havePodsWithAffinityNodeInfoList = make([]fwk.NodeInfo, 0, cache.nodeTree.numNodes)
	snapshot.havePodsWithRequiredAntiAffinityNodeInfoList = make([]fwk.NodeInfo, 0, cache.nodeTree.numNodes)
	snapshot.usedPVCSet = sets.New[string]()
	if updateAll {
		// Take a snapshot of the nodes order in the tree
		snapshot.nodeInfoList = make([]fwk.NodeInfo, 0, cache.nodeTree.numNodes)
		nodesList, err := cache.nodeTree.list()
		if err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Error occurred while retrieving the list of names of the nodes from node tree")
		}
		for _, nodeName := range nodesList {
			if nodeInfo := snapshot.nodeInfoMap[nodeName]; nodeInfo != nil {
				snapshot.nodeInfoList = append(snapshot.nodeInfoList, nodeInfo)
				if len(nodeInfo.PodsWithAffinity) > 0 {
					snapshot.havePodsWithAffinityNodeInfoList = append(snapshot.havePodsWithAffinityNodeInfoList, nodeInfo)
				}
				if len(nodeInfo.PodsWithRequiredAntiAffinity) > 0 {
					snapshot.havePodsWithRequiredAntiAffinityNodeInfoList = append(snapshot.havePodsWithRequiredAntiAffinityNodeInfoList, nodeInfo)
				}
				for key := range nodeInfo.PVCRefCounts {
					snapshot.usedPVCSet.Insert(key)
				}
			} else {
				utilruntime.HandleErrorWithLogger(logger, nil, "Node exists in nodeTree but not in NodeInfoMap, this should not happen", "node", klog.KRef("", nodeName))
			}
		}
	} else {
		for _, nodeInfo := range snapshot.nodeInfoList {
			if len(nodeInfo.GetPodsWithAffinity()) > 0 {
				snapshot.havePodsWithAffinityNodeInfoList = append(snapshot.havePodsWithAffinityNodeInfoList, nodeInfo)
			}
			if len(nodeInfo.GetPodsWithRequiredAntiAffinity()) > 0 {
				snapshot.havePodsWithRequiredAntiAffinityNodeInfoList = append(snapshot.havePodsWithRequiredAntiAffinityNodeInfoList, nodeInfo)
			}
			for key := range nodeInfo.GetPVCRefCounts() {
				snapshot.usedPVCSet.Insert(key)
			}
		}
	}
}

// If certain nodes were deleted after the last snapshot was taken, we should remove them from the snapshot.
func (cache *cacheImpl) removeDeletedNodesFromSnapshot(snapshot *Snapshot) {
	toDelete := len(snapshot.nodeInfoMap) - cache.nodeTree.numNodes
	for name := range snapshot.nodeInfoMap {
		if toDelete <= 0 {
			break
		}
		if n, ok := cache.nodes[name]; !ok || n.info.Node() == nil {
			delete(snapshot.nodeInfoMap, name)
			toDelete--
		}
	}
}

// NodeCount returns the number of nodes in the cache.
// DO NOT use outside of tests.
func (cache *cacheImpl) NodeCount() int {
	cache.mu.RLock()
	defer cache.mu.RUnlock()
	return len(cache.nodes)
}

// PodCount returns the number of pods in the cache (including those from deleted nodes).
// DO NOT use outside of tests.
func (cache *cacheImpl) PodCount() (int, error) {
	cache.mu.RLock()
	defer cache.mu.RUnlock()
	// podFilter is expected to return true for most or all of the pods. We
	// can avoid expensive array growth without wasting too much memory by
	// pre-allocating capacity.
	count := 0
	for _, n := range cache.nodes {
		count += len(n.info.Pods)
	}
	return count, nil
}

func (cache *cacheImpl) AssumePod(logger klog.Logger, pod *v1.Pod) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()
	if _, ok := cache.podStates[key]; ok {
		return fmt.Errorf("pod %v(%v) is in the cache, so can't be assumed", key, klog.KObj(pod))
	}

	return cache.addPod(logger, pod, true)
}

func (cache *cacheImpl) FinishBinding(logger klog.Logger, pod *v1.Pod) error {
	return cache.finishBinding(logger, pod, time.Now())
}

// finishBinding exists to make tests deterministic by injecting now as an argument
func (cache *cacheImpl) finishBinding(logger klog.Logger, pod *v1.Pod, now time.Time) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}

	cache.mu.RLock()
	defer cache.mu.RUnlock()

	logger.V(5).Info("Finished binding for pod, can be expired", "podKey", key, "pod", klog.KObj(pod))
	currState, ok := cache.podStates[key]
	if ok && cache.assumedPods.Has(key) {
		if cache.ttl == time.Duration(0) {
			currState.deadline = nil
		} else {
			dl := now.Add(cache.ttl)
			currState.deadline = &dl
		}
		currState.bindingFinished = true
	}
	return nil
}

func (cache *cacheImpl) ForgetPod(logger klog.Logger, pod *v1.Pod) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	currState, ok := cache.podStates[key]
	if ok && currState.pod.Spec.NodeName != pod.Spec.NodeName {
		return fmt.Errorf("pod %v(%v) was assumed on %v but assigned to %v", key, klog.KObj(pod), pod.Spec.NodeName, currState.pod.Spec.NodeName)
	}

	// Only assumed pod can be forgotten.
	if ok && cache.assumedPods.Has(key) {
		return cache.removePod(logger, pod)
	}
	return fmt.Errorf("pod %v(%v) wasn't assumed so cannot be forgotten", key, klog.KObj(pod))
}

// Assumes that lock is already acquired.
func (cache *cacheImpl) addPod(logger klog.Logger, pod *v1.Pod, assumePod bool) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}
	n, ok := cache.nodes[pod.Spec.NodeName]
	if !ok {
		n = newNodeInfoListItem(framework.NewNodeInfo())
		cache.nodes[pod.Spec.NodeName] = n
	}
	n.info.AddPod(pod)
	cache.moveNodeInfoToHead(logger, pod.Spec.NodeName)
	ps := &podState{
		pod: pod,
	}
	cache.podStates[key] = ps
	if assumePod {
		cache.assumedPods.Insert(key)
	}
	return nil
}

// Assumes that lock is already acquired.
func (cache *cacheImpl) updatePod(logger klog.Logger, oldPod, newPod *v1.Pod) error {
	if err := cache.removePod(logger, oldPod); err != nil {
		return err
	}
	return cache.addPod(logger, newPod, false)
}

// Assumes that lock is already acquired.
// Removes a pod from the cached node info. If the node information was already
// removed and there are no more pods left in the node, cleans up the node from
// the cache.
func (cache *cacheImpl) removePod(logger klog.Logger, pod *v1.Pod) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}

	n, ok := cache.nodes[pod.Spec.NodeName]
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "Node not found when trying to remove pod", "node", klog.KRef("", pod.Spec.NodeName), "podKey", key, "pod", klog.KObj(pod))
	} else {
		if err := n.info.RemovePod(logger, pod); err != nil {
			return err
		}
		if len(n.info.Pods) == 0 && n.info.Node() == nil {
			cache.removeNodeInfoFromList(logger, pod.Spec.NodeName)
		} else {
			cache.moveNodeInfoToHead(logger, pod.Spec.NodeName)
		}
	}

	delete(cache.podStates, key)
	delete(cache.assumedPods, key)
	return nil
}

func (cache *cacheImpl) AddPod(logger klog.Logger, pod *v1.Pod) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	currState, ok := cache.podStates[key]
	switch {
	case ok && cache.assumedPods.Has(key):
		// When assuming, we've already added the Pod to cache,
		// Just update here to make sure the Pod's status is up-to-date.
		if err = cache.updatePod(logger, currState.pod, pod); err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Error occurred while updating pod")
		}
		if currState.pod.Spec.NodeName != pod.Spec.NodeName {
			// The pod was added to a different node than it was assumed to.
			logger.Info("Pod was added to a different node than it was assumed", "podKey", key, "pod", klog.KObj(pod), "assumedNode", klog.KRef("", pod.Spec.NodeName), "currentNode", klog.KRef("", currState.pod.Spec.NodeName))
			return nil
		}
	case !ok:
		// Pod was expired. We should add it back.
		if err = cache.addPod(logger, pod, false); err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Error occurred while adding pod")
		}
	default:
		return fmt.Errorf("pod %v(%v) was already in added state", key, klog.KObj(pod))
	}
	return nil
}

func (cache *cacheImpl) UpdatePod(logger klog.Logger, oldPod, newPod *v1.Pod) error {
	key, err := framework.GetPodKey(oldPod)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	currState, ok := cache.podStates[key]
	if !ok {
		return fmt.Errorf("pod %v(%v) is not added to scheduler cache, so cannot be updated", key, klog.KObj(oldPod))
	}

	// An assumed pod won't have Update/Remove event. It needs to have Add event
	// before Update event, in which case the state would change from Assumed to Added.
	if cache.assumedPods.Has(key) {
		return fmt.Errorf("assumed pod %v(%v) should not be updated", key, klog.KObj(oldPod))
	}

	if currState.pod.Spec.NodeName != newPod.Spec.NodeName {
		utilruntime.HandleErrorWithLogger(logger, nil, "Pod updated on a different node than previously added to. Scheduler cache is corrupted and can badly affect scheduling decisions", "podKey", key, "pod", klog.KObj(oldPod))
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	return cache.updatePod(logger, oldPod, newPod)
}

func (cache *cacheImpl) RemovePod(logger klog.Logger, pod *v1.Pod) error {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	currState, ok := cache.podStates[key]
	if !ok {
		return fmt.Errorf("pod %v(%v) is not found in scheduler cache, so cannot be removed from it", key, klog.KObj(pod))
	}
	if currState.pod.Spec.NodeName != pod.Spec.NodeName {
		utilruntime.HandleErrorWithLogger(logger, nil, "Pod was added to a different node than it was assumed", "podKey", key, "pod", klog.KObj(pod), "assumedNode", klog.KRef("", pod.Spec.NodeName), "currentNode", klog.KRef("", currState.pod.Spec.NodeName))
		if pod.Spec.NodeName != "" {
			// An empty NodeName is possible when the scheduler misses a Delete
			// event and it gets the last known state from the informer cache.
			utilruntime.HandleErrorWithLogger(logger, nil, "Scheduler cache is corrupted and can badly affect scheduling decisions")
			klog.FlushAndExit(klog.ExitFlushTimeout, 1)
		}
	}
	return cache.removePod(logger, currState.pod)
}

func (cache *cacheImpl) IsAssumedPod(pod *v1.Pod) (bool, error) {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return false, err
	}

	cache.mu.RLock()
	defer cache.mu.RUnlock()

	return cache.assumedPods.Has(key), nil
}

// GetPod might return a pod for which its node has already been deleted from
// the main cache. This is useful to properly process pod update events.
func (cache *cacheImpl) GetPod(pod *v1.Pod) (*v1.Pod, error) {
	key, err := framework.GetPodKey(pod)
	if err != nil {
		return nil, err
	}

	cache.mu.RLock()
	defer cache.mu.RUnlock()

	podState, ok := cache.podStates[key]
	if !ok {
		return nil, fmt.Errorf("pod %v(%v) does not exist in scheduler cache", key, klog.KObj(pod))
	}

	return podState.pod, nil
}

func (cache *cacheImpl) AddNode(logger klog.Logger, node *v1.Node) *framework.NodeInfo {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	n, ok := cache.nodes[node.Name]
	if !ok {
		n = newNodeInfoListItem(framework.NewNodeInfo())
		cache.nodes[node.Name] = n
	} else {
		cache.removeNodeImageStates(n.info.Node())
	}
	cache.moveNodeInfoToHead(logger, node.Name)

	cache.nodeTree.addNode(logger, node)
	cache.addNodeImageStates(node, n.info)
	n.info.SetNode(node)
	return n.info.SnapshotConcrete()
}

func (cache *cacheImpl) UpdateNode(logger klog.Logger, oldNode, newNode *v1.Node) *framework.NodeInfo {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	n, ok := cache.nodes[newNode.Name]
	if !ok {
		n = newNodeInfoListItem(framework.NewNodeInfo())
		cache.nodes[newNode.Name] = n
		cache.nodeTree.addNode(logger, newNode)
	} else {
		cache.removeNodeImageStates(n.info.Node())
	}
	cache.moveNodeInfoToHead(logger, newNode.Name)

	cache.nodeTree.updateNode(logger, oldNode, newNode)
	cache.addNodeImageStates(newNode, n.info)
	n.info.SetNode(newNode)
	return n.info.SnapshotConcrete()
}

// RemoveNode removes a node from the cache's tree.
// The node might still have pods because their deletion events didn't arrive
// yet. Those pods are considered removed from the cache, being the node tree
// the source of truth.
// However, we keep a ghost node with the list of pods until all pod deletion
// events have arrived. A ghost node is skipped from snapshots.
func (cache *cacheImpl) RemoveNode(logger klog.Logger, node *v1.Node) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	n, ok := cache.nodes[node.Name]
	if !ok {
		return fmt.Errorf("node %v is not found", node.Name)
	}
	n.info.RemoveNode()
	// We remove NodeInfo for this node only if there aren't any pods on this node.
	// We can't do it unconditionally, because notifications about pods are delivered
	// in a different watch, and thus can potentially be observed later, even though
	// they happened before node removal.
	if len(n.info.Pods) == 0 {
		cache.removeNodeInfoFromList(logger, node.Name)
	} else {
		cache.moveNodeInfoToHead(logger, node.Name)
	}
	if err := cache.nodeTree.removeNode(logger, node); err != nil {
		return err
	}
	cache.removeNodeImageStates(node)
	return nil
}

// addNodeImageStates adds states of the images on given node to the given nodeInfo and update the imageStates in
// scheduler cache. This function assumes the lock to scheduler cache has been acquired.
func (cache *cacheImpl) addNodeImageStates(node *v1.Node, nodeInfo *framework.NodeInfo) {
	newSum := make(map[string]*fwk.ImageStateSummary)

	for _, image := range node.Status.Images {
		for _, name := range image.Names {
			// update the entry in imageStates
			state, ok := cache.imageStates[name]
			if !ok {
				state = &fwk.ImageStateSummary{
					Size:  image.SizeBytes,
					Nodes: sets.New(node.Name),
				}
				cache.imageStates[name] = state
			} else {
				state.Nodes.Insert(node.Name)
			}
			// create the ImageStateSummary for this image
			if _, ok := newSum[name]; !ok {
				newSum[name] = state
			}
		}
	}
	nodeInfo.ImageStates = newSum
}

// removeNodeImageStates removes the given node record from image entries having the node
// in imageStates cache. After the removal, if any image becomes free, i.e., the image
// is no longer available on any node, the image entry will be removed from imageStates.
func (cache *cacheImpl) removeNodeImageStates(node *v1.Node) {
	if node == nil {
		return
	}

	for _, image := range node.Status.Images {
		for _, name := range image.Names {
			state, ok := cache.imageStates[name]
			if ok {
				state.Nodes.Delete(node.Name)
				if state.Nodes.Len() == 0 {
					// Remove the unused image to make sure the length of
					// imageStates represents the total number of different
					// images on all nodes
					delete(cache.imageStates, name)
				}
			}
		}
	}
}

func (cache *cacheImpl) run(logger klog.Logger) {
	go wait.Until(func() {
		cache.cleanupAssumedPods(logger, time.Now())
	}, cache.period, cache.stop)
}

// cleanupAssumedPods exists for making test deterministic by taking time as input argument.
// It also reports metrics on the cache size for nodes, pods, and assumed pods.
func (cache *cacheImpl) cleanupAssumedPods(logger klog.Logger, now time.Time) {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	defer cache.updateMetrics()

	// The size of assumedPods should be small
	for key := range cache.assumedPods {
		ps, ok := cache.podStates[key]
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Key found in assumed set but not in podStates, potentially a logical error")
			klog.FlushAndExit(klog.ExitFlushTimeout, 1)
		}
		if !ps.bindingFinished {
			logger.V(5).Info("Could not expire cache for pod as binding is still in progress", "podKey", key, "pod", klog.KObj(ps.pod))
			continue
		}
		if cache.ttl != 0 && now.After(*ps.deadline) {
			logger.Info("Pod expired", "podKey", key, "pod", klog.KObj(ps.pod))
			if err := cache.removePod(logger, ps.pod); err != nil {
				utilruntime.HandleErrorWithLogger(logger, err, "ExpirePod failed", "podKey", key, "pod", klog.KObj(ps.pod))
			}
		}
	}
}

// updateMetrics updates cache size metric values for pods, assumed pods, and nodes
func (cache *cacheImpl) updateMetrics() {
	metrics.CacheSize.WithLabelValues("assumed_pods").Set(float64(len(cache.assumedPods)))
	metrics.CacheSize.WithLabelValues("pods").Set(float64(len(cache.podStates)))
	metrics.CacheSize.WithLabelValues("nodes").Set(float64(len(cache.nodes)))
}

// BindPod handles the pod binding by adding a bind API call to the dispatcher.
// This method should be used only if the SchedulerAsyncAPICalls feature gate is enabled.
func (cache *cacheImpl) BindPod(binding *v1.Binding) (<-chan error, error) {
	// Don't store anything in the cache, as the pod is already assumed, and in case of a binding failure, it will be forgotten.
	onFinish := make(chan error, 1)
	err := cache.apiDispatcher.Add(apicalls.Implementations.PodBinding(binding), fwk.APICallOptions{
		OnFinish: onFinish,
	})
	if fwk.IsUnexpectedError(err) {
		return onFinish, err
	}
	return onFinish, nil
}
