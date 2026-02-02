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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

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
	// assumedPods maps a pod key to an assumed pod object during a single (workload) scheduling cycle.
	// This map should be emptied before the next cycle starts.
	assumedPods map[string]*v1.Pod
}

var _ fwk.SharedLister = &Snapshot{}

// NewEmptySnapshot initializes a Snapshot struct and returns it.
func NewEmptySnapshot() *Snapshot {
	return &Snapshot{
		nodeInfoMap: make(map[string]*framework.NodeInfo),
		usedPVCSet:  sets.New[string](),
		assumedPods: make(map[string]*v1.Pod),
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

	return s
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

// NumNodes returns the number of nodes in the snapshot.
func (s *Snapshot) NumNodes() int {
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
		return fmt.Errorf("assumed node %q not found in the snapshot", pod.Spec.NodeName)
	}
	// Calling AddPodInfo increases the Generation number of the nodeInfo.
	// Since this operation only affects the snapshot,
	// we should keep the old number to remain consistent with the cached value.
	oldGeneration := nodeInfo.Generation
	nodeInfo.AddPodInfo(podInfo)
	nodeInfo.Generation = oldGeneration
	s.assumedPods[key] = pod
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
		nodeInfo.RemovePod(logger, pod)
		nodeInfo.Generation = oldGeneration
	}
	return nil
}

// forgetAllAssumedPods forgets all assumed pods from the snapshot.
// This function is not thread safe, so it should be executed when no other routines can write/read from the snapshot.
func (s *Snapshot) forgetAllAssumedPods(logger klog.Logger) {
	for _, pod := range s.assumedPods {
		s.ForgetPod(logger, pod)
	}
}
