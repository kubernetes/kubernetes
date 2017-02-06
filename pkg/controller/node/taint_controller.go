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

package node

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"

	"github.com/golang/glog"
)

const (
	nodeUpdateChannelSize = 10000
	podUpdateChannelSize  = 50000
	retries               = 4
)

func computeTaintDifference(left []v1.Taint, right []v1.Taint) []v1.Taint {
	result := []v1.Taint{}
	for i := range left {
		found := false
		for j := range right {
			if left[i] == right[j] {
				found = true
				break
			}
		}
		if !found {
			result = append(result, left[i])
		}
	}
	return result
}

// copy of 'computeTaintDifference' - long live lack of generics...
func computeTolerationDifference(left []v1.Toleration, right []v1.Toleration) []v1.Toleration {
	result := []v1.Toleration{}
	for i := range left {
		found := false
		for j := range right {
			if left[i] == right[j] {
				found = true
				break
			}
		}
		if !found {
			result = append(result, left[i])
		}
	}
	return result
}

type nodeUpdateItem struct {
	oldNode   *v1.Node
	newNode   *v1.Node
	oldTaints []v1.Taint
	newTaints []v1.Taint
}

type podUpdateItem struct {
	oldPod         *v1.Pod
	newPod         *v1.Pod
	oldTolerations []v1.Toleration
	newTolerations []v1.Toleration
}

// NoExecuteTaint manager listens to Taint/Toleration changes and is resposible for removing Pods
// from Nodes tainted with NoExecute Taints.
type NoExecuteTaintManager struct {
	client             clientset.Interface
	taintEvictionQueue *TimedWorkerQueue
	// keeps a map from nodeName to all noExecute taints on that Node
	taintedNodes     map[string][]v1.Taint
	taintedNodesLock sync.Mutex

	nodeUpdateChannel chan nodeUpdateItem
	podUpdateChannel  chan podUpdateItem
}

func deletePodHandler(c clientset.Interface) func(args *WorkArgs) {
	return func(args *WorkArgs) {
		ns := args.NamespacedName.Namespace
		name := args.NamespacedName.Name
		glog.V(0).Infof("NoExecuteTaintManager is deleting Pod: %v", args.NamespacedName.String())
		c.Core().Pods(ns).Delete(name, &metav1.DeleteOptions{})
	}
}

func getNonExecuteTaints(taints []v1.Taint) []v1.Taint {
	result := []v1.Taint{}
	for i := range taints {
		if taints[i].Effect == v1.TaintEffectNoExecute {
			result = append(result, taints[i])
		}
	}
	return result
}

// Returns true and list of Tolerations matching all Taints if all are tolerated, or false otherwise.
func getMatchingTolerations(taints []v1.Taint, tolerations []v1.Toleration) (bool, []v1.Toleration) {
	if len(tolerations) == 0 && len(taints) > 0 {
		return false, []v1.Toleration{}
	}
	result := []v1.Toleration{}
	for i := range taints {
		tolerated := false
		for j := range tolerations {
			if tolerations[j].ToleratesTaint(&taints[i]) {
				result = append(result, tolerations[j])
				tolerated = true
				break
			}
		}
		if !tolerated {
			return false, []v1.Toleration{}
		}
	}
	return true, result
}

func getPodsAssignedToNode(c clientset.Interface, nodeName string) ([]v1.Pod, error) {
	selector := fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
	pods, err := c.Core().Pods(v1.NamespaceAll).List(metav1.ListOptions{
		FieldSelector: selector.String(),
		LabelSelector: labels.Everything().String(),
	})
	for i := 0; i < retries && err != nil; i++ {
		pods, err = c.Core().Pods(v1.NamespaceAll).List(metav1.ListOptions{
			FieldSelector: selector.String(),
			LabelSelector: labels.Everything().String(),
		})
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return []v1.Pod{}, fmt.Errorf("Failed to get Pods assigned to node %v. Skipping update.", nodeName)
	}
	return pods.Items, nil
}

// Returns minimal toleration time from the given slice, or -1 if it's infinite.
func getMinTolerationTime(tolerations []v1.Toleration) time.Duration {
	minTolerationTime := int64(-1)
	for i := range tolerations {
		if tolerations[i].TolerationSeconds != nil {
			if minTolerationTime < 0 {
				minTolerationTime = *(tolerations[i].TolerationSeconds)
			} else {
				tolerationSeconds := *(tolerations[i].TolerationSeconds)
				if tolerationSeconds < minTolerationTime {
					if tolerationSeconds < 0 {
						minTolerationTime = 0
					} else {
						minTolerationTime = tolerationSeconds
					}
				}
			}
		}
	}
	return time.Duration(minTolerationTime) * time.Second
}

// NewNoExecuteTaintManager creates a new NoExecuteTaintManager that will use passed clientset to
// communicate with the API server.
func NewNoExecuteTaintManager(c clientset.Interface) *NoExecuteTaintManager {
	return &NoExecuteTaintManager{
		client:             c,
		taintEvictionQueue: CreateWorkerQueue(deletePodHandler(c)),
		taintedNodes:       make(map[string][]v1.Taint),
		nodeUpdateChannel:  make(chan nodeUpdateItem, nodeUpdateChannelSize),
		podUpdateChannel:   make(chan podUpdateItem, podUpdateChannelSize),
	}
}

// Run starts NoExecuteTaintManager which will run in loop until `stopCh` is closed.
func (tc *NoExecuteTaintManager) Run(stopCh <-chan struct{}) {
	// When processing events we want to prioritize Node updates over Pod updates,
	// as NodeUpdates that interest NoExecuteTaintManager should be handled as soon as possible -
	// we don't want user (or system) to wait until PodUpdate queue is drained before it can
	// start evicting Pods from tainted Nodes.
	for {
		select {
		case <-stopCh:
			break
		case nodeUpdate := <-tc.nodeUpdateChannel:
			tc.handleNodeUpdate(nodeUpdate)
		case podUpdate := <-tc.podUpdateChannel:
			// If we found a Pod update we need to empty Node queue first.
		priority:
			for {
				select {
				case nodeUpdate := <-tc.nodeUpdateChannel:
					tc.handleNodeUpdate(nodeUpdate)
				default:
					break priority
				}
			}
			// After Node queue is emptied we process podUpdate.
			tc.handlePodUpdate(podUpdate)
		}
	}
}

// PodUpdated is used to notify NoExecuteTaintManager about Pod changes.
func (tc *NoExecuteTaintManager) PodUpdated(oldPod *v1.Pod, newPod *v1.Pod) {
	var err error
	oldTolerations := []v1.Toleration{}
	if oldPod != nil {
		oldTolerations, err = v1.GetPodTolerations(oldPod)
		if err != nil {
			glog.Errorf("Failed to get Tolerations from the old Pod: %v", err)
			return
		}
	}
	newTolerations := []v1.Toleration{}
	if newPod != nil {
		newTolerations, err = v1.GetPodTolerations(newPod)
		if err != nil {
			glog.Errorf("Failed to get Tolerations from the new Pod: %v", err)
			return
		}
	}

	if oldPod != nil && newPod != nil && api.Semantic.DeepEqual(oldTolerations, newTolerations) && oldPod.Spec.NodeName == newPod.Spec.NodeName {
		return
	}
	select {
	case tc.podUpdateChannel <- podUpdateItem{
		oldPod:         oldPod,
		newPod:         newPod,
		oldTolerations: oldTolerations,
		newTolerations: newTolerations,
	}:
	default:
		// If our update queue is full it means that something's blocked, and we
		// should try restarting controller.
		panic(fmt.Errorf("NoExecuteTaintManager Pod update queue is full"))
	}
}

// NodeUpdated is used to notify NoExecuteTaintManager about Node changes.
func (tc *NoExecuteTaintManager) NodeUpdated(oldNode *v1.Node, newNode *v1.Node) {
	var err error
	oldTaints := []v1.Taint{}
	if oldNode != nil {
		oldTaints, err = v1.GetNodeTaints(oldNode)
		if err != nil {
			glog.Errorf("Failed to get Taints from the old Node: %v", err)
			return
		}
	}
	oldTaints = getNonExecuteTaints(oldTaints)

	newTaints := []v1.Taint{}
	if newNode != nil {
		newTaints, err = v1.GetNodeTaints(newNode)
		if err != nil {
			glog.Errorf("Failed to get Taints from the new Node: %v", err)
			return
		}
	}
	newTaints = getNonExecuteTaints(newTaints)

	if oldNode != nil && newNode != nil && api.Semantic.DeepEqual(oldTaints, newTaints) {
		return
	}
	select {
	case tc.nodeUpdateChannel <- nodeUpdateItem{
		oldNode:   oldNode,
		newNode:   newNode,
		oldTaints: oldTaints,
		newTaints: newTaints,
	}:
	default:
		// If our update queue is full it means that something's blocked, and we
		// should try restarting controller.
		panic(fmt.Errorf("NoExecuteTaintManager Node update queue is full"))
	}
}

func (tc *NoExecuteTaintManager) processPodOnNode(
	podNamespacedName types.NamespacedName,
	nodeName string,
	tolerations []v1.Toleration,
	taints []v1.Taint,
	now time.Time,
) {
	allTolerated, usedTolerations := getMatchingTolerations(taints, tolerations)
	if !allTolerated {
		glog.V(2).Infof("Not all taints are tolerated after upgrade for Pod %v on %v", podNamespacedName.String(), nodeName)
		tc.taintEvictionQueue.CancelWork(podNamespacedName.String())
		tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), time.Now(), time.Now())
		return
	}
	minTolerationTime := getMinTolerationTime(usedTolerations)
	// getMinTolerationTime returns negative value to denote infinite toleration.
	if minTolerationTime < 0 {
		glog.V(4).Infof("New tolerations for %v tolerate forever. Scheduled deletion won't be cancelled if already scheduled.", podNamespacedName.String())
		return
	}

	startTime := now
	triggerTime := startTime.Add(minTolerationTime)
	scheduledEviction := tc.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String())
	if scheduledEviction != nil {
		startTime = scheduledEviction.CreatedAt
		if startTime.Add(minTolerationTime).Before(triggerTime) {
			return
		} else {
			tc.taintEvictionQueue.CancelWork(podNamespacedName.String())
		}
	}
	tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), startTime, triggerTime)
}

func (tc *NoExecuteTaintManager) handlePodUpdate(podUpdate podUpdateItem) {
	// Delete
	if podUpdate.newPod == nil {
		pod := podUpdate.oldPod
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		glog.V(4).Infof("Noticed pod deletion: %v", podNamespacedName.String())
		tc.taintEvictionQueue.CancelWork(podNamespacedName.String())
		return
	}
	// Create or Update
	pod := podUpdate.newPod
	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	glog.V(4).Infof("Noticed pod update: %v", podNamespacedName.String())
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		return
	}
	taints, ok := func() ([]v1.Taint, bool) {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		taints, ok := tc.taintedNodes[nodeName]
		return taints, ok
	}()
	if !ok {
		return
	}
	tc.processPodOnNode(podNamespacedName, nodeName, podUpdate.newTolerations, taints, time.Now())
}

func (tc *NoExecuteTaintManager) handleNodeUpdate(nodeUpdate nodeUpdateItem) {
	// Delete
	if nodeUpdate.newNode == nil {
		node := nodeUpdate.oldNode
		glog.V(4).Infof("Noticed node deletion: %v", node.Name)
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		delete(tc.taintedNodes, node.Name)
		return
	}
	// Create or Update
	glog.V(4).Infof("Noticed node update: %v", nodeUpdate)
	node := nodeUpdate.newNode
	taints := nodeUpdate.newTaints
	func() {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		tc.taintedNodes[node.Name] = taints
	}()
	pods, err := getPodsAssignedToNode(tc.client, node.Name)
	if err != nil {
		glog.Errorf(err.Error())
		return
	}
	if len(pods) == 0 {
		return
	}
	if len(taints) == 0 {
		glog.V(4).Infof("All taints were removed from the Node. Cancelling all evictions...")
		for i := range pods {
			tc.taintEvictionQueue.CancelWork(types.NamespacedName{Namespace: pods[i].Namespace, Name: pods[i].Name}.String())
		}
	}

	now := time.Now()
	for i := range pods {
		pod := &pods[i]
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		tolerations, err := v1.GetPodTolerations(pod)
		if err != nil {
			glog.Errorf("Failed to get Tolerations from Pod %v: %v", podNamespacedName.String(), err)
			continue
		}
		tc.processPodOnNode(podNamespacedName, node.Name, tolerations, taints, now)
	}
}
