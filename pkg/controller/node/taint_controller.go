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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"

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

type TaintController struct {
	client             clientset.Interface
	taintEvictionQueue *TimedWorkerQueue
	// keeps a map from nodeName to all noExecute taints on that Node
	taintedNodes map[string][]v1.Taint

	nodeUpdateChannel chan nodeUpdateItem
	podUpdateChannel  chan podUpdateItem
}

func deletePodHandler(c clientset.Interface) func(args *WorkArgs) {
	return func(args *WorkArgs) {
		ns := args.NamespacedName.Namespace
		name := args.NamespacedName.Name
		glog.V(0).Infof("TaintController is deleting Pod: %v", args.NamespacedName.String())
		c.Core().Pods(ns).Delete(name, &v1.DeleteOptions{})
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
	result := []v1.Toleration{}
	for i := range taints {
		tolerated := false
		for j := range tolerations {
			if tolerations[j].ToleratesTaint(&taints[i]) {
				result = append(result, tolerations[i])
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

func getPodsAssignedToNode(c clientset.Interface, nodeName string) []v1.Pod {
	selector := fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
	pods, err := c.Core().Pods(v1.NamespaceAll).List(v1.ListOptions{
		FieldSelector: selector.String(),
		LabelSelector: labels.Everything().String(),
	})
	for i := 0; i < retries && err != nil; i++ {
		pods, err = c.Core().Pods(v1.NamespaceAll).List(v1.ListOptions{
			FieldSelector: selector.String(),
			LabelSelector: labels.Everything().String(),
		})
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		glog.Errorf("Failed to get Pods assigned to node %v. Skipping update.", nodeName)
	}
	return pods.Items
}

func getMinTolerationTime(tolerations []v1.Toleration) time.Duration {
	minTolerationTime := int64(-1)
	for i := range tolerations {
		if tolerations[i].TolerationSeconds != nil {
			if minTolerationTime < 0 {
				minTolerationTime = *(tolerations[i].TolerationSeconds)
			} else {
				if *(tolerations[i].TolerationSeconds) < minTolerationTime {
					minTolerationTime = *(tolerations[i].TolerationSeconds)
				}
			}
		}
	}
	return time.Duration(minTolerationTime) * time.Second
}

func NewTaintController(c clientset.Interface) *TaintController {
	return &TaintController{
		client:             c,
		taintEvictionQueue: CreateWorkerQueue(deletePodHandler(c)),
		taintedNodes:       make(map[string][]v1.Taint),
		nodeUpdateChannel:  make(chan nodeUpdateItem, nodeUpdateChannelSize),
		podUpdateChannel:   make(chan podUpdateItem, podUpdateChannelSize),
	}
}

func (tc *TaintController) Run(stopCh <-chan struct{}) {
	for {
		select {
		case <-stopCh:
			break
		case nodeUpdate := <-tc.nodeUpdateChannel:
			tc.handleNodeUpdate(nodeUpdate)
		case podUpdate := <-tc.podUpdateChannel:
		priority:
			for {
				select {
				case nodeUpdate := <-tc.nodeUpdateChannel:
					tc.handleNodeUpdate(nodeUpdate)
				default:
					break priority
				}
			}
			tc.handlePodUpdate(podUpdate)
		}
	}
}

func (tc *TaintController) PodUpdated(oldPod *v1.Pod, newPod *v1.Pod) {
	var err error
	oldTolerations := []v1.Toleration{}
	if oldPod != nil {
		oldTolerations, err = v1.GetPodTolerations(oldPod)
		if err != nil {
			glog.Errorf("Failed to get Tolerations from the old Pod")
			return
		}
	}
	newTolerations := []v1.Toleration{}
	if newPod != nil {
		newTolerations, err = v1.GetPodTolerations(newPod)
		if err != nil {
			glog.Errorf("Failed to get Tolerations from the new Pod")
			return
		}
	}

	if oldPod != newPod && api.Semantic.DeepEqual(oldTolerations, newTolerations) {
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
		panic(fmt.Errorf("TaintController Pod update queue is full"))
	}
}

func (tc *TaintController) NodeUpdated(oldNode *v1.Node, newNode *v1.Node) {
	var err error
	oldTaints := []v1.Taint{}
	if oldNode != nil {
		oldTaints, err = v1.GetNodeTaints(oldNode)
		if err != nil {
			glog.Errorf("Failed to get Taints from the old Node")
			return
		}
	}
	oldTaints = getNonExecuteTaints(oldTaints)

	newTaints := []v1.Taint{}
	if newNode != nil {
		newTaints, err = v1.GetNodeTaints(newNode)
		if err != nil {
			glog.Errorf("Failed to get Taints from the new Node")
			return
		}
	}
	newTaints = getNonExecuteTaints(newTaints)

	if oldNode != newNode && api.Semantic.DeepEqual(oldTaints, newTaints) {
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
		panic(fmt.Errorf("TaintController Node update queue is full"))
	}
}

func (tc *TaintController) processPodOnNode(
	podNamespacedName types.NamespacedName,
	nodeName string,
	tolerations []v1.Toleration,
	taints []v1.Taint,
	now time.Time,
) {
	allTolerated, usedTolerations := getMatchingTolerations(taints, tolerations)
	if !allTolerated {
		glog.Infof("Not all taints are tolerated after upgrade for Pod %v on %v", podNamespacedName.String(), nodeName)
		tc.taintEvictionQueue.CancelWork(podNamespacedName.String())
		tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), time.Now(), time.Now())
		return
	}
	minTolerationTime := getMinTolerationTime(usedTolerations)
	if minTolerationTime < 0 {
		glog.Infof("New tolerations for %v tolerate forever. Scheduled deletion won't be cancelled.", podNamespacedName.String())
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

func (tc *TaintController) handlePodUpdate(podUpdate podUpdateItem) {
	// Create
	if podUpdate.oldPod == nil {
		pod := podUpdate.newPod
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		nodeName := pod.Spec.NodeName
		glog.Infof("Noticed pod creation: %v", podNamespacedName.String())
		if nodeName == "" {
			return
		}
		taints, ok := tc.taintedNodes[nodeName]
		if !ok {
			return
		}
		tolerations := podUpdate.newTolerations
		tolerated, matchingTolerations := getMatchingTolerations(taints, tolerations)
		if tolerated {
			return
		}
		forever := true
		for i := range matchingTolerations {
			if matchingTolerations[i].TolerationSeconds != nil {
				forever = false
				break
			}
		}
		if !forever {
			glog.Infof("Pod %v that doesn't tolerate NoExecute forever taint on Node %v was scheduled there.", podNamespacedName.String(), nodeName)
			// It shouldn't have scheduled here in the first place - remove it right away
			tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), time.Now(), time.Now())
		}
		return
	}
	// Delete
	if podUpdate.newPod == nil {
		pod := podUpdate.newPod
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		glog.Infof("Noticed pod deletion: %v", podNamespacedName.String())
		tc.taintEvictionQueue.CancelWork(podNamespacedName.String())
		return
	}
	// Update
	pod := podUpdate.newPod
	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	glog.Infof("Noticed pod update: %v", podNamespacedName.String())
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		return
	}
	taints, ok := tc.taintedNodes[nodeName]
	if !ok {
		return
	}
	tolerations := podUpdate.newTolerations
	tc.processPodOnNode(podNamespacedName, nodeName, tolerations, taints, time.Now())
}

func (tc *TaintController) handleNodeUpdate(nodeUpdate nodeUpdateItem) {
	// Create
	if nodeUpdate.oldNode == nil {
		node := nodeUpdate.newNode
		taints := nodeUpdate.newTaints
		if len(taints) == 0 {
			return
		}
		tc.taintedNodes[node.Name] = taints
		pods := getPodsAssignedToNode(tc.client, node.Name)
		if len(pods) == 0 {
			return
		}
		now := time.Now()
		for i := range pods {
			tolerations, err := v1.GetPodTolerations(&pods[i])
			if err != nil {
				glog.Errorf("Failed to get tolerations from the Pod %v/%v, skipping", pods[i].Namespace, pods[i].Name)
				continue
			}
			podNamespacedName := types.NamespacedName{Namespace: pods[i].Namespace, Name: pods[i].Name}
			if len(tolerations) == 0 {
				tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), now, now)
				continue
			}
			tolerated, usedTolerations := getMatchingTolerations(taints, tolerations)
			if !tolerated {
				tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), now, now)
				continue
			}

			minTolerationTime := getMinTolerationTime(usedTolerations)
			if minTolerationTime < 0 {
				glog.Infof("New tolerations for %v tolerate forever. Scheduled deletion won't be cancelled.", podNamespacedName.String())
				return
			}
			tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), now, now.Add(minTolerationTime))
		}
	}
	// Delete
	if nodeUpdate.newNode == nil {
		node := nodeUpdate.oldNode
		delete(tc.taintedNodes, node.Name)
		return
	}
	// Update
	glog.Infof("Noticed node update: %v", nodeUpdate)
	node := nodeUpdate.newNode
	taints := getNonExecuteTaints(nodeUpdate.newTaints)
	pods := getPodsAssignedToNode(tc.client, node.Name)
	if len(pods) == 0 {
		return
	}

	now := time.Now()
	for i := range pods {
		pod := &pods[i]
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		tolerations, err := v1.GetPodTolerations(&pods[i])
		if err != nil {
			glog.Errorf("Failed to get Tolerations from Pod %v", podNamespacedName.String())
			continue
		}
		tc.processPodOnNode(podNamespacedName, node.Name, tolerations, taints, now)
	}
}
