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

	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"

	"github.com/golang/glog"
)

const (
	nodeUpdateChannelSize = 10
	podUpdateChannelSize  = 1
	retries               = 5
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

// Needed to make workqueue work
type updateItemInterface interface{}

type nodeUpdateItem struct {
	oldNode   *v1.Node
	newNode   *v1.Node
	newTaints []v1.Taint
}

type podUpdateItem struct {
	oldPod         *v1.Pod
	newPod         *v1.Pod
	newTolerations []v1.Toleration
}

// NoExecuteTaint manager listens to Taint/Toleration changes and is resposible for removing Pods
// from Nodes tainted with NoExecute Taints.
type NoExecuteTaintManager struct {
	client   clientset.Interface
	recorder record.EventRecorder

	taintEvictionQueue *TimedWorkerQueue
	// keeps a map from nodeName to all noExecute taints on that Node
	taintedNodesLock sync.Mutex
	taintedNodes     map[string][]v1.Taint

	nodeUpdateChannel chan *nodeUpdateItem
	podUpdateChannel  chan *podUpdateItem

	nodeUpdateQueue workqueue.Interface
	podUpdateQueue  workqueue.Interface
}

func deletePodHandler(c clientset.Interface, emitEventFunc func(types.NamespacedName)) func(args *WorkArgs) error {
	return func(args *WorkArgs) error {
		ns := args.NamespacedName.Namespace
		name := args.NamespacedName.Name
		glog.V(0).Infof("NoExecuteTaintManager is deleting Pod: %v", args.NamespacedName.String())
		if emitEventFunc != nil {
			emitEventFunc(args.NamespacedName)
		}
		var err error
		for i := 0; i < retries; i++ {
			err = c.Core().Pods(ns).Delete(name, &metav1.DeleteOptions{})
			if err == nil {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		return err
	}
}

func getNoExecuteTaints(taints []v1.Taint) []v1.Taint {
	result := []v1.Taint{}
	for i := range taints {
		if taints[i].Effect == v1.TaintEffectNoExecute {
			result = append(result, taints[i])
		}
	}
	return result
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
	if len(tolerations) == 0 {
		return 0
	}
	if tolerations[0].TolerationSeconds != nil {
		tolerationSeconds := *(tolerations[0].TolerationSeconds)
		if tolerationSeconds <= 0 {
			return 0
		} else {
			minTolerationTime = tolerationSeconds
		}
	}
	for i := range tolerations {
		if tolerations[i].TolerationSeconds != nil {
			tolerationSeconds := *(tolerations[i].TolerationSeconds)
			if tolerationSeconds <= 0 {
				return 0
			} else if tolerationSeconds < minTolerationTime {
				minTolerationTime = tolerationSeconds
			}
		}
	}
	return time.Duration(minTolerationTime) * time.Second
}

// NewNoExecuteTaintManager creates a new NoExecuteTaintManager that will use passed clientset to
// communicate with the API server.
func NewNoExecuteTaintManager(c clientset.Interface) *NoExecuteTaintManager {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "controllermanager"})
	eventBroadcaster.StartLogging(glog.Infof)
	if c != nil {
		glog.V(0).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(c.Core().RESTClient()).Events("")})
	} else {
		glog.Fatalf("kubeClient is nil when starting NodeController")
	}

	tm := &NoExecuteTaintManager{
		client:            c,
		recorder:          recorder,
		taintedNodes:      make(map[string][]v1.Taint),
		nodeUpdateChannel: make(chan *nodeUpdateItem, nodeUpdateChannelSize),
		podUpdateChannel:  make(chan *podUpdateItem, podUpdateChannelSize),

		nodeUpdateQueue: workqueue.New(),
		podUpdateQueue:  workqueue.New(),
	}
	tm.taintEvictionQueue = CreateWorkerQueue(deletePodHandler(c, tm.emitPodDeletionEvent))

	return tm
}

// Run starts NoExecuteTaintManager which will run in loop until `stopCh` is closed.
func (tc *NoExecuteTaintManager) Run(stopCh <-chan struct{}) {
	glog.V(0).Infof("Starting NoExecuteTaintManager")
	// Functions that are responsible for taking work items out of the workqueues and putting them
	// into channels.
	go func(stopCh <-chan struct{}) {
		for {
			item, shutdown := tc.nodeUpdateQueue.Get()
			if shutdown {
				break
			}
			nodeUpdate := item.(*nodeUpdateItem)
			select {
			case <-stopCh:
				break
			case tc.nodeUpdateChannel <- nodeUpdate:
			}
		}
	}(stopCh)

	go func(stopCh <-chan struct{}) {
		for {
			item, shutdown := tc.podUpdateQueue.Get()
			if shutdown {
				break
			}
			podUpdate := item.(*podUpdateItem)
			select {
			case <-stopCh:
				break
			case tc.podUpdateChannel <- podUpdate:
			}
		}
	}(stopCh)

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
	updateItem := &podUpdateItem{
		oldPod:         oldPod,
		newPod:         newPod,
		newTolerations: newTolerations,
	}

	tc.podUpdateQueue.Add(updateItemInterface(updateItem))
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
	oldTaints = getNoExecuteTaints(oldTaints)

	newTaints := []v1.Taint{}
	if newNode != nil {
		newTaints, err = v1.GetNodeTaints(newNode)
		if err != nil {
			glog.Errorf("Failed to get Taints from the new Node: %v", err)
			return
		}
	}
	newTaints = getNoExecuteTaints(newTaints)

	if oldNode != nil && newNode != nil && api.Semantic.DeepEqual(oldTaints, newTaints) {
		return
	}
	updateItem := &nodeUpdateItem{
		oldNode:   oldNode,
		newNode:   newNode,
		newTaints: newTaints,
	}

	tc.nodeUpdateQueue.Add(updateItemInterface(updateItem))
}

func (tc *NoExecuteTaintManager) cancelWorkWithEvent(nsName types.NamespacedName) {
	if tc.taintEvictionQueue.CancelWork(nsName.String()) {
		tc.emitCancelPodDeletionEvent(nsName)
	}
}

func (tc *NoExecuteTaintManager) processPodOnNode(
	podNamespacedName types.NamespacedName,
	nodeName string,
	tolerations []v1.Toleration,
	taints []v1.Taint,
	now time.Time,
) {
	if len(taints) == 0 {
		tc.cancelWorkWithEvent(podNamespacedName)
	}
	allTolerated, usedTolerations := v1.GetMatchingTolerations(taints, tolerations)
	if !allTolerated {
		glog.V(2).Infof("Not all taints are tolerated after update for Pod %v on %v", podNamespacedName.String(), nodeName)
		// We're canceling scheduled work (if any), as we're going to delete the Pod right away.
		tc.cancelWorkWithEvent(podNamespacedName)
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
			tc.cancelWorkWithEvent(podNamespacedName)
		}
	}
	tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), startTime, triggerTime)
}

func (tc *NoExecuteTaintManager) handlePodUpdate(podUpdate *podUpdateItem) {
	// Delete
	if podUpdate.newPod == nil {
		pod := podUpdate.oldPod
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		glog.V(4).Infof("Noticed pod deletion: %#v", podNamespacedName)
		tc.cancelWorkWithEvent(podNamespacedName)
		return
	}
	// Create or Update
	pod := podUpdate.newPod
	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	glog.V(4).Infof("Noticed pod update: %#v", podNamespacedName)
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
	// It's possible that Node was deleted, or Taints were removed before, which triggered
	// eviction cancelling if it was needed.
	if !ok {
		return
	}
	tc.processPodOnNode(podNamespacedName, nodeName, podUpdate.newTolerations, taints, time.Now())
}

func (tc *NoExecuteTaintManager) handleNodeUpdate(nodeUpdate *nodeUpdateItem) {
	// Delete
	if nodeUpdate.newNode == nil {
		node := nodeUpdate.oldNode
		glog.V(4).Infof("Noticed node deletion: %#v", node.Name)
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		delete(tc.taintedNodes, node.Name)
		return
	}
	// Create or Update
	glog.V(4).Infof("Noticed node update: %#v", nodeUpdate)
	node := nodeUpdate.newNode
	taints := nodeUpdate.newTaints
	func() {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		glog.V(4).Infof("Updating known taints on node %v: %v", node.Name, taints)
		if len(taints) == 0 {
			delete(tc.taintedNodes, node.Name)
		} else {
			tc.taintedNodes[node.Name] = taints
		}
	}()
	pods, err := getPodsAssignedToNode(tc.client, node.Name)
	if err != nil {
		glog.Errorf(err.Error())
		return
	}
	if len(pods) == 0 {
		return
	}
	// Short circuit, to make this controller a bit faster.
	if len(taints) == 0 {
		glog.V(4).Infof("All taints were removed from the Node %v. Cancelling all evictions...", node.Name)
		for i := range pods {
			tc.cancelWorkWithEvent(types.NamespacedName{Namespace: pods[i].Namespace, Name: pods[i].Name})
		}
		return
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

func (tc *NoExecuteTaintManager) emitPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &clientv1.ObjectReference{
		Kind:      "Pod",
		Name:      nsName.Name,
		Namespace: nsName.Namespace,
	}
	tc.recorder.Eventf(ref, clientv1.EventTypeNormal, "TaintManagerEviction", "Marking for deletion Pod %s", nsName.String())
}

func (tc *NoExecuteTaintManager) emitCancelPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &clientv1.ObjectReference{
		Kind:      "Pod",
		Name:      nsName.Name,
		Namespace: nsName.Namespace,
	}
	tc.recorder.Eventf(ref, clientv1.EventTypeNormal, "TaintManagerEviction", "Cancelling deletion of Pod %s", nsName.String())
}
