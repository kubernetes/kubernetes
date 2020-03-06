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

package scheduler

import (
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"

	"k8s.io/klog"
)

const (
	// TODO (k82cn): Figure out a reasonable number of workers/channels and propagate
	// the number of workers up making it a parameter of Run() function.

	// NodeUpdateChannelSize defines the size of channel for node update events.
	NodeUpdateChannelSize = 10
	// UpdateWorkerSize defines the size of workers for node update or/and pod update.
	UpdateWorkerSize     = 8
	podUpdateChannelSize = 1
	retries              = 5
)

type nodeUpdateItem struct {
	nodeName string
}

type podUpdateItem struct {
	podName      string
	podNamespace string
	nodeName     string
}

func hash(val string, max int) int {
	hasher := fnv.New32a()
	io.WriteString(hasher, val)
	return int(hasher.Sum32() % uint32(max))
}

// GetPodFunc returns the pod for the specified name/namespace, or a NotFound error if missing.
type GetPodFunc func(name, namespace string) (*v1.Pod, error)

// GetNodeFunc returns the node for the specified name, or a NotFound error if missing.
type GetNodeFunc func(name string) (*v1.Node, error)

// GetPodsByNodeNameFunc returns the list of pods assigned to the specified node.
type GetPodsByNodeNameFunc func(nodeName string) ([]*v1.Pod, error)

// NoExecuteTaintManager listens to Taint/Toleration changes and is responsible for removing Pods
// from Nodes tainted with NoExecute Taints.
type NoExecuteTaintManager struct {
	client                clientset.Interface
	recorder              record.EventRecorder
	getPod                GetPodFunc
	getNode               GetNodeFunc
	getPodsAssignedToNode GetPodsByNodeNameFunc

	taintEvictionQueue *TimedWorkerQueue
	// keeps a map from nodeName to all noExecute taints on that Node
	taintedNodesLock sync.Mutex
	taintedNodes     map[string][]v1.Taint

	nodeUpdateChannels []chan nodeUpdateItem
	podUpdateChannels  []chan podUpdateItem

	nodeUpdateQueue workqueue.Interface
	podUpdateQueue  workqueue.Interface
}

func deletePodHandler(c clientset.Interface, emitEventFunc func(types.NamespacedName)) func(args *WorkArgs) error {
	return func(args *WorkArgs) error {
		ns := args.NamespacedName.Namespace
		name := args.NamespacedName.Name
		klog.V(0).Infof("NoExecuteTaintManager is deleting Pod: %v", args.NamespacedName.String())
		if emitEventFunc != nil {
			emitEventFunc(args.NamespacedName)
		}
		var err error
		for i := 0; i < retries; i++ {
			err = c.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
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

// getMinTolerationTime returns minimal toleration time from the given slice, or -1 if it's infinite.
func getMinTolerationTime(tolerations []v1.Toleration) time.Duration {
	minTolerationTime := int64(math.MaxInt64)
	if len(tolerations) == 0 {
		return 0
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

	if minTolerationTime == int64(math.MaxInt64) {
		return -1
	}
	return time.Duration(minTolerationTime) * time.Second
}

// NewNoExecuteTaintManager creates a new NoExecuteTaintManager that will use passed clientset to
// communicate with the API server.
func NewNoExecuteTaintManager(c clientset.Interface, getPod GetPodFunc, getNode GetNodeFunc, getPodsAssignedToNode GetPodsByNodeNameFunc) *NoExecuteTaintManager {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "taint-controller"})
	eventBroadcaster.StartLogging(klog.Infof)
	if c != nil {
		klog.V(0).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.CoreV1().Events("")})
	} else {
		klog.Fatalf("kubeClient is nil when starting NodeController")
	}

	tm := &NoExecuteTaintManager{
		client:                c,
		recorder:              recorder,
		getPod:                getPod,
		getNode:               getNode,
		getPodsAssignedToNode: getPodsAssignedToNode,
		taintedNodes:          make(map[string][]v1.Taint),

		nodeUpdateQueue: workqueue.NewNamed("noexec_taint_node"),
		podUpdateQueue:  workqueue.NewNamed("noexec_taint_pod"),
	}
	tm.taintEvictionQueue = CreateWorkerQueue(deletePodHandler(c, tm.emitPodDeletionEvent))

	return tm
}

// Run starts NoExecuteTaintManager which will run in loop until `stopCh` is closed.
func (tc *NoExecuteTaintManager) Run(stopCh <-chan struct{}) {
	klog.V(0).Infof("Starting NoExecuteTaintManager")

	for i := 0; i < UpdateWorkerSize; i++ {
		tc.nodeUpdateChannels = append(tc.nodeUpdateChannels, make(chan nodeUpdateItem, NodeUpdateChannelSize))
		tc.podUpdateChannels = append(tc.podUpdateChannels, make(chan podUpdateItem, podUpdateChannelSize))
	}

	// Functions that are responsible for taking work items out of the workqueues and putting them
	// into channels.
	go func(stopCh <-chan struct{}) {
		for {
			item, shutdown := tc.nodeUpdateQueue.Get()
			if shutdown {
				break
			}
			nodeUpdate := item.(nodeUpdateItem)
			hash := hash(nodeUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-stopCh:
				tc.nodeUpdateQueue.Done(item)
				return
			case tc.nodeUpdateChannels[hash] <- nodeUpdate:
				// tc.nodeUpdateQueue.Done is called by the nodeUpdateChannels worker
			}
		}
	}(stopCh)

	go func(stopCh <-chan struct{}) {
		for {
			item, shutdown := tc.podUpdateQueue.Get()
			if shutdown {
				break
			}
			// The fact that pods are processed by the same worker as nodes is used to avoid races
			// between node worker setting tc.taintedNodes and pod worker reading this to decide
			// whether to delete pod.
			// It's possible that even without this assumption this code is still correct.
			podUpdate := item.(podUpdateItem)
			hash := hash(podUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-stopCh:
				tc.podUpdateQueue.Done(item)
				return
			case tc.podUpdateChannels[hash] <- podUpdate:
				// tc.podUpdateQueue.Done is called by the podUpdateChannels worker
			}
		}
	}(stopCh)

	wg := sync.WaitGroup{}
	wg.Add(UpdateWorkerSize)
	for i := 0; i < UpdateWorkerSize; i++ {
		go tc.worker(i, wg.Done, stopCh)
	}
	wg.Wait()
}

func (tc *NoExecuteTaintManager) worker(worker int, done func(), stopCh <-chan struct{}) {
	defer done()

	// When processing events we want to prioritize Node updates over Pod updates,
	// as NodeUpdates that interest NoExecuteTaintManager should be handled as soon as possible -
	// we don't want user (or system) to wait until PodUpdate queue is drained before it can
	// start evicting Pods from tainted Nodes.
	for {
		select {
		case <-stopCh:
			return
		case nodeUpdate := <-tc.nodeUpdateChannels[worker]:
			tc.handleNodeUpdate(nodeUpdate)
			tc.nodeUpdateQueue.Done(nodeUpdate)
		case podUpdate := <-tc.podUpdateChannels[worker]:
			// If we found a Pod update we need to empty Node queue first.
		priority:
			for {
				select {
				case nodeUpdate := <-tc.nodeUpdateChannels[worker]:
					tc.handleNodeUpdate(nodeUpdate)
					tc.nodeUpdateQueue.Done(nodeUpdate)
				default:
					break priority
				}
			}
			// After Node queue is emptied we process podUpdate.
			tc.handlePodUpdate(podUpdate)
			tc.podUpdateQueue.Done(podUpdate)
		}
	}
}

// PodUpdated is used to notify NoExecuteTaintManager about Pod changes.
func (tc *NoExecuteTaintManager) PodUpdated(oldPod *v1.Pod, newPod *v1.Pod) {
	podName := ""
	podNamespace := ""
	nodeName := ""
	oldTolerations := []v1.Toleration{}
	if oldPod != nil {
		podName = oldPod.Name
		podNamespace = oldPod.Namespace
		nodeName = oldPod.Spec.NodeName
		oldTolerations = oldPod.Spec.Tolerations
	}
	newTolerations := []v1.Toleration{}
	if newPod != nil {
		podName = newPod.Name
		podNamespace = newPod.Namespace
		nodeName = newPod.Spec.NodeName
		newTolerations = newPod.Spec.Tolerations
	}

	if oldPod != nil && newPod != nil && helper.Semantic.DeepEqual(oldTolerations, newTolerations) && oldPod.Spec.NodeName == newPod.Spec.NodeName {
		return
	}
	updateItem := podUpdateItem{
		podName:      podName,
		podNamespace: podNamespace,
		nodeName:     nodeName,
	}

	tc.podUpdateQueue.Add(updateItem)
}

// NodeUpdated is used to notify NoExecuteTaintManager about Node changes.
func (tc *NoExecuteTaintManager) NodeUpdated(oldNode *v1.Node, newNode *v1.Node) {
	nodeName := ""
	oldTaints := []v1.Taint{}
	if oldNode != nil {
		nodeName = oldNode.Name
		oldTaints = getNoExecuteTaints(oldNode.Spec.Taints)
	}

	newTaints := []v1.Taint{}
	if newNode != nil {
		nodeName = newNode.Name
		newTaints = getNoExecuteTaints(newNode.Spec.Taints)
	}

	if oldNode != nil && newNode != nil && helper.Semantic.DeepEqual(oldTaints, newTaints) {
		return
	}
	updateItem := nodeUpdateItem{
		nodeName: nodeName,
	}

	tc.nodeUpdateQueue.Add(updateItem)
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
	allTolerated, usedTolerations := v1helper.GetMatchingTolerations(taints, tolerations)
	if !allTolerated {
		klog.V(2).Infof("Not all taints are tolerated after update for Pod %v on %v", podNamespacedName.String(), nodeName)
		// We're canceling scheduled work (if any), as we're going to delete the Pod right away.
		tc.cancelWorkWithEvent(podNamespacedName)
		tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), time.Now(), time.Now())
		return
	}
	minTolerationTime := getMinTolerationTime(usedTolerations)
	// getMinTolerationTime returns negative value to denote infinite toleration.
	if minTolerationTime < 0 {
		klog.V(4).Infof("New tolerations for %v tolerate forever. Scheduled deletion won't be cancelled if already scheduled.", podNamespacedName.String())
		return
	}

	startTime := now
	triggerTime := startTime.Add(minTolerationTime)
	scheduledEviction := tc.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String())
	if scheduledEviction != nil {
		startTime = scheduledEviction.CreatedAt
		if startTime.Add(minTolerationTime).Before(triggerTime) {
			return
		}
		tc.cancelWorkWithEvent(podNamespacedName)
	}
	tc.taintEvictionQueue.AddWork(NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), startTime, triggerTime)
}

func (tc *NoExecuteTaintManager) handlePodUpdate(podUpdate podUpdateItem) {
	pod, err := tc.getPod(podUpdate.podName, podUpdate.podNamespace)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Delete
			podNamespacedName := types.NamespacedName{Namespace: podUpdate.podNamespace, Name: podUpdate.podName}
			klog.V(4).Infof("Noticed pod deletion: %#v", podNamespacedName)
			tc.cancelWorkWithEvent(podNamespacedName)
			return
		}
		utilruntime.HandleError(fmt.Errorf("could not get pod %s/%s: %v", podUpdate.podName, podUpdate.podNamespace, err))
		return
	}

	// We key the workqueue and shard workers by nodeName. If we don't match the current state we should not be the one processing the current object.
	if pod.Spec.NodeName != podUpdate.nodeName {
		return
	}

	// Create or Update
	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	klog.V(4).Infof("Noticed pod update: %#v", podNamespacedName)
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
	tc.processPodOnNode(podNamespacedName, nodeName, pod.Spec.Tolerations, taints, time.Now())
}

func (tc *NoExecuteTaintManager) handleNodeUpdate(nodeUpdate nodeUpdateItem) {
	node, err := tc.getNode(nodeUpdate.nodeName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Delete
			klog.V(4).Infof("Noticed node deletion: %#v", nodeUpdate.nodeName)
			tc.taintedNodesLock.Lock()
			defer tc.taintedNodesLock.Unlock()
			delete(tc.taintedNodes, nodeUpdate.nodeName)
			return
		}
		utilruntime.HandleError(fmt.Errorf("cannot get node %s: %v", nodeUpdate.nodeName, err))
		return
	}

	// Create or Update
	klog.V(4).Infof("Noticed node update: %#v", nodeUpdate)
	taints := getNoExecuteTaints(node.Spec.Taints)
	func() {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		klog.V(4).Infof("Updating known taints on node %v: %v", node.Name, taints)
		if len(taints) == 0 {
			delete(tc.taintedNodes, node.Name)
		} else {
			tc.taintedNodes[node.Name] = taints
		}
	}()

	// This is critical that we update tc.taintedNodes before we call getPodsAssignedToNode:
	// getPodsAssignedToNode can be delayed as long as all future updates to pods will call
	// tc.PodUpdated which will use tc.taintedNodes to potentially delete delayed pods.
	pods, err := tc.getPodsAssignedToNode(node.Name)
	if err != nil {
		klog.Errorf(err.Error())
		return
	}
	if len(pods) == 0 {
		return
	}
	// Short circuit, to make this controller a bit faster.
	if len(taints) == 0 {
		klog.V(4).Infof("All taints were removed from the Node %v. Cancelling all evictions...", node.Name)
		for i := range pods {
			tc.cancelWorkWithEvent(types.NamespacedName{Namespace: pods[i].Namespace, Name: pods[i].Name})
		}
		return
	}

	now := time.Now()
	for _, pod := range pods {
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		tc.processPodOnNode(podNamespacedName, node.Name, pod.Spec.Tolerations, taints, now)
	}
}

func (tc *NoExecuteTaintManager) emitPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		Kind:      "Pod",
		Name:      nsName.Name,
		Namespace: nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "TaintManagerEviction", "Marking for deletion Pod %s", nsName.String())
}

func (tc *NoExecuteTaintManager) emitCancelPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		Kind:      "Pod",
		Name:      nsName.Name,
		Namespace: nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "TaintManagerEviction", "Cancelling deletion of Pod %s", nsName.String())
}
