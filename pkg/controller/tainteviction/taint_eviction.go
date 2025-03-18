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

package tainteviction

import (
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	corev1informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller/tainteviction/metrics"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
	utilpod "k8s.io/kubernetes/pkg/util/pod"
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

// GetPodsByNodeNameFunc returns the list of pods assigned to the specified node.
type GetPodsByNodeNameFunc func(nodeName string) ([]*v1.Pod, error)

// Controller listens to Taint/Toleration changes and is responsible for removing Pods
// from Nodes tainted with NoExecute Taints.
type Controller struct {
	name string

	client                clientset.Interface
	broadcaster           record.EventBroadcaster
	recorder              record.EventRecorder
	podLister             corelisters.PodLister
	podListerSynced       cache.InformerSynced
	nodeLister            corelisters.NodeLister
	nodeListerSynced      cache.InformerSynced
	getPodsAssignedToNode GetPodsByNodeNameFunc

	taintEvictionQueue *TimedWorkerQueue
	// keeps a map from nodeName to all noExecute taints on that Node
	taintedNodesLock sync.Mutex
	taintedNodes     map[string][]v1.Taint

	nodeUpdateChannels []chan nodeUpdateItem
	podUpdateChannels  []chan podUpdateItem

	nodeUpdateQueue workqueue.TypedInterface[nodeUpdateItem]
	podUpdateQueue  workqueue.TypedInterface[podUpdateItem]
}

func deletePodHandler(c clientset.Interface, emitEventFunc func(types.NamespacedName), controllerName string) func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
	return func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		ns := args.NamespacedName.Namespace
		name := args.NamespacedName.Name
		klog.FromContext(ctx).Info("Deleting pod", "controller", controllerName, "pod", args.NamespacedName)
		if emitEventFunc != nil {
			emitEventFunc(args.NamespacedName)
		}
		var err error
		for i := 0; i < retries; i++ {
			err = addConditionAndDeletePod(ctx, c, name, ns)
			if err == nil {
				metrics.PodDeletionsTotal.Inc()
				metrics.PodDeletionsLatency.Observe(float64(time.Since(fireAt) * time.Second))
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		return err
	}
}

func addConditionAndDeletePod(ctx context.Context, c clientset.Interface, name, ns string) (err error) {
	pod, err := c.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	newStatus := pod.Status.DeepCopy()
	updated := apipod.UpdatePodCondition(newStatus, &v1.PodCondition{
		Type:               v1.DisruptionTarget,
		ObservedGeneration: apipod.GetPodObservedGenerationIfEnabledOnCondition(pod, v1.DisruptionTarget),
		Status:             v1.ConditionTrue,
		Reason:             "DeletionByTaintManager",
		Message:            "Taint manager: deleting due to NoExecute taint",
	})
	if updated {
		if _, _, _, err := utilpod.PatchPodStatus(ctx, c, pod.Namespace, pod.Name, pod.UID, pod.Status, *newStatus); err != nil {
			return err
		}
	}
	return c.CoreV1().Pods(ns).Delete(ctx, name, metav1.DeleteOptions{})
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

// New creates a new Controller that will use passed clientset to communicate with the API server.
func New(ctx context.Context, c clientset.Interface, podInformer corev1informers.PodInformer, nodeInformer corev1informers.NodeInformer, controllerName string) (*Controller, error) {
	logger := klog.FromContext(ctx)
	metrics.Register()
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName})

	podIndexer := podInformer.Informer().GetIndexer()

	tm := &Controller{
		name: controllerName,

		client:           c,
		broadcaster:      eventBroadcaster,
		recorder:         recorder,
		podLister:        podInformer.Lister(),
		podListerSynced:  podInformer.Informer().HasSynced,
		nodeLister:       nodeInformer.Lister(),
		nodeListerSynced: nodeInformer.Informer().HasSynced,
		getPodsAssignedToNode: func(nodeName string) ([]*v1.Pod, error) {
			objs, err := podIndexer.ByIndex("spec.nodeName", nodeName)
			if err != nil {
				return nil, err
			}
			pods := make([]*v1.Pod, 0, len(objs))
			for _, obj := range objs {
				pod, ok := obj.(*v1.Pod)
				if !ok {
					continue
				}
				pods = append(pods, pod)
			}
			return pods, nil
		},
		taintedNodes: make(map[string][]v1.Taint),

		nodeUpdateQueue: workqueue.NewTypedWithConfig(workqueue.TypedQueueConfig[nodeUpdateItem]{Name: "noexec_taint_node"}),
		podUpdateQueue:  workqueue.NewTypedWithConfig(workqueue.TypedQueueConfig[podUpdateItem]{Name: "noexec_taint_pod"}),
	}
	tm.taintEvictionQueue = CreateWorkerQueue(deletePodHandler(c, tm.emitPodDeletionEvent, tm.name))

	_, err := podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod := obj.(*v1.Pod)
			tm.PodUpdated(nil, pod)
		},
		UpdateFunc: func(prev, obj interface{}) {
			prevPod := prev.(*v1.Pod)
			newPod := obj.(*v1.Pod)
			tm.PodUpdated(prevPod, newPod)
		},
		DeleteFunc: func(obj interface{}) {
			pod, isPod := obj.(*v1.Pod)
			// We can get DeletedFinalStateUnknown instead of *v1.Pod here and we need to handle that correctly.
			if !isPod {
				deletedState, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					logger.Error(nil, "Received unexpected object", "object", obj)
					return
				}
				pod, ok = deletedState.Obj.(*v1.Pod)
				if !ok {
					logger.Error(nil, "DeletedFinalStateUnknown contained non-Pod object", "object", deletedState.Obj)
					return
				}
			}
			tm.PodUpdated(pod, nil)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("unable to add pod event handler: %w", err)
	}

	_, err = nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controllerutil.CreateAddNodeHandler(func(node *v1.Node) error {
			tm.NodeUpdated(nil, node)
			return nil
		}),
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(oldNode, newNode *v1.Node) error {
			tm.NodeUpdated(oldNode, newNode)
			return nil
		}),
		DeleteFunc: controllerutil.CreateDeleteNodeHandler(logger, func(node *v1.Node) error {
			tm.NodeUpdated(node, nil)
			return nil
		}),
	})
	if err != nil {
		return nil, fmt.Errorf("unable to add node event handler: %w", err)
	}

	return tm, nil
}

// Run starts the controller which will run in loop until `stopCh` is closed.
func (tc *Controller) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()
	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", tc.name)
	defer logger.Info("Shutting down controller", "controller", tc.name)

	// Start events processing pipeline.
	tc.broadcaster.StartStructuredLogging(3)
	if tc.client != nil {
		logger.Info("Sending events to api server")
		tc.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: tc.client.CoreV1().Events("")})
	} else {
		logger.Error(nil, "kubeClient is nil", "controller", tc.name)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	defer tc.broadcaster.Shutdown()
	defer tc.nodeUpdateQueue.ShutDown()
	defer tc.podUpdateQueue.ShutDown()

	// wait for the cache to be synced
	if !cache.WaitForNamedCacheSync(tc.name, ctx.Done(), tc.podListerSynced, tc.nodeListerSynced) {
		return
	}

	for i := 0; i < UpdateWorkerSize; i++ {
		tc.nodeUpdateChannels = append(tc.nodeUpdateChannels, make(chan nodeUpdateItem, NodeUpdateChannelSize))
		tc.podUpdateChannels = append(tc.podUpdateChannels, make(chan podUpdateItem, podUpdateChannelSize))
	}

	// Functions that are responsible for taking work items out of the workqueues and putting them
	// into channels.
	go func(stopCh <-chan struct{}) {
		for {
			nodeUpdate, shutdown := tc.nodeUpdateQueue.Get()
			if shutdown {
				break
			}
			hash := hash(nodeUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-stopCh:
				tc.nodeUpdateQueue.Done(nodeUpdate)
				return
			case tc.nodeUpdateChannels[hash] <- nodeUpdate:
				// tc.nodeUpdateQueue.Done is called by the nodeUpdateChannels worker
			}
		}
	}(ctx.Done())

	go func(stopCh <-chan struct{}) {
		for {
			podUpdate, shutdown := tc.podUpdateQueue.Get()
			if shutdown {
				break
			}
			// The fact that pods are processed by the same worker as nodes is used to avoid races
			// between node worker setting tc.taintedNodes and pod worker reading this to decide
			// whether to delete pod.
			// It's possible that even without this assumption this code is still correct.
			hash := hash(podUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-stopCh:
				tc.podUpdateQueue.Done(podUpdate)
				return
			case tc.podUpdateChannels[hash] <- podUpdate:
				// tc.podUpdateQueue.Done is called by the podUpdateChannels worker
			}
		}
	}(ctx.Done())

	wg := sync.WaitGroup{}
	wg.Add(UpdateWorkerSize)
	for i := 0; i < UpdateWorkerSize; i++ {
		go tc.worker(ctx, i, wg.Done, ctx.Done())
	}
	wg.Wait()
}

func (tc *Controller) worker(ctx context.Context, worker int, done func(), stopCh <-chan struct{}) {
	defer done()

	// When processing events we want to prioritize Node updates over Pod updates,
	// as NodeUpdates that interest the controller should be handled as soon as possible -
	// we don't want user (or system) to wait until PodUpdate queue is drained before it can
	// start evicting Pods from tainted Nodes.
	for {
		select {
		case <-stopCh:
			return
		case nodeUpdate := <-tc.nodeUpdateChannels[worker]:
			tc.handleNodeUpdate(ctx, nodeUpdate)
			tc.nodeUpdateQueue.Done(nodeUpdate)
		case podUpdate := <-tc.podUpdateChannels[worker]:
			// If we found a Pod update we need to empty Node queue first.
		priority:
			for {
				select {
				case nodeUpdate := <-tc.nodeUpdateChannels[worker]:
					tc.handleNodeUpdate(ctx, nodeUpdate)
					tc.nodeUpdateQueue.Done(nodeUpdate)
				default:
					break priority
				}
			}
			// After Node queue is emptied we process podUpdate.
			tc.handlePodUpdate(ctx, podUpdate)
			tc.podUpdateQueue.Done(podUpdate)
		}
	}
}

// PodUpdated is used to notify NoExecuteTaintManager about Pod changes.
func (tc *Controller) PodUpdated(oldPod *v1.Pod, newPod *v1.Pod) {
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
func (tc *Controller) NodeUpdated(oldNode *v1.Node, newNode *v1.Node) {
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

func (tc *Controller) cancelWorkWithEvent(logger klog.Logger, nsName types.NamespacedName) {
	if tc.taintEvictionQueue.CancelWork(logger, nsName.String()) {
		tc.emitCancelPodDeletionEvent(nsName)
	}
}

func (tc *Controller) processPodOnNode(
	ctx context.Context,
	podNamespacedName types.NamespacedName,
	nodeName string,
	tolerations []v1.Toleration,
	taints []v1.Taint,
	now time.Time,
) {
	logger := klog.FromContext(ctx)
	if len(taints) == 0 {
		tc.cancelWorkWithEvent(logger, podNamespacedName)
	}
	allTolerated, usedTolerations := v1helper.GetMatchingTolerations(taints, tolerations)
	if !allTolerated {
		logger.V(2).Info("Not all taints are tolerated after update for pod on node", "pod", podNamespacedName.String(), "node", klog.KRef("", nodeName))
		// We're canceling scheduled work (if any), as we're going to delete the Pod right away.
		tc.cancelWorkWithEvent(logger, podNamespacedName)
		tc.taintEvictionQueue.AddWork(ctx, NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), now, now)
		return
	}
	minTolerationTime := getMinTolerationTime(usedTolerations)
	// getMinTolerationTime returns negative value to denote infinite toleration.
	if minTolerationTime < 0 {
		logger.V(4).Info("Current tolerations for pod tolerate forever, cancelling any scheduled deletion", "pod", podNamespacedName.String())
		tc.cancelWorkWithEvent(logger, podNamespacedName)
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
		tc.cancelWorkWithEvent(logger, podNamespacedName)
	}
	tc.taintEvictionQueue.AddWork(ctx, NewWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace), startTime, triggerTime)
}

func (tc *Controller) handlePodUpdate(ctx context.Context, podUpdate podUpdateItem) {
	pod, err := tc.podLister.Pods(podUpdate.podNamespace).Get(podUpdate.podName)
	logger := klog.FromContext(ctx)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Delete
			podNamespacedName := types.NamespacedName{Namespace: podUpdate.podNamespace, Name: podUpdate.podName}
			logger.V(4).Info("Noticed pod deletion", "pod", podNamespacedName)
			tc.cancelWorkWithEvent(logger, podNamespacedName)
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
	logger.V(4).Info("Noticed pod update", "pod", podNamespacedName)
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
	tc.processPodOnNode(ctx, podNamespacedName, nodeName, pod.Spec.Tolerations, taints, time.Now())
}

func (tc *Controller) handleNodeUpdate(ctx context.Context, nodeUpdate nodeUpdateItem) {
	node, err := tc.nodeLister.Get(nodeUpdate.nodeName)
	logger := klog.FromContext(ctx)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Delete
			logger.V(4).Info("Noticed node deletion", "node", klog.KRef("", nodeUpdate.nodeName))
			tc.taintedNodesLock.Lock()
			defer tc.taintedNodesLock.Unlock()
			delete(tc.taintedNodes, nodeUpdate.nodeName)
			return
		}
		utilruntime.HandleError(fmt.Errorf("cannot get node %s: %v", nodeUpdate.nodeName, err))
		return
	}

	// Create or Update
	logger.V(4).Info("Noticed node update", "node", klog.KObj(node))
	taints := getNoExecuteTaints(node.Spec.Taints)
	func() {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		logger.V(4).Info("Updating known taints on node", "node", klog.KObj(node), "taints", taints)
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
		logger.Error(err, "Failed to get pods assigned to node", "node", klog.KObj(node))
		return
	}
	if len(pods) == 0 {
		return
	}
	// Short circuit, to make this controller a bit faster.
	if len(taints) == 0 {
		logger.V(4).Info("All taints were removed from the node. Cancelling all evictions...", "node", klog.KObj(node))
		for i := range pods {
			tc.cancelWorkWithEvent(logger, types.NamespacedName{Namespace: pods[i].Namespace, Name: pods[i].Name})
		}
		return
	}

	now := time.Now()
	for _, pod := range pods {
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		tc.processPodOnNode(ctx, podNamespacedName, node.Name, pod.Spec.Tolerations, taints, now)
	}
}

func (tc *Controller) emitPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       nsName.Name,
		Namespace:  nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "TaintManagerEviction", "Marking for deletion Pod %s", nsName.String())
}

func (tc *Controller) emitCancelPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       nsName.Name,
		Namespace:  nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "TaintManagerEviction", "Cancelling deletion of Pod %s", nsName.String())
}
