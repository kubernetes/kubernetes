package scheduler

import (
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"
	utilpod "k8s.io/kubernetes/pkg/util/pod"
)

const (
	NodeUpdateChannelSize = 10
	UpdateWorkerSize      = 8
	retries               = 5
)

func hash(val string, max int) int {
	hasher := fnv.New32a()
	io.WriteString(hasher, val)
	return int(hasher.Sum32() % uint32(max))
}

// GetPodsByNodeNameFunc returns the list of pods assigned to the specified node.
type GetPodsByNodeNameFunc func(nodeName string) ([]*v1.Pod, error)

type nodeUpdateItem struct {
	nodeName string
}

type LabelManager struct {
	client                clientset.Interface
	broadcaster           record.EventBroadcaster
	recorder              record.EventRecorder
	podLister             corelisters.PodLister
	nodeLister            corelisters.NodeLister
	getPodsAssignedToNode GetPodsByNodeNameFunc

	nodeUpdateChannels []chan nodeUpdateItem
	nodeUpdateQueue    workqueue.Interface
}

func NewLabelManager(ctx context.Context, c clientset.Interface, podLister corelisters.PodLister, nodeLister corelisters.NodeLister, getPodsAssignedToNode GetPodsByNodeNameFunc) *LabelManager {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "node-label-manager"})

	ctl := &LabelManager{
		client:                c,
		broadcaster:           eventBroadcaster,
		recorder:              recorder,
		podLister:             podLister,
		nodeLister:            nodeLister,
		getPodsAssignedToNode: getPodsAssignedToNode,
		nodeUpdateQueue:       workqueue.NewNamed("node-label-manager"),
	}

	return ctl
}

func (c *LabelManager) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()
	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", "node-label-manager")
	defer logger.Info("Shutting down controller", "controller", "node-label-manager")

	// Start events processing pipeline.
	c.broadcaster.StartStructuredLogging(0)
	if c.client != nil {
		logger.Info("Sending events to api server")
		c.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.client.CoreV1().Events("")})
	} else {
		logger.Error(nil, "kubeClient is nil", "controller", "node-label-manager")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	defer c.broadcaster.Shutdown()

	defer c.nodeUpdateQueue.ShutDown()

	for i := 0; i < UpdateWorkerSize; i++ {
		c.nodeUpdateChannels = append(c.nodeUpdateChannels, make(chan nodeUpdateItem, NodeUpdateChannelSize))
	}

	go func(stopCh <-chan struct{}) {
		for {
			item, shutdown := c.nodeUpdateQueue.Get()
			if shutdown {
				break
			}
			nodeUpdate := item.(nodeUpdateItem)
			hash := hash(nodeUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-stopCh:
				c.nodeUpdateQueue.Done(item)
				return
			case c.nodeUpdateChannels[hash] <- nodeUpdate:
				// tc.nodeUpdateQueue.Done is called by the nodeUpdateChannels worker
			}
		}
	}(ctx.Done())

	wg := sync.WaitGroup{}
	wg.Add(UpdateWorkerSize)
	for i := 0; i < UpdateWorkerSize; i++ {
		go c.worker(ctx, i, wg.Done, ctx.Done())
	}
	wg.Wait()
}

func (c *LabelManager) worker(ctx context.Context, worker int, done func(), stopCh <-chan struct{}) {
	defer done()
	for {
		select {
		case <-stopCh:
			return
		case nodeUpdate := <-c.nodeUpdateChannels[worker]:
			c.handleNodeUpdate(ctx, nodeUpdate)
			c.nodeUpdateQueue.Done(nodeUpdate)
		}
	}
}

func (c *LabelManager) handleNodeUpdate(ctx context.Context, nodeUpdate nodeUpdateItem) {
	node, err := c.nodeLister.Get(nodeUpdate.nodeName)
	logger := klog.FromContext(ctx)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(4).Info("Noticed node deletion", "node", klog.KRef("", nodeUpdate.nodeName))
			return
		}
		utilruntime.HandleError(fmt.Errorf("cannot get node %s: %v", nodeUpdate.nodeName, err))
		return
	}

	logger.V(4).Info("Noticed node update", "node", klog.KObj(node))

	pods, err := c.getPodsAssignedToNode(node.Name)
	if err != nil {
		logger.Error(err, "Failed to get pods assigned to node", "node", klog.KObj(node))
		return
	}
	if len(pods) == 0 {
		return
	}

	for _, pod := range pods {
		c.processPodOnNode(ctx, pod, node)
	}
}

func (c *LabelManager) processPodOnNode(
	ctx context.Context,
	pod *v1.Pod,
	modifiedNode *v1.Node,
) {
	logger := klog.FromContext(ctx)

	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	requiredNodeAffinity := nodeaffinity.GetExecutionRequiredNodeAffinity(pod)
	isMatched, err := requiredNodeAffinity.Match(modifiedNode)
	if err != nil {
		logger.Error(err, "failed matching node affinity", "node", modifiedNode.Name, "pod", podNamespacedName)
		return
	}
	if isMatched {
		return
	}
	deletePod(ctx, c.client, podNamespacedName, "node-label-manager", c.emitPodDeletionEvent)
}

func (c *LabelManager) NodeUpdated(oldNode *v1.Node, newNode *v1.Node) {
	nodeName := ""
	oldLabels := make(map[string]string)
	if oldNode != nil {
		nodeName = oldNode.Name
		oldLabels = oldNode.Labels
	}

	newLabels := make(map[string]string)
	if newNode != nil {
		nodeName = newNode.Name
		newLabels = newNode.Labels
	}

	if oldNode != nil && newNode != nil && helper.Semantic.DeepEqual(oldLabels, newLabels) {
		return
	}
	updateItem := nodeUpdateItem{
		nodeName: nodeName,
	}
	c.nodeUpdateQueue.Add(updateItem)
}

func (c *LabelManager) emitPodDeletionEvent(nsName types.NamespacedName) {
	if c.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       nsName.Name,
		Namespace:  nsName.Namespace,
	}
	c.recorder.Eventf(ref, v1.EventTypeNormal, "NodeLabelsManagerEviction", "Marking for deletion Pod %s", nsName.String())
}

func deletePod(ctx context.Context, c clientset.Interface, nsName types.NamespacedName, controllerName string, emitEventFunc func(types.NamespacedName)) error {
	ns := nsName.Namespace
	name := nsName.Name
	klog.FromContext(ctx).Info("Deleting pod", "controller", controllerName, "pod", nsName)

	var err error
	for i := 0; i < retries; i++ {
		if err = addConditionAndDeletePod(ctx, c, name, ns); err == nil {
			if emitEventFunc != nil {
				emitEventFunc(nsName)
			}
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	return err
}

func addConditionAndDeletePod(ctx context.Context, c clientset.Interface, name, ns string) (err error) {
	if feature.DefaultFeatureGate.Enabled(features.PodDisruptionConditions) {
		pod, err := c.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		newStatus := pod.Status.DeepCopy()
		updated := apipod.UpdatePodCondition(newStatus, &v1.PodCondition{
			Type:    v1.DisruptionTarget,
			Status:  v1.ConditionTrue,
			Reason:  "DeletionByNodeLabelController",
			Message: "NodeLabelController: deleting due to unmatched node affinity",
		})
		if updated {
			if _, _, _, err := utilpod.PatchPodStatus(ctx, c, pod.Namespace, pod.Name, pod.UID, pod.Status, *newStatus); err != nil {
				return err
			}
		}
	}
	return c.CoreV1().Pods(ns).Delete(ctx, name, metav1.DeleteOptions{})
}
