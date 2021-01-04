package scheduler

import (
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

// ExecutionNodeAffinityManager listens to Node label changes and is responsible for removing Pods from Nodes.
type ExecutionNodeAffinityManager struct {
	client   clientset.Interface
	recorder record.EventRecorder

	getNode               GetNodeFunc
	getPodsAssignedToNode GetPodsByNodeNameFunc

	nodeUpdateChannels []chan nodeUpdateItem
	nodeUpdateQueue    workqueue.Interface

	affinityEvictionQueue *TimedWorkerQueue
}

// NewNodeAffinityManager creates a new ExecutionNodeAffinityManager
func NewNodeAffinityManager(c clientset.Interface, getNode GetNodeFunc, getPodsAssignedToNode GetPodsByNodeNameFunc) *ExecutionNodeAffinityManager {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "node-affinity-controller"})
	eventBroadcaster.StartStructuredLogging(0)
	if c != nil {
		klog.V(4).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.CoreV1().Events("")})
	} else {
		klog.Fatalf("kubeClient is nil when starting NodeController")
	}

	m := &ExecutionNodeAffinityManager{
		client:                c,
		recorder:              recorder,
		getNode:               getNode,
		getPodsAssignedToNode: getPodsAssignedToNode,
		nodeUpdateQueue:       workqueue.NewNamed("node_affinity_evict_node"),
	}
	m.affinityEvictionQueue = CreateWorkerQueue(deletePodHandler(c, m.emitPodDeletionEvent))

	return m
}

func (m *ExecutionNodeAffinityManager) emitPodDeletionEvent(nsName types.NamespacedName) {
	klog.V(4).Infof("NewNodeAffinityManager is deleting Pod: %v", nsName.String())

	if m.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		Kind:      "Pod",
		Name:      nsName.Name,
		Namespace: nsName.Namespace,
	}
	m.recorder.Eventf(ref, v1.EventTypeNormal, "NodeAffinityManagerEviction", "Marking for deletion Pod %s", nsName.String())
}

func (m *ExecutionNodeAffinityManager) emitCancelPodDeletionEvent(nsName types.NamespacedName) {
	if m.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		Kind:      "Pod",
		Name:      nsName.Name,
		Namespace: nsName.Namespace,
	}
	m.recorder.Eventf(ref, v1.EventTypeNormal, "NodeAffinityManagerEviction", "Cancelling deletion of Pod %s", nsName.String())
}

// NodeUpdated is used to notify ExecutionNodeAffinityManager about Node changes.
func (m *ExecutionNodeAffinityManager) NodeUpdated(oldNode *v1.Node, newNode *v1.Node) {
	nodeName := ""
	var oldLabels map[string]string
	if oldNode != nil {
		nodeName = oldNode.Name
		oldLabels = oldNode.Labels
	}

	var newLabels map[string]string
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

	m.nodeUpdateQueue.Add(updateItem)
}

// Run starts ExecutionNodeAffinityManager which will run in loop until `stopCh` is closed.
func (m *ExecutionNodeAffinityManager) Run(stopCh <-chan struct{}) {
	klog.V(0).Infof("Starting ExecutionNodeAffinityManager")

	for i := 0; i < UpdateWorkerSize; i++ {
		m.nodeUpdateChannels = append(m.nodeUpdateChannels, make(chan nodeUpdateItem, podUpdateChannelSize))
	}

	// Functions that are responsible for taking work items out of the workqueues and putting them
	// into channels.
	go func(stopCh <-chan struct{}) {
		for {
			item, shutdown := m.nodeUpdateQueue.Get()
			if shutdown {
				break
			}
			nodeUpdate := item.(nodeUpdateItem)
			hash := hash(nodeUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-stopCh:
				m.nodeUpdateQueue.Done(item)
				return
			case m.nodeUpdateChannels[hash] <- nodeUpdate:
				// tc.nodeUpdateQueue.Done is called by the nodeUpdateChannels worker
			}
		}
	}(stopCh)

	wg := sync.WaitGroup{}
	wg.Add(UpdateWorkerSize)
	for i := 0; i < UpdateWorkerSize; i++ {
		go m.worker(i, wg.Done, stopCh)
	}
	wg.Wait()
}

func (m *ExecutionNodeAffinityManager) worker(worker int, done func(), stopCh <-chan struct{}) {
	defer done()

	for {
		select {
		case <-stopCh:
			return
		case nodeUpdate := <-m.nodeUpdateChannels[worker]:
			m.handleNodeUpdate(nodeUpdate)
			m.nodeUpdateQueue.Done(nodeUpdate)
		}
	}
}

func (m *ExecutionNodeAffinityManager) cancelWorkWithEvent(nsName types.NamespacedName) {
	if m.affinityEvictionQueue.CancelWork(nsName.String()) {
		m.emitCancelPodDeletionEvent(nsName)
	}
}

func (m *ExecutionNodeAffinityManager) handleNodeUpdate(nodeUpdate nodeUpdateItem) {
	node, err := m.getNode(nodeUpdate.nodeName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			klog.V(4).Infof("Noticed node deletion: %#v", nodeUpdate.nodeName)
			return
		}
		utilruntime.HandleError(fmt.Errorf("cannot get node %s: %v", nodeUpdate.nodeName, err))
		return
	}

	klog.V(4).Infof("Noticed node update: %#v", nodeUpdate.nodeName)

	pods, err := m.getPodsAssignedToNode(node.Name)
	if err != nil {
		klog.Errorf(err.Error())
		return
	}
	if len(pods) == 0 {
		return
	}

	for _, pod := range pods {
		if pod.Spec.Affinity == nil || pod.Spec.Affinity.NodeAffinity == nil || pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingRequiredDuringExecution == nil {
			continue
		}

		ns, err := nodeaffinity.NewNodeSelector(pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingRequiredDuringExecution)
		if err != nil {
			klog.V(4).Infof("Noticed pod node affnity parse error : %#v", err.Error())
			continue
		}
		if !ns.Match(node) {
			klog.V(4).Infof("Pod %v evicted.", pod.Name)
			m.affinityEvictionQueue.AddWork(NewWorkArgs(pod.Name, pod.Namespace), time.Now(), time.Now())
		} else {
			m.cancelWorkWithEvent(types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name})
		}
	}
}
