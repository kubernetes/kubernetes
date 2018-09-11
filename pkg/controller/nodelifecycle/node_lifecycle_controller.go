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

// The Controller sets tainted annotations on nodes.
// Tainted nodes should not be used for new work loads and
// some effort should be given to getting existing work
// loads off of tainted nodes.

package nodelifecycle

import (
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	extensionsinformers "k8s.io/client-go/informers/extensions/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	corelisters "k8s.io/client-go/listers/core/v1"
	extensionslisters "k8s.io/client-go/listers/extensions/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/workqueue"
	v1node "k8s.io/kubernetes/pkg/api/v1/node"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle/scheduler"
	nodeutil "k8s.io/kubernetes/pkg/controller/util/node"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/system"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
)

func init() {
	// Register prometheus metrics
	Register()
}

var (
	// UnreachableTaintTemplate is the taint for when a node becomes unreachable.
	UnreachableTaintTemplate = &v1.Taint{
		Key:    algorithm.TaintNodeUnreachable,
		Effect: v1.TaintEffectNoExecute,
	}

	// NotReadyTaintTemplate is the taint for when a node is not ready for
	// executing pods
	NotReadyTaintTemplate = &v1.Taint{
		Key:    algorithm.TaintNodeNotReady,
		Effect: v1.TaintEffectNoExecute,
	}

	// map {NodeConditionType: {ConditionStatus: TaintKey}}
	// represents which NodeConditionType under which ConditionStatus should be
	// tainted with which TaintKey
	// for certain NodeConditionType, there are multiple {ConditionStatus,TaintKey} pairs
	nodeConditionToTaintKeyStatusMap = map[v1.NodeConditionType]map[v1.ConditionStatus]string{
		v1.NodeReady: {
			v1.ConditionFalse:   algorithm.TaintNodeNotReady,
			v1.ConditionUnknown: algorithm.TaintNodeUnreachable,
		},
		v1.NodeMemoryPressure: {
			v1.ConditionTrue: algorithm.TaintNodeMemoryPressure,
		},
		v1.NodeOutOfDisk: {
			v1.ConditionTrue: algorithm.TaintNodeOutOfDisk,
		},
		v1.NodeDiskPressure: {
			v1.ConditionTrue: algorithm.TaintNodeDiskPressure,
		},
		v1.NodeNetworkUnavailable: {
			v1.ConditionTrue: algorithm.TaintNodeNetworkUnavailable,
		},
		v1.NodePIDPressure: {
			v1.ConditionTrue: algorithm.TaintNodePIDPressure,
		},
	}

	taintKeyToNodeConditionMap = map[string]v1.NodeConditionType{
		algorithm.TaintNodeNotReady:           v1.NodeReady,
		algorithm.TaintNodeUnreachable:        v1.NodeReady,
		algorithm.TaintNodeNetworkUnavailable: v1.NodeNetworkUnavailable,
		algorithm.TaintNodeMemoryPressure:     v1.NodeMemoryPressure,
		algorithm.TaintNodeOutOfDisk:          v1.NodeOutOfDisk,
		algorithm.TaintNodeDiskPressure:       v1.NodeDiskPressure,
		algorithm.TaintNodePIDPressure:        v1.NodePIDPressure,
	}
)

// ZoneState is the state of a given zone.
type ZoneState string

const (
	stateInitial           = ZoneState("Initial")
	stateNormal            = ZoneState("Normal")
	stateFullDisruption    = ZoneState("FullDisruption")
	statePartialDisruption = ZoneState("PartialDisruption")
)

const (
	// The amount of time the nodecontroller should sleep between retrying NodeStatus updates
	retrySleepTime = 20 * time.Millisecond
)

type nodeStatusData struct {
	probeTimestamp           metav1.Time
	readyTransitionTimestamp metav1.Time
	status                   v1.NodeStatus
}

// Controller is the controller that manages node's life cycle.
type Controller struct {
	taintManager *scheduler.NoExecuteTaintManager

	podInformerSynced cache.InformerSynced
	cloud             cloudprovider.Interface
	kubeClient        clientset.Interface

	// This timestamp is to be used instead of LastProbeTime stored in Condition. We do this
	// to aviod the problem with time skew across the cluster.
	now func() metav1.Time

	enterPartialDisruptionFunc func(nodeNum int) float32
	enterFullDisruptionFunc    func(nodeNum int) float32
	computeZoneStateFunc       func(nodeConditions []*v1.NodeCondition) (int, ZoneState)

	knownNodeSet map[string]*v1.Node
	// per Node map storing last observed Status together with a local time when it was observed.
	nodeStatusMap map[string]nodeStatusData

	// Lock to access evictor workers
	evictorLock sync.Mutex

	// workers that evicts pods from unresponsive nodes.
	zonePodEvictor map[string]*scheduler.RateLimitedTimedQueue

	// workers that are responsible for tainting nodes.
	zoneNoExecuteTainter map[string]*scheduler.RateLimitedTimedQueue

	zoneStates map[string]ZoneState

	daemonSetStore          extensionslisters.DaemonSetLister
	daemonSetInformerSynced cache.InformerSynced

	nodeLister                  corelisters.NodeLister
	nodeInformerSynced          cache.InformerSynced
	nodeExistsInCloudProvider   func(types.NodeName) (bool, error)
	nodeShutdownInCloudProvider func(context.Context, *v1.Node) (bool, error)

	recorder record.EventRecorder

	// Value controlling Controller monitoring period, i.e. how often does Controller
	// check node status posted from kubelet. This value should be lower than nodeMonitorGracePeriod.
	// TODO: Change node status monitor to watch based.
	nodeMonitorPeriod time.Duration

	// Value used if sync_nodes_status=False, only for node startup. When node
	// is just created, e.g. cluster bootstrap or node creation, we give a longer grace period.
	nodeStartupGracePeriod time.Duration

	// Value used if sync_nodes_status=False. Controller will not proactively
	// sync node status in this case, but will monitor node status updated from kubelet. If
	// it doesn't receive update for this amount of time, it will start posting "NodeReady==
	// ConditionUnknown". The amount of time before which Controller start evicting pods
	// is controlled via flag 'pod-eviction-timeout'.
	// Note: be cautious when changing the constant, it must work with nodeStatusUpdateFrequency
	// in kubelet. There are several constraints:
	// 1. nodeMonitorGracePeriod must be N times more than nodeStatusUpdateFrequency, where
	//    N means number of retries allowed for kubelet to post node status. It is pointless
	//    to make nodeMonitorGracePeriod be less than nodeStatusUpdateFrequency, since there
	//    will only be fresh values from Kubelet at an interval of nodeStatusUpdateFrequency.
	//    The constant must be less than podEvictionTimeout.
	// 2. nodeMonitorGracePeriod can't be too large for user experience - larger value takes
	//    longer for user to see up-to-date node status.
	nodeMonitorGracePeriod time.Duration

	podEvictionTimeout          time.Duration
	evictionLimiterQPS          float32
	secondaryEvictionLimiterQPS float32
	largeClusterThreshold       int32
	unhealthyZoneThreshold      float32

	// if set to true Controller will start TaintManager that will evict Pods from
	// tainted nodes, if they're not tolerated.
	runTaintManager bool

	// if set to true Controller will taint Nodes with 'TaintNodeNotReady' and 'TaintNodeUnreachable'
	// taints instead of evicting Pods itself.
	useTaintBasedEvictions bool

	// if set to true, NodeController will taint Nodes based on its condition for 'NetworkUnavailable',
	// 'MemoryPressure', 'OutOfDisk' and 'DiskPressure'.
	taintNodeByCondition bool

	nodeUpdateQueue workqueue.Interface
}

// NewNodeLifecycleController returns a new taint controller.
func NewNodeLifecycleController(podInformer coreinformers.PodInformer,
	nodeInformer coreinformers.NodeInformer,
	daemonSetInformer extensionsinformers.DaemonSetInformer,
	cloud cloudprovider.Interface,
	kubeClient clientset.Interface,
	nodeMonitorPeriod time.Duration,
	nodeStartupGracePeriod time.Duration,
	nodeMonitorGracePeriod time.Duration,
	podEvictionTimeout time.Duration,
	evictionLimiterQPS float32,
	secondaryEvictionLimiterQPS float32,
	largeClusterThreshold int32,
	unhealthyZoneThreshold float32,
	runTaintManager bool,
	useTaintBasedEvictions bool,
	taintNodeByCondition bool) (*Controller, error) {

	if kubeClient == nil {
		glog.Fatalf("kubeClient is nil when starting Controller")
	}

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "node-controller"})

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("node_lifecycle_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}

	nc := &Controller{
		cloud:         cloud,
		kubeClient:    kubeClient,
		now:           metav1.Now,
		knownNodeSet:  make(map[string]*v1.Node),
		nodeStatusMap: make(map[string]nodeStatusData),
		nodeExistsInCloudProvider: func(nodeName types.NodeName) (bool, error) {
			return nodeutil.ExistsInCloudProvider(cloud, nodeName)
		},
		nodeShutdownInCloudProvider: func(ctx context.Context, node *v1.Node) (bool, error) {
			return nodeutil.ShutdownInCloudProvider(ctx, cloud, node)
		},
		recorder:                    recorder,
		nodeMonitorPeriod:           nodeMonitorPeriod,
		nodeStartupGracePeriod:      nodeStartupGracePeriod,
		nodeMonitorGracePeriod:      nodeMonitorGracePeriod,
		zonePodEvictor:              make(map[string]*scheduler.RateLimitedTimedQueue),
		zoneNoExecuteTainter:        make(map[string]*scheduler.RateLimitedTimedQueue),
		zoneStates:                  make(map[string]ZoneState),
		podEvictionTimeout:          podEvictionTimeout,
		evictionLimiterQPS:          evictionLimiterQPS,
		secondaryEvictionLimiterQPS: secondaryEvictionLimiterQPS,
		largeClusterThreshold:       largeClusterThreshold,
		unhealthyZoneThreshold:      unhealthyZoneThreshold,
		runTaintManager:             runTaintManager,
		useTaintBasedEvictions:      useTaintBasedEvictions && runTaintManager,
		taintNodeByCondition:        taintNodeByCondition,
		nodeUpdateQueue:             workqueue.New(),
	}
	if useTaintBasedEvictions {
		glog.Infof("Controller is using taint based evictions.")
	}

	nc.enterPartialDisruptionFunc = nc.ReducedQPSFunc
	nc.enterFullDisruptionFunc = nc.HealthyQPSFunc
	nc.computeZoneStateFunc = nc.ComputeZoneState

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod := obj.(*v1.Pod)
			if nc.taintManager != nil {
				nc.taintManager.PodUpdated(nil, pod)
			}
		},
		UpdateFunc: func(prev, obj interface{}) {
			prevPod := prev.(*v1.Pod)
			newPod := obj.(*v1.Pod)
			if nc.taintManager != nil {
				nc.taintManager.PodUpdated(prevPod, newPod)
			}
		},
		DeleteFunc: func(obj interface{}) {
			pod, isPod := obj.(*v1.Pod)
			// We can get DeletedFinalStateUnknown instead of *v1.Pod here and we need to handle that correctly.
			if !isPod {
				deletedState, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					glog.Errorf("Received unexpected object: %v", obj)
					return
				}
				pod, ok = deletedState.Obj.(*v1.Pod)
				if !ok {
					glog.Errorf("DeletedFinalStateUnknown contained non-Pod object: %v", deletedState.Obj)
					return
				}
			}
			if nc.taintManager != nil {
				nc.taintManager.PodUpdated(pod, nil)
			}
		},
	})
	nc.podInformerSynced = podInformer.Informer().HasSynced

	if nc.runTaintManager {
		nc.taintManager = scheduler.NewNoExecuteTaintManager(kubeClient)
		nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: nodeutil.CreateAddNodeHandler(func(node *v1.Node) error {
				nc.taintManager.NodeUpdated(nil, node)
				return nil
			}),
			UpdateFunc: nodeutil.CreateUpdateNodeHandler(func(oldNode, newNode *v1.Node) error {
				nc.taintManager.NodeUpdated(oldNode, newNode)
				return nil
			}),
			DeleteFunc: nodeutil.CreateDeleteNodeHandler(func(node *v1.Node) error {
				nc.taintManager.NodeUpdated(node, nil)
				return nil
			}),
		})
	}

	if nc.taintNodeByCondition {
		glog.Infof("Controller will taint node by condition.")
		nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: nodeutil.CreateAddNodeHandler(func(node *v1.Node) error {
				nc.nodeUpdateQueue.Add(node.Name)
				return nil
			}),
			UpdateFunc: nodeutil.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
				nc.nodeUpdateQueue.Add(newNode.Name)
				return nil
			}),
		})
	}

	// NOTE(resouer): nodeInformer to substitute deprecated taint key (notReady -> not-ready).
	// Remove this logic when we don't need this backwards compatibility
	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: nodeutil.CreateAddNodeHandler(func(node *v1.Node) error {
			return nc.doFixDeprecatedTaintKeyPass(node)
		}),
		UpdateFunc: nodeutil.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			return nc.doFixDeprecatedTaintKeyPass(newNode)
		}),
	})

	nc.nodeLister = nodeInformer.Lister()
	nc.nodeInformerSynced = nodeInformer.Informer().HasSynced

	nc.daemonSetStore = daemonSetInformer.Lister()
	nc.daemonSetInformerSynced = daemonSetInformer.Informer().HasSynced

	return nc, nil
}

// Run starts an asynchronous loop that monitors the status of cluster nodes.
func (nc *Controller) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Infof("Starting node controller")
	defer glog.Infof("Shutting down node controller")

	if !controller.WaitForCacheSync("taint", stopCh, nc.nodeInformerSynced, nc.podInformerSynced, nc.daemonSetInformerSynced) {
		return
	}

	if nc.runTaintManager {
		go nc.taintManager.Run(stopCh)
	}

	if nc.taintNodeByCondition {
		// Close node update queue to cleanup go routine.
		defer nc.nodeUpdateQueue.ShutDown()

		// Start workers to update NoSchedule taint for nodes.
		for i := 0; i < scheduler.UpdateWorkerSize; i++ {
			// Thanks to "workqueue", each worker just need to get item from queue, because
			// the item is flagged when got from queue: if new event come, the new item will
			// be re-queued until "Done", so no more than one worker handle the same item and
			// no event missed.
			go wait.Until(nc.doNoScheduleTaintingPassWorker, time.Second, stopCh)
		}
	}

	if nc.useTaintBasedEvictions {
		// Handling taint based evictions. Because we don't want a dedicated logic in TaintManager for NC-originated
		// taints and we normally don't rate limit evictions caused by taints, we need to rate limit adding taints.
		go wait.Until(nc.doNoExecuteTaintingPass, scheduler.NodeEvictionPeriod, stopCh)
	} else {
		// Managing eviction of nodes:
		// When we delete pods off a node, if the node was not empty at the time we then
		// queue an eviction watcher. If we hit an error, retry deletion.
		go wait.Until(nc.doEvictionPass, scheduler.NodeEvictionPeriod, stopCh)
	}

	// Incorporate the results of node status pushed from kubelet to master.
	go wait.Until(func() {
		if err := nc.monitorNodeStatus(); err != nil {
			glog.Errorf("Error monitoring node status: %v", err)
		}
	}, nc.nodeMonitorPeriod, stopCh)

	<-stopCh
}

// doFixDeprecatedTaintKeyPass checks and replaces deprecated taint key with proper key name if needed.
func (nc *Controller) doFixDeprecatedTaintKeyPass(node *v1.Node) error {
	taintsToAdd := []*v1.Taint{}
	taintsToDel := []*v1.Taint{}

	for _, taint := range node.Spec.Taints {
		if taint.Key == algorithm.DeprecatedTaintNodeNotReady {
			tDel := taint
			taintsToDel = append(taintsToDel, &tDel)

			tAdd := taint
			tAdd.Key = algorithm.TaintNodeNotReady
			taintsToAdd = append(taintsToAdd, &tAdd)
		}

		if taint.Key == algorithm.DeprecatedTaintNodeUnreachable {
			tDel := taint
			taintsToDel = append(taintsToDel, &tDel)

			tAdd := taint
			tAdd.Key = algorithm.TaintNodeUnreachable
			taintsToAdd = append(taintsToAdd, &tAdd)
		}
	}

	if len(taintsToAdd) == 0 && len(taintsToDel) == 0 {
		return nil
	}

	glog.Warningf("Detected deprecated taint keys: %v on node: %v, will substitute them with %v",
		taintsToDel, node.GetName(), taintsToAdd)

	if !nodeutil.SwapNodeControllerTaint(nc.kubeClient, taintsToAdd, taintsToDel, node) {
		return fmt.Errorf("failed to swap taints of node %+v", node)
	}
	return nil
}

func (nc *Controller) doNoScheduleTaintingPassWorker() {
	for {
		obj, shutdown := nc.nodeUpdateQueue.Get()
		// "nodeUpdateQueue" will be shutdown when "stopCh" closed;
		// we do not need to re-check "stopCh" again.
		if shutdown {
			return
		}
		nodeName := obj.(string)

		if err := nc.doNoScheduleTaintingPass(nodeName); err != nil {
			// TODO (k82cn): Add nodeName back to the queue.
			glog.Errorf("Failed to taint NoSchedule on node <%s>, requeue it: %v", nodeName, err)
		}
		nc.nodeUpdateQueue.Done(nodeName)
	}
}

func (nc *Controller) doNoScheduleTaintingPass(nodeName string) error {
	node, err := nc.nodeLister.Get(nodeName)
	if err != nil {
		// If node not found, just ignore it.
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// Map node's condition to Taints.
	var taints []v1.Taint
	for _, condition := range node.Status.Conditions {
		if taintMap, found := nodeConditionToTaintKeyStatusMap[condition.Type]; found {
			if taintKey, found := taintMap[condition.Status]; found {
				taints = append(taints, v1.Taint{
					Key:    taintKey,
					Effect: v1.TaintEffectNoSchedule,
				})
			}
		}
	}
	if node.Spec.Unschedulable {
		// If unschedulable, append related taint.
		taints = append(taints, v1.Taint{
			Key:    algorithm.TaintNodeUnschedulable,
			Effect: v1.TaintEffectNoSchedule,
		})
	}

	// Get exist taints of node.
	nodeTaints := taintutils.TaintSetFilter(node.Spec.Taints, func(t *v1.Taint) bool {
		// only NoSchedule taints are candidates to be compared with "taints" later
		if t.Effect != v1.TaintEffectNoSchedule {
			return false
		}
		// Find unschedulable taint of node.
		if t.Key == algorithm.TaintNodeUnschedulable {
			return true
		}
		// Find node condition taints of node.
		_, found := taintKeyToNodeConditionMap[t.Key]
		return found
	})
	taintsToAdd, taintsToDel := taintutils.TaintSetDiff(taints, nodeTaints)
	// If nothing to add not delete, return true directly.
	if len(taintsToAdd) == 0 && len(taintsToDel) == 0 {
		return nil
	}
	if !nodeutil.SwapNodeControllerTaint(nc.kubeClient, taintsToAdd, taintsToDel, node) {
		return fmt.Errorf("failed to swap taints of node %+v", node)
	}
	return nil
}

func (nc *Controller) doNoExecuteTaintingPass() {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	for k := range nc.zoneNoExecuteTainter {
		// Function should return 'false' and a time after which it should be retried, or 'true' if it shouldn't (it succeeded).
		nc.zoneNoExecuteTainter[k].Try(func(value scheduler.TimedValue) (bool, time.Duration) {
			node, err := nc.nodeLister.Get(value.Value)
			if apierrors.IsNotFound(err) {
				glog.Warningf("Node %v no longer present in nodeLister!", value.Value)
				return true, 0
			} else if err != nil {
				glog.Warningf("Failed to get Node %v from the nodeLister: %v", value.Value, err)
				// retry in 50 millisecond
				return false, 50 * time.Millisecond
			} else {
				zone := utilnode.GetZoneKey(node)
				evictionsNumber.WithLabelValues(zone).Inc()
			}
			_, condition := v1node.GetNodeCondition(&node.Status, v1.NodeReady)
			// Because we want to mimic NodeStatus.Condition["Ready"] we make "unreachable" and "not ready" taints mutually exclusive.
			taintToAdd := v1.Taint{}
			oppositeTaint := v1.Taint{}
			if condition.Status == v1.ConditionFalse {
				taintToAdd = *NotReadyTaintTemplate
				oppositeTaint = *UnreachableTaintTemplate
			} else if condition.Status == v1.ConditionUnknown {
				taintToAdd = *UnreachableTaintTemplate
				oppositeTaint = *NotReadyTaintTemplate
			} else {
				// It seems that the Node is ready again, so there's no need to taint it.
				glog.V(4).Infof("Node %v was in a taint queue, but it's ready now. Ignoring taint request.", value.Value)
				return true, 0
			}

			return nodeutil.SwapNodeControllerTaint(nc.kubeClient, []*v1.Taint{&taintToAdd}, []*v1.Taint{&oppositeTaint}, node), 0
		})
	}
}

func (nc *Controller) doEvictionPass() {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	for k := range nc.zonePodEvictor {
		// Function should return 'false' and a time after which it should be retried, or 'true' if it shouldn't (it succeeded).
		nc.zonePodEvictor[k].Try(func(value scheduler.TimedValue) (bool, time.Duration) {
			node, err := nc.nodeLister.Get(value.Value)
			if apierrors.IsNotFound(err) {
				glog.Warningf("Node %v no longer present in nodeLister!", value.Value)
			} else if err != nil {
				glog.Warningf("Failed to get Node %v from the nodeLister: %v", value.Value, err)
			} else {
				zone := utilnode.GetZoneKey(node)
				evictionsNumber.WithLabelValues(zone).Inc()
			}
			nodeUID, _ := value.UID.(string)
			remaining, err := nodeutil.DeletePods(nc.kubeClient, nc.recorder, value.Value, nodeUID, nc.daemonSetStore)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to evict node %q: %v", value.Value, err))
				return false, 0
			}
			if remaining {
				glog.Infof("Pods awaiting deletion due to Controller eviction")
			}
			return true, 0
		})
	}
}

// monitorNodeStatus verifies node status are constantly updated by kubelet, and if not,
// post "NodeReady==ConditionUnknown". It also evicts all pods if node is not ready or
// not reachable for a long period of time.
func (nc *Controller) monitorNodeStatus() error {
	// We are listing nodes from local cache as we can tolerate some small delays
	// comparing to state from etcd and there is eventual consistency anyway.
	nodes, err := nc.nodeLister.List(labels.Everything())
	if err != nil {
		return err
	}
	added, deleted, newZoneRepresentatives := nc.classifyNodes(nodes)

	for i := range newZoneRepresentatives {
		nc.addPodEvictorForNewZone(newZoneRepresentatives[i])
	}

	for i := range added {
		glog.V(1).Infof("Controller observed a new Node: %#v", added[i].Name)
		nodeutil.RecordNodeEvent(nc.recorder, added[i].Name, string(added[i].UID), v1.EventTypeNormal, "RegisteredNode", fmt.Sprintf("Registered Node %v in Controller", added[i].Name))
		nc.knownNodeSet[added[i].Name] = added[i]
		nc.addPodEvictorForNewZone(added[i])
		if nc.useTaintBasedEvictions {
			nc.markNodeAsReachable(added[i])
		} else {
			nc.cancelPodEviction(added[i])
		}
	}

	for i := range deleted {
		glog.V(1).Infof("Controller observed a Node deletion: %v", deleted[i].Name)
		nodeutil.RecordNodeEvent(nc.recorder, deleted[i].Name, string(deleted[i].UID), v1.EventTypeNormal, "RemovingNode", fmt.Sprintf("Removing Node %v from Controller", deleted[i].Name))
		delete(nc.knownNodeSet, deleted[i].Name)
	}

	zoneToNodeConditions := map[string][]*v1.NodeCondition{}
	for i := range nodes {
		var gracePeriod time.Duration
		var observedReadyCondition v1.NodeCondition
		var currentReadyCondition *v1.NodeCondition
		node := nodes[i].DeepCopy()
		if err := wait.PollImmediate(retrySleepTime, retrySleepTime*scheduler.NodeStatusUpdateRetry, func() (bool, error) {
			gracePeriod, observedReadyCondition, currentReadyCondition, err = nc.tryUpdateNodeStatus(node)
			if err == nil {
				return true, nil
			}
			name := node.Name
			node, err = nc.kubeClient.CoreV1().Nodes().Get(name, metav1.GetOptions{})
			if err != nil {
				glog.Errorf("Failed while getting a Node to retry updating NodeStatus. Probably Node %s was deleted.", name)
				return false, err
			}
			return false, nil
		}); err != nil {
			glog.Errorf("Update status of Node '%v' from Controller error: %v. "+
				"Skipping - no pods will be evicted.", node.Name, err)
			continue
		}

		// We do not treat a master node as a part of the cluster for network disruption checking.
		if !system.IsMasterNode(node.Name) {
			zoneToNodeConditions[utilnode.GetZoneKey(node)] = append(zoneToNodeConditions[utilnode.GetZoneKey(node)], currentReadyCondition)
		}

		decisionTimestamp := nc.now()
		if currentReadyCondition != nil {
			// Check eviction timeout against decisionTimestamp
			if observedReadyCondition.Status == v1.ConditionFalse {
				if nc.useTaintBasedEvictions {
					// We want to update the taint straight away if Node is already tainted with the UnreachableTaint
					if taintutils.TaintExists(node.Spec.Taints, UnreachableTaintTemplate) {
						taintToAdd := *NotReadyTaintTemplate
						if !nodeutil.SwapNodeControllerTaint(nc.kubeClient, []*v1.Taint{&taintToAdd}, []*v1.Taint{UnreachableTaintTemplate}, node) {
							glog.Errorf("Failed to instantly swap UnreachableTaint to NotReadyTaint. Will try again in the next cycle.")
						}
					} else if nc.markNodeForTainting(node) {
						glog.V(2).Infof("Node %v is NotReady as of %v. Adding it to the Taint queue.",
							node.Name,
							decisionTimestamp,
						)
					}
				} else {
					if decisionTimestamp.After(nc.nodeStatusMap[node.Name].readyTransitionTimestamp.Add(nc.podEvictionTimeout)) {
						if nc.evictPods(node) {
							glog.V(2).Infof("Node is NotReady. Adding Pods on Node %s to eviction queue: %v is later than %v + %v",
								node.Name,
								decisionTimestamp,
								nc.nodeStatusMap[node.Name].readyTransitionTimestamp,
								nc.podEvictionTimeout,
							)
						}
					}
				}
			}
			if observedReadyCondition.Status == v1.ConditionUnknown {
				if nc.useTaintBasedEvictions {
					// We want to update the taint straight away if Node is already tainted with the UnreachableTaint
					if taintutils.TaintExists(node.Spec.Taints, NotReadyTaintTemplate) {
						taintToAdd := *UnreachableTaintTemplate
						if !nodeutil.SwapNodeControllerTaint(nc.kubeClient, []*v1.Taint{&taintToAdd}, []*v1.Taint{NotReadyTaintTemplate}, node) {
							glog.Errorf("Failed to instantly swap UnreachableTaint to NotReadyTaint. Will try again in the next cycle.")
						}
					} else if nc.markNodeForTainting(node) {
						glog.V(2).Infof("Node %v is unresponsive as of %v. Adding it to the Taint queue.",
							node.Name,
							decisionTimestamp,
						)
					}
				} else {
					if decisionTimestamp.After(nc.nodeStatusMap[node.Name].probeTimestamp.Add(nc.podEvictionTimeout)) {
						if nc.evictPods(node) {
							glog.V(2).Infof("Node is unresponsive. Adding Pods on Node %s to eviction queues: %v is later than %v + %v",
								node.Name,
								decisionTimestamp,
								nc.nodeStatusMap[node.Name].readyTransitionTimestamp,
								nc.podEvictionTimeout-gracePeriod,
							)
						}
					}
				}
			}
			if observedReadyCondition.Status == v1.ConditionTrue {
				if nc.useTaintBasedEvictions {
					removed, err := nc.markNodeAsReachable(node)
					if err != nil {
						glog.Errorf("Failed to remove taints from node %v. Will retry in next iteration.", node.Name)
					}
					if removed {
						glog.V(2).Infof("Node %s is healthy again, removing all taints", node.Name)
					}
				} else {
					if nc.cancelPodEviction(node) {
						glog.V(2).Infof("Node %s is ready again, cancelled pod eviction", node.Name)
					}
				}
				// remove shutdown taint this is needed always depending do we use taintbased or not
				err := nc.markNodeAsNotShutdown(node)
				if err != nil {
					glog.Errorf("Failed to remove taints from node %v. Will retry in next iteration.", node.Name)
				}
			}

			// Report node event.
			if currentReadyCondition.Status != v1.ConditionTrue && observedReadyCondition.Status == v1.ConditionTrue {
				nodeutil.RecordNodeStatusChange(nc.recorder, node, "NodeNotReady")
				if err = nodeutil.MarkAllPodsNotReady(nc.kubeClient, node); err != nil {
					utilruntime.HandleError(fmt.Errorf("Unable to mark all pods NotReady on node %v: %v", node.Name, err))
				}
			}

			// Check with the cloud provider to see if the node still exists. If it
			// doesn't, delete the node immediately.
			if currentReadyCondition.Status != v1.ConditionTrue && nc.cloud != nil {
				// check is node shutdowned, if yes do not deleted it. Instead add taint
				shutdown, err := nc.nodeShutdownInCloudProvider(context.TODO(), node)
				if err != nil {
					glog.Errorf("Error determining if node %v shutdown in cloud: %v", node.Name, err)
				}
				// node shutdown
				if shutdown && err == nil {
					err = controller.AddOrUpdateTaintOnNode(nc.kubeClient, node.Name, controller.ShutdownTaint)
					if err != nil {
						glog.Errorf("Error patching node taints: %v", err)
					}
					continue
				}
				exists, err := nc.nodeExistsInCloudProvider(types.NodeName(node.Name))
				if err != nil {
					glog.Errorf("Error determining if node %v exists in cloud: %v", node.Name, err)
					continue
				}
				if !exists {
					glog.V(2).Infof("Deleting node (no longer present in cloud provider): %s", node.Name)
					nodeutil.RecordNodeEvent(nc.recorder, node.Name, string(node.UID), v1.EventTypeNormal, "DeletingNode", fmt.Sprintf("Deleting Node %v because it's not present according to cloud provider", node.Name))
					go func(nodeName string) {
						defer utilruntime.HandleCrash()
						// Kubelet is not reporting and Cloud Provider says node
						// is gone. Delete it without worrying about grace
						// periods.
						if err := nodeutil.ForcefullyDeleteNode(nc.kubeClient, nodeName); err != nil {
							glog.Errorf("Unable to forcefully delete node %q: %v", nodeName, err)
						}
					}(node.Name)
				}
			}
		}
	}
	nc.handleDisruption(zoneToNodeConditions, nodes)

	return nil
}

// tryUpdateNodeStatus checks a given node's conditions and tries to update it. Returns grace period to
// which given node is entitled, state of current and last observed Ready Condition, and an error if it occurred.
func (nc *Controller) tryUpdateNodeStatus(node *v1.Node) (time.Duration, v1.NodeCondition, *v1.NodeCondition, error) {
	var err error
	var gracePeriod time.Duration
	var observedReadyCondition v1.NodeCondition
	_, currentReadyCondition := v1node.GetNodeCondition(&node.Status, v1.NodeReady)
	if currentReadyCondition == nil {
		// If ready condition is nil, then kubelet (or nodecontroller) never posted node status.
		// A fake ready condition is created, where LastProbeTime and LastTransitionTime is set
		// to node.CreationTimestamp to avoid handle the corner case.
		observedReadyCondition = v1.NodeCondition{
			Type:               v1.NodeReady,
			Status:             v1.ConditionUnknown,
			LastHeartbeatTime:  node.CreationTimestamp,
			LastTransitionTime: node.CreationTimestamp,
		}
		gracePeriod = nc.nodeStartupGracePeriod
		nc.nodeStatusMap[node.Name] = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           node.CreationTimestamp,
			readyTransitionTimestamp: node.CreationTimestamp,
		}
	} else {
		// If ready condition is not nil, make a copy of it, since we may modify it in place later.
		observedReadyCondition = *currentReadyCondition
		gracePeriod = nc.nodeMonitorGracePeriod
	}

	savedNodeStatus, found := nc.nodeStatusMap[node.Name]
	// There are following cases to check:
	// - both saved and new status have no Ready Condition set - we leave everything as it is,
	// - saved status have no Ready Condition, but current one does - Controller was restarted with Node data already present in etcd,
	// - saved status have some Ready Condition, but current one does not - it's an error, but we fill it up because that's probably a good thing to do,
	// - both saved and current statuses have Ready Conditions and they have the same LastProbeTime - nothing happened on that Node, it may be
	//   unresponsive, so we leave it as it is,
	// - both saved and current statuses have Ready Conditions, they have different LastProbeTimes, but the same Ready Condition State -
	//   everything's in order, no transition occurred, we update only probeTimestamp,
	// - both saved and current statuses have Ready Conditions, different LastProbeTimes and different Ready Condition State -
	//   Ready Condition changed it state since we last seen it, so we update both probeTimestamp and readyTransitionTimestamp.
	// TODO: things to consider:
	//   - if 'LastProbeTime' have gone back in time its probably an error, currently we ignore it,
	//   - currently only correct Ready State transition outside of Node Controller is marking it ready by Kubelet, we don't check
	//     if that's the case, but it does not seem necessary.
	var savedCondition *v1.NodeCondition
	if found {
		_, savedCondition = v1node.GetNodeCondition(&savedNodeStatus.status, v1.NodeReady)
	}
	_, observedCondition := v1node.GetNodeCondition(&node.Status, v1.NodeReady)
	if !found {
		glog.Warningf("Missing timestamp for Node %s. Assuming now as a timestamp.", node.Name)
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: nc.now(),
		}
	} else if savedCondition == nil && observedCondition != nil {
		glog.V(1).Infof("Creating timestamp entry for newly observed Node %s", node.Name)
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: nc.now(),
		}
	} else if savedCondition != nil && observedCondition == nil {
		glog.Errorf("ReadyCondition was removed from Status of Node %s", node.Name)
		// TODO: figure out what to do in this case. For now we do the same thing as above.
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: nc.now(),
		}
	} else if savedCondition != nil && observedCondition != nil && savedCondition.LastHeartbeatTime != observedCondition.LastHeartbeatTime {
		var transitionTime metav1.Time
		// If ReadyCondition changed since the last time we checked, we update the transition timestamp to "now",
		// otherwise we leave it as it is.
		if savedCondition.LastTransitionTime != observedCondition.LastTransitionTime {
			glog.V(3).Infof("ReadyCondition for Node %s transitioned from %v to %v", node.Name, savedCondition, observedCondition)
			transitionTime = nc.now()
		} else {
			transitionTime = savedNodeStatus.readyTransitionTimestamp
		}
		if glog.V(5) {
			glog.V(5).Infof("Node %s ReadyCondition updated. Updating timestamp: %+v vs %+v.", node.Name, savedNodeStatus.status, node.Status)
		} else {
			glog.V(3).Infof("Node %s ReadyCondition updated. Updating timestamp.", node.Name)
		}
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: transitionTime,
		}
	}
	nc.nodeStatusMap[node.Name] = savedNodeStatus

	if nc.now().After(savedNodeStatus.probeTimestamp.Add(gracePeriod)) {
		// NodeReady condition was last set longer ago than gracePeriod, so update it to Unknown
		// (regardless of its current value) in the master.
		if currentReadyCondition == nil {
			glog.V(2).Infof("node %v is never updated by kubelet", node.Name)
			node.Status.Conditions = append(node.Status.Conditions, v1.NodeCondition{
				Type:               v1.NodeReady,
				Status:             v1.ConditionUnknown,
				Reason:             "NodeStatusNeverUpdated",
				Message:            fmt.Sprintf("Kubelet never posted node status."),
				LastHeartbeatTime:  node.CreationTimestamp,
				LastTransitionTime: nc.now(),
			})
		} else {
			glog.V(4).Infof("node %v hasn't been updated for %+v. Last ready condition is: %+v",
				node.Name, nc.now().Time.Sub(savedNodeStatus.probeTimestamp.Time), observedReadyCondition)
			if observedReadyCondition.Status != v1.ConditionUnknown {
				currentReadyCondition.Status = v1.ConditionUnknown
				currentReadyCondition.Reason = "NodeStatusUnknown"
				currentReadyCondition.Message = "Kubelet stopped posting node status."
				// LastProbeTime is the last time we heard from kubelet.
				currentReadyCondition.LastHeartbeatTime = observedReadyCondition.LastHeartbeatTime
				currentReadyCondition.LastTransitionTime = nc.now()
			}
		}

		// remaining node conditions should also be set to Unknown
		remainingNodeConditionTypes := []v1.NodeConditionType{
			v1.NodeOutOfDisk,
			v1.NodeMemoryPressure,
			v1.NodeDiskPressure,
			// We don't change 'NodeNetworkUnavailable' condition, as it's managed on a control plane level.
			// v1.NodeNetworkUnavailable,
		}

		nowTimestamp := nc.now()
		for _, nodeConditionType := range remainingNodeConditionTypes {
			_, currentCondition := v1node.GetNodeCondition(&node.Status, nodeConditionType)
			if currentCondition == nil {
				glog.V(2).Infof("Condition %v of node %v was never updated by kubelet", nodeConditionType, node.Name)
				node.Status.Conditions = append(node.Status.Conditions, v1.NodeCondition{
					Type:               nodeConditionType,
					Status:             v1.ConditionUnknown,
					Reason:             "NodeStatusNeverUpdated",
					Message:            "Kubelet never posted node status.",
					LastHeartbeatTime:  node.CreationTimestamp,
					LastTransitionTime: nowTimestamp,
				})
			} else {
				glog.V(4).Infof("node %v hasn't been updated for %+v. Last %v is: %+v",
					node.Name, nc.now().Time.Sub(savedNodeStatus.probeTimestamp.Time), nodeConditionType, currentCondition)
				if currentCondition.Status != v1.ConditionUnknown {
					currentCondition.Status = v1.ConditionUnknown
					currentCondition.Reason = "NodeStatusUnknown"
					currentCondition.Message = "Kubelet stopped posting node status."
					currentCondition.LastTransitionTime = nowTimestamp
				}
			}
		}

		_, currentCondition := v1node.GetNodeCondition(&node.Status, v1.NodeReady)
		if !apiequality.Semantic.DeepEqual(currentCondition, &observedReadyCondition) {
			if _, err = nc.kubeClient.CoreV1().Nodes().UpdateStatus(node); err != nil {
				glog.Errorf("Error updating node %s: %v", node.Name, err)
				return gracePeriod, observedReadyCondition, currentReadyCondition, err
			}
			nc.nodeStatusMap[node.Name] = nodeStatusData{
				status:                   node.Status,
				probeTimestamp:           nc.nodeStatusMap[node.Name].probeTimestamp,
				readyTransitionTimestamp: nc.now(),
			}
			return gracePeriod, observedReadyCondition, currentReadyCondition, nil
		}
	}

	return gracePeriod, observedReadyCondition, currentReadyCondition, err
}

func (nc *Controller) handleDisruption(zoneToNodeConditions map[string][]*v1.NodeCondition, nodes []*v1.Node) {
	newZoneStates := map[string]ZoneState{}
	allAreFullyDisrupted := true
	for k, v := range zoneToNodeConditions {
		zoneSize.WithLabelValues(k).Set(float64(len(v)))
		unhealthy, newState := nc.computeZoneStateFunc(v)
		zoneHealth.WithLabelValues(k).Set(float64(100*(len(v)-unhealthy)) / float64(len(v)))
		unhealthyNodes.WithLabelValues(k).Set(float64(unhealthy))
		if newState != stateFullDisruption {
			allAreFullyDisrupted = false
		}
		newZoneStates[k] = newState
		if _, had := nc.zoneStates[k]; !had {
			glog.Errorf("Setting initial state for unseen zone: %v", k)
			nc.zoneStates[k] = stateInitial
		}
	}

	allWasFullyDisrupted := true
	for k, v := range nc.zoneStates {
		if _, have := zoneToNodeConditions[k]; !have {
			zoneSize.WithLabelValues(k).Set(0)
			zoneHealth.WithLabelValues(k).Set(100)
			unhealthyNodes.WithLabelValues(k).Set(0)
			delete(nc.zoneStates, k)
			continue
		}
		if v != stateFullDisruption {
			allWasFullyDisrupted = false
			break
		}
	}

	// At least one node was responding in previous pass or in the current pass. Semantics is as follows:
	// - if the new state is "partialDisruption" we call a user defined function that returns a new limiter to use,
	// - if the new state is "normal" we resume normal operation (go back to default limiter settings),
	// - if new state is "fullDisruption" we restore normal eviction rate,
	//   - unless all zones in the cluster are in "fullDisruption" - in that case we stop all evictions.
	if !allAreFullyDisrupted || !allWasFullyDisrupted {
		// We're switching to full disruption mode
		if allAreFullyDisrupted {
			glog.V(0).Info("Controller detected that all Nodes are not-Ready. Entering master disruption mode.")
			for i := range nodes {
				if nc.useTaintBasedEvictions {
					_, err := nc.markNodeAsReachable(nodes[i])
					if err != nil {
						glog.Errorf("Failed to remove taints from Node %v", nodes[i].Name)
					}
				} else {
					nc.cancelPodEviction(nodes[i])
				}
			}
			// We stop all evictions.
			for k := range nc.zoneStates {
				if nc.useTaintBasedEvictions {
					nc.zoneNoExecuteTainter[k].SwapLimiter(0)
				} else {
					nc.zonePodEvictor[k].SwapLimiter(0)
				}
			}
			for k := range nc.zoneStates {
				nc.zoneStates[k] = stateFullDisruption
			}
			// All rate limiters are updated, so we can return early here.
			return
		}
		// We're exiting full disruption mode
		if allWasFullyDisrupted {
			glog.V(0).Info("Controller detected that some Nodes are Ready. Exiting master disruption mode.")
			// When exiting disruption mode update probe timestamps on all Nodes.
			now := nc.now()
			for i := range nodes {
				v := nc.nodeStatusMap[nodes[i].Name]
				v.probeTimestamp = now
				v.readyTransitionTimestamp = now
				nc.nodeStatusMap[nodes[i].Name] = v
			}
			// We reset all rate limiters to settings appropriate for the given state.
			for k := range nc.zoneStates {
				nc.setLimiterInZone(k, len(zoneToNodeConditions[k]), newZoneStates[k])
				nc.zoneStates[k] = newZoneStates[k]
			}
			return
		}
		// We know that there's at least one not-fully disrupted so,
		// we can use default behavior for rate limiters
		for k, v := range nc.zoneStates {
			newState := newZoneStates[k]
			if v == newState {
				continue
			}
			glog.V(0).Infof("Controller detected that zone %v is now in state %v.", k, newState)
			nc.setLimiterInZone(k, len(zoneToNodeConditions[k]), newState)
			nc.zoneStates[k] = newState
		}
	}
}

func (nc *Controller) setLimiterInZone(zone string, zoneSize int, state ZoneState) {
	switch state {
	case stateNormal:
		if nc.useTaintBasedEvictions {
			nc.zoneNoExecuteTainter[zone].SwapLimiter(nc.evictionLimiterQPS)
		} else {
			nc.zonePodEvictor[zone].SwapLimiter(nc.evictionLimiterQPS)
		}
	case statePartialDisruption:
		if nc.useTaintBasedEvictions {
			nc.zoneNoExecuteTainter[zone].SwapLimiter(
				nc.enterPartialDisruptionFunc(zoneSize))
		} else {
			nc.zonePodEvictor[zone].SwapLimiter(
				nc.enterPartialDisruptionFunc(zoneSize))
		}
	case stateFullDisruption:
		if nc.useTaintBasedEvictions {
			nc.zoneNoExecuteTainter[zone].SwapLimiter(
				nc.enterFullDisruptionFunc(zoneSize))
		} else {
			nc.zonePodEvictor[zone].SwapLimiter(
				nc.enterFullDisruptionFunc(zoneSize))
		}
	}
}

// classifyNodes classifies the allNodes to three categories:
//   1. added: the nodes that in 'allNodes', but not in 'knownNodeSet'
//   2. deleted: the nodes that in 'knownNodeSet', but not in 'allNodes'
//   3. newZoneRepresentatives: the nodes that in both 'knownNodeSet' and 'allNodes', but no zone states
func (nc *Controller) classifyNodes(allNodes []*v1.Node) (added, deleted, newZoneRepresentatives []*v1.Node) {
	for i := range allNodes {
		if _, has := nc.knownNodeSet[allNodes[i].Name]; !has {
			added = append(added, allNodes[i])
		} else {
			// Currently, we only consider new zone as updated.
			zone := utilnode.GetZoneKey(allNodes[i])
			if _, found := nc.zoneStates[zone]; !found {
				newZoneRepresentatives = append(newZoneRepresentatives, allNodes[i])
			}
		}
	}

	// If there's a difference between lengths of known Nodes and observed nodes
	// we must have removed some Node.
	if len(nc.knownNodeSet)+len(added) != len(allNodes) {
		knowSetCopy := map[string]*v1.Node{}
		for k, v := range nc.knownNodeSet {
			knowSetCopy[k] = v
		}
		for i := range allNodes {
			delete(knowSetCopy, allNodes[i].Name)
		}
		for i := range knowSetCopy {
			deleted = append(deleted, knowSetCopy[i])
		}
	}
	return
}

// HealthyQPSFunc returns the default value for cluster eviction rate - we take
// nodeNum for consistency with ReducedQPSFunc.
func (nc *Controller) HealthyQPSFunc(nodeNum int) float32 {
	return nc.evictionLimiterQPS
}

// ReducedQPSFunc returns the QPS for when a the cluster is large make
// evictions slower, if they're small stop evictions altogether.
func (nc *Controller) ReducedQPSFunc(nodeNum int) float32 {
	if int32(nodeNum) > nc.largeClusterThreshold {
		return nc.secondaryEvictionLimiterQPS
	}
	return 0
}

// addPodEvictorForNewZone checks if new zone appeared, and if so add new evictor.
func (nc *Controller) addPodEvictorForNewZone(node *v1.Node) {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	zone := utilnode.GetZoneKey(node)
	if _, found := nc.zoneStates[zone]; !found {
		nc.zoneStates[zone] = stateInitial
		if !nc.useTaintBasedEvictions {
			nc.zonePodEvictor[zone] =
				scheduler.NewRateLimitedTimedQueue(
					flowcontrol.NewTokenBucketRateLimiter(nc.evictionLimiterQPS, scheduler.EvictionRateLimiterBurst))
		} else {
			nc.zoneNoExecuteTainter[zone] =
				scheduler.NewRateLimitedTimedQueue(
					flowcontrol.NewTokenBucketRateLimiter(nc.evictionLimiterQPS, scheduler.EvictionRateLimiterBurst))
		}
		// Init the metric for the new zone.
		glog.Infof("Initializing eviction metric for zone: %v", zone)
		evictionsNumber.WithLabelValues(zone).Add(0)
	}
}

// cancelPodEviction removes any queued evictions, typically because the node is available again. It
// returns true if an eviction was queued.
func (nc *Controller) cancelPodEviction(node *v1.Node) bool {
	zone := utilnode.GetZoneKey(node)
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	wasDeleting := nc.zonePodEvictor[zone].Remove(node.Name)
	if wasDeleting {
		glog.V(2).Infof("Cancelling pod Eviction on Node: %v", node.Name)
		return true
	}
	return false
}

// evictPods queues an eviction for the provided node name, and returns false if the node is already
// queued for eviction.
func (nc *Controller) evictPods(node *v1.Node) bool {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	return nc.zonePodEvictor[utilnode.GetZoneKey(node)].Add(node.Name, string(node.UID))
}

func (nc *Controller) markNodeForTainting(node *v1.Node) bool {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	return nc.zoneNoExecuteTainter[utilnode.GetZoneKey(node)].Add(node.Name, string(node.UID))
}

func (nc *Controller) markNodeAsReachable(node *v1.Node) (bool, error) {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	err := controller.RemoveTaintOffNode(nc.kubeClient, node.Name, node, UnreachableTaintTemplate)
	if err != nil {
		glog.Errorf("Failed to remove taint from node %v: %v", node.Name, err)
		return false, err
	}
	err = controller.RemoveTaintOffNode(nc.kubeClient, node.Name, node, NotReadyTaintTemplate)
	if err != nil {
		glog.Errorf("Failed to remove taint from node %v: %v", node.Name, err)
		return false, err
	}
	return nc.zoneNoExecuteTainter[utilnode.GetZoneKey(node)].Remove(node.Name), nil
}

func (nc *Controller) markNodeAsNotShutdown(node *v1.Node) error {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	err := controller.RemoveTaintOffNode(nc.kubeClient, node.Name, node, controller.ShutdownTaint)
	if err != nil {
		glog.Errorf("Failed to remove taint from node %v: %v", node.Name, err)
		return err
	}
	return nil
}

// ComputeZoneState returns a slice of NodeReadyConditions for all Nodes in a given zone.
// The zone is considered:
// - fullyDisrupted if there're no Ready Nodes,
// - partiallyDisrupted if at least than nc.unhealthyZoneThreshold percent of Nodes are not Ready,
// - normal otherwise
func (nc *Controller) ComputeZoneState(nodeReadyConditions []*v1.NodeCondition) (int, ZoneState) {
	readyNodes := 0
	notReadyNodes := 0
	for i := range nodeReadyConditions {
		if nodeReadyConditions[i] != nil && nodeReadyConditions[i].Status == v1.ConditionTrue {
			readyNodes++
		} else {
			notReadyNodes++
		}
	}
	switch {
	case readyNodes == 0 && notReadyNodes > 0:
		return notReadyNodes, stateFullDisruption
	case notReadyNodes > 2 && float32(notReadyNodes)/float32(notReadyNodes+readyNodes) >= nc.unhealthyZoneThreshold:
		return notReadyNodes, statePartialDisruption
	default:
		return notReadyNodes, stateNormal
	}
}

func hash(val string, max int) int {
	hasher := fnv.New32a()
	io.WriteString(hasher, val)
	return int(hasher.Sum32()) % max
}
