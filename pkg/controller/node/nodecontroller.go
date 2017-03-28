/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"

	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	extensionsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/extensions/v1beta1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	extensionslisters "k8s.io/kubernetes/pkg/client/listers/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/system"
	utilversion "k8s.io/kubernetes/pkg/util/version"

	"github.com/golang/glog"
)

func init() {
	// Register prometheus metrics
	Register()
}

var (
	ErrCloudInstance        = errors.New("cloud provider doesn't support instances.")
	gracefulDeletionVersion = utilversion.MustParseSemantic("v1.1.0")

	// The minimum kubelet version for which the nodecontroller
	// can safely flip pod.Status to NotReady.
	podStatusReconciliationVersion = utilversion.MustParseSemantic("v1.2.0")

	UnreachableTaintTemplate = &v1.Taint{
		Key:    metav1.TaintNodeUnreachable,
		Effect: v1.TaintEffectNoExecute,
	}

	NotReadyTaintTemplate = &v1.Taint{
		Key:    metav1.TaintNodeNotReady,
		Effect: v1.TaintEffectNoExecute,
	}
)

const (
	// nodeStatusUpdateRetry controls the number of retries of writing NodeStatus update.
	nodeStatusUpdateRetry = 5
	// controls how often NodeController will try to evict Pods from non-responsive Nodes.
	nodeEvictionPeriod = 100 * time.Millisecond
	// Burst value for all eviction rate limiters
	evictionRateLimiterBurst = 1
	// The amount of time the nodecontroller polls on the list nodes endpoint.
	apiserverStartupGracePeriod = 10 * time.Minute
	// The amount of time the nodecontroller should sleep between retrying NodeStatus updates
	retrySleepTime = 20 * time.Millisecond
)

type zoneState string

const (
	stateInitial           = zoneState("Initial")
	stateNormal            = zoneState("Normal")
	stateFullDisruption    = zoneState("FullDisruption")
	statePartialDisruption = zoneState("PartialDisruption")
)

type nodeStatusData struct {
	probeTimestamp           metav1.Time
	readyTransitionTimestamp metav1.Time
	status                   v1.NodeStatus
}

type NodeController struct {
	allocateNodeCIDRs bool
	cloud             cloudprovider.Interface
	clusterCIDR       *net.IPNet
	serviceCIDR       *net.IPNet
	knownNodeSet      map[string]*v1.Node
	kubeClient        clientset.Interface
	// Method for easy mocking in unittest.
	lookupIP func(host string) ([]net.IP, error)
	// Value used if sync_nodes_status=False. NodeController will not proactively
	// sync node status in this case, but will monitor node status updated from kubelet. If
	// it doesn't receive update for this amount of time, it will start posting "NodeReady==
	// ConditionUnknown". The amount of time before which NodeController start evicting pods
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
	// Value controlling NodeController monitoring period, i.e. how often does NodeController
	// check node status posted from kubelet. This value should be lower than nodeMonitorGracePeriod.
	// TODO: Change node status monitor to watch based.
	nodeMonitorPeriod time.Duration
	// Value used if sync_nodes_status=False, only for node startup. When node
	// is just created, e.g. cluster bootstrap or node creation, we give a longer grace period.
	nodeStartupGracePeriod time.Duration
	// per Node map storing last observed Status together with a local time when it was observed.
	// This timestamp is to be used instead of LastProbeTime stored in Condition. We do this
	// to aviod the problem with time skew across the cluster.
	nodeStatusMap map[string]nodeStatusData
	now           func() metav1.Time
	// Lock to access evictor workers
	evictorLock sync.Mutex
	// workers that evicts pods from unresponsive nodes.
	zonePodEvictor map[string]*RateLimitedTimedQueue
	// workers that are responsible for tainting nodes.
	zoneNotReadyOrUnreachableTainer map[string]*RateLimitedTimedQueue
	podEvictionTimeout              time.Duration
	// The maximum duration before a pod evicted from a node can be forcefully terminated.
	maximumGracePeriod time.Duration
	recorder           record.EventRecorder

	nodeLister         corelisters.NodeLister
	nodeInformerSynced cache.InformerSynced

	daemonSetStore          extensionslisters.DaemonSetLister
	daemonSetInformerSynced cache.InformerSynced

	podInformerSynced cache.InformerSynced

	// allocate/recycle CIDRs for node if allocateNodeCIDRs == true
	cidrAllocator CIDRAllocator
	// manages taints
	taintManager *NoExecuteTaintManager

	forcefullyDeletePod        func(*v1.Pod) error
	nodeExistsInCloudProvider  func(types.NodeName) (bool, error)
	computeZoneStateFunc       func(nodeConditions []*v1.NodeCondition) (int, zoneState)
	enterPartialDisruptionFunc func(nodeNum int) float32
	enterFullDisruptionFunc    func(nodeNum int) float32

	zoneStates                  map[string]zoneState
	evictionLimiterQPS          float32
	secondaryEvictionLimiterQPS float32
	largeClusterThreshold       int32
	unhealthyZoneThreshold      float32

	// if set to true NodeController will start TaintManager that will evict Pods from
	// tainted nodes, if they're not tolerated.
	runTaintManager bool

	// if set to true NodeController will taint Nodes with 'TaintNodeNotReady' and 'TaintNodeUnreachable'
	// taints instead of evicting Pods itself.
	useTaintBasedEvictions bool
}

// NewNodeController returns a new node controller to sync instances from cloudprovider.
// This method returns an error if it is unable to initialize the CIDR bitmap with
// podCIDRs it has already allocated to nodes. Since we don't allow podCIDR changes
// currently, this should be handled as a fatal error.
func NewNodeController(
	podInformer coreinformers.PodInformer,
	nodeInformer coreinformers.NodeInformer,
	daemonSetInformer extensionsinformers.DaemonSetInformer,
	cloud cloudprovider.Interface,
	kubeClient clientset.Interface,
	podEvictionTimeout time.Duration,
	evictionLimiterQPS float32,
	secondaryEvictionLimiterQPS float32,
	largeClusterThreshold int32,
	unhealthyZoneThreshold float32,
	nodeMonitorGracePeriod time.Duration,
	nodeStartupGracePeriod time.Duration,
	nodeMonitorPeriod time.Duration,
	clusterCIDR *net.IPNet,
	serviceCIDR *net.IPNet,
	nodeCIDRMaskSize int,
	allocateNodeCIDRs bool,
	runTaintManager bool,
	useTaintBasedEvictions bool) (*NodeController, error) {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "controllermanager"})
	eventBroadcaster.StartLogging(glog.Infof)
	if kubeClient != nil {
		glog.V(0).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.Core().RESTClient()).Events("")})
	} else {
		glog.Fatalf("kubeClient is nil when starting NodeController")
	}

	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("node_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}

	if allocateNodeCIDRs {
		if clusterCIDR == nil {
			glog.Fatal("NodeController: Must specify clusterCIDR if allocateNodeCIDRs == true.")
		}
		mask := clusterCIDR.Mask
		if maskSize, _ := mask.Size(); maskSize > nodeCIDRMaskSize {
			glog.Fatal("NodeController: Invalid clusterCIDR, mask size of clusterCIDR must be less than nodeCIDRMaskSize.")
		}
	}

	nc := &NodeController{
		cloud:                           cloud,
		knownNodeSet:                    make(map[string]*v1.Node),
		kubeClient:                      kubeClient,
		recorder:                        recorder,
		podEvictionTimeout:              podEvictionTimeout,
		maximumGracePeriod:              5 * time.Minute,
		zonePodEvictor:                  make(map[string]*RateLimitedTimedQueue),
		zoneNotReadyOrUnreachableTainer: make(map[string]*RateLimitedTimedQueue),
		nodeStatusMap:                   make(map[string]nodeStatusData),
		nodeMonitorGracePeriod:          nodeMonitorGracePeriod,
		nodeMonitorPeriod:               nodeMonitorPeriod,
		nodeStartupGracePeriod:          nodeStartupGracePeriod,
		lookupIP:                        net.LookupIP,
		now:                             metav1.Now,
		clusterCIDR:                     clusterCIDR,
		serviceCIDR:                     serviceCIDR,
		allocateNodeCIDRs:               allocateNodeCIDRs,
		forcefullyDeletePod:             func(p *v1.Pod) error { return forcefullyDeletePod(kubeClient, p) },
		nodeExistsInCloudProvider:       func(nodeName types.NodeName) (bool, error) { return nodeExistsInCloudProvider(cloud, nodeName) },
		evictionLimiterQPS:              evictionLimiterQPS,
		secondaryEvictionLimiterQPS:     secondaryEvictionLimiterQPS,
		largeClusterThreshold:           largeClusterThreshold,
		unhealthyZoneThreshold:          unhealthyZoneThreshold,
		zoneStates:                      make(map[string]zoneState),
		runTaintManager:                 runTaintManager,
		useTaintBasedEvictions:          useTaintBasedEvictions && runTaintManager,
	}
	if useTaintBasedEvictions {
		glog.Infof("NodeController is using taint based evictions.")
	}
	nc.enterPartialDisruptionFunc = nc.ReducedQPSFunc
	nc.enterFullDisruptionFunc = nc.HealthyQPSFunc
	nc.computeZoneStateFunc = nc.ComputeZoneState

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			nc.maybeDeleteTerminatingPod(obj)
			pod := obj.(*v1.Pod)
			if nc.taintManager != nil {
				nc.taintManager.PodUpdated(nil, pod)
			}
		},
		UpdateFunc: func(prev, obj interface{}) {
			nc.maybeDeleteTerminatingPod(obj)
			prevPod := prev.(*v1.Pod)
			newPod := obj.(*v1.Pod)
			if nc.taintManager != nil {
				nc.taintManager.PodUpdated(prevPod, newPod)
			}
		},
		DeleteFunc: func(obj interface{}) {
			pod, isPod := obj.(*v1.Pod)
			// We can get DeletedFinalStateUnknown instead of *v1.Node here and we need to handle that correctly. #34692
			if !isPod {
				deletedState, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					glog.Errorf("Received unexpected object: %v", obj)
					return
				}
				pod, ok = deletedState.Obj.(*v1.Pod)
				if !ok {
					glog.Errorf("DeletedFinalStateUnknown contained non-Node object: %v", deletedState.Obj)
					return
				}
			}
			if nc.taintManager != nil {
				nc.taintManager.PodUpdated(pod, nil)
			}
		},
	})
	nc.podInformerSynced = podInformer.Informer().HasSynced

	nodeEventHandlerFuncs := cache.ResourceEventHandlerFuncs{}
	if nc.allocateNodeCIDRs {
		var nodeList *v1.NodeList
		var err error
		// We must poll because apiserver might not be up. This error causes
		// controller manager to restart.
		if pollErr := wait.Poll(10*time.Second, apiserverStartupGracePeriod, func() (bool, error) {
			nodeList, err = kubeClient.Core().Nodes().List(metav1.ListOptions{
				FieldSelector: fields.Everything().String(),
				LabelSelector: labels.Everything().String(),
			})
			if err != nil {
				glog.Errorf("Failed to list all nodes: %v", err)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			return nil, fmt.Errorf("Failed to list all nodes in %v, cannot proceed without updating CIDR map", apiserverStartupGracePeriod)
		}
		nc.cidrAllocator, err = NewCIDRRangeAllocator(kubeClient, clusterCIDR, serviceCIDR, nodeCIDRMaskSize, nodeList)
		if err != nil {
			return nil, err
		}

		nodeEventHandlerFuncs = cache.ResourceEventHandlerFuncs{
			AddFunc: func(originalObj interface{}) {
				obj, err := api.Scheme.DeepCopy(originalObj)
				if err != nil {
					utilruntime.HandleError(err)
					return
				}
				node := obj.(*v1.Node)

				if err := nc.cidrAllocator.AllocateOrOccupyCIDR(node); err != nil {
					utilruntime.HandleError(fmt.Errorf("Error allocating CIDR: %v", err))
				}
				if nc.taintManager != nil {
					nc.taintManager.NodeUpdated(nil, node)
				}
			},
			UpdateFunc: func(oldNode, newNode interface{}) {
				node := newNode.(*v1.Node)
				prevNode := oldNode.(*v1.Node)
				// If the PodCIDR is not empty we either:
				// - already processed a Node that already had a CIDR after NC restarted
				//   (cidr is marked as used),
				// - already processed a Node successfully and allocated a CIDR for it
				//   (cidr is marked as used),
				// - already processed a Node but we did saw a "timeout" response and
				//   request eventually got through in this case we haven't released
				//   the allocated CIDR (cidr is still marked as used).
				// There's a possible error here:
				// - NC sees a new Node and assigns a CIDR X to it,
				// - Update Node call fails with a timeout,
				// - Node is updated by some other component, NC sees an update and
				//   assigns CIDR Y to the Node,
				// - Both CIDR X and CIDR Y are marked as used in the local cache,
				//   even though Node sees only CIDR Y
				// The problem here is that in in-memory cache we see CIDR X as marked,
				// which prevents it from being assigned to any new node. The cluster
				// state is correct.
				// Restart of NC fixes the issue.
				if node.Spec.PodCIDR == "" {
					nodeCopy, err := api.Scheme.Copy(node)
					if err != nil {
						utilruntime.HandleError(err)
						return
					}

					if err := nc.cidrAllocator.AllocateOrOccupyCIDR(nodeCopy.(*v1.Node)); err != nil {
						utilruntime.HandleError(fmt.Errorf("Error allocating CIDR: %v", err))
					}
				}
				if nc.taintManager != nil {
					nc.taintManager.NodeUpdated(prevNode, node)
				}
			},
			DeleteFunc: func(originalObj interface{}) {
				obj, err := api.Scheme.DeepCopy(originalObj)
				if err != nil {
					utilruntime.HandleError(err)
					return
				}

				node, isNode := obj.(*v1.Node)
				// We can get DeletedFinalStateUnknown instead of *v1.Node here and we need to handle that correctly. #34692
				if !isNode {
					deletedState, ok := obj.(cache.DeletedFinalStateUnknown)
					if !ok {
						glog.Errorf("Received unexpected object: %v", obj)
						return
					}
					node, ok = deletedState.Obj.(*v1.Node)
					if !ok {
						glog.Errorf("DeletedFinalStateUnknown contained non-Node object: %v", deletedState.Obj)
						return
					}
				}
				if nc.taintManager != nil {
					nc.taintManager.NodeUpdated(node, nil)
				}
				if err := nc.cidrAllocator.ReleaseCIDR(node); err != nil {
					glog.Errorf("Error releasing CIDR: %v", err)
				}
			},
		}
	} else {
		nodeEventHandlerFuncs = cache.ResourceEventHandlerFuncs{
			AddFunc: func(originalObj interface{}) {
				obj, err := api.Scheme.DeepCopy(originalObj)
				if err != nil {
					utilruntime.HandleError(err)
					return
				}
				node := obj.(*v1.Node)
				if nc.taintManager != nil {
					nc.taintManager.NodeUpdated(nil, node)
				}
			},
			UpdateFunc: func(oldNode, newNode interface{}) {
				node := newNode.(*v1.Node)
				prevNode := oldNode.(*v1.Node)
				if nc.taintManager != nil {
					nc.taintManager.NodeUpdated(prevNode, node)

				}
			},
			DeleteFunc: func(originalObj interface{}) {
				obj, err := api.Scheme.DeepCopy(originalObj)
				if err != nil {
					utilruntime.HandleError(err)
					return
				}

				node, isNode := obj.(*v1.Node)
				// We can get DeletedFinalStateUnknown instead of *v1.Node here and we need to handle that correctly. #34692
				if !isNode {
					deletedState, ok := obj.(cache.DeletedFinalStateUnknown)
					if !ok {
						glog.Errorf("Received unexpected object: %v", obj)
						return
					}
					node, ok = deletedState.Obj.(*v1.Node)
					if !ok {
						glog.Errorf("DeletedFinalStateUnknown contained non-Node object: %v", deletedState.Obj)
						return
					}
				}
				if nc.taintManager != nil {
					nc.taintManager.NodeUpdated(node, nil)
				}
			},
		}
	}

	if nc.runTaintManager {
		nc.taintManager = NewNoExecuteTaintManager(kubeClient)
	}

	nodeInformer.Informer().AddEventHandler(nodeEventHandlerFuncs)
	nc.nodeLister = nodeInformer.Lister()
	nc.nodeInformerSynced = nodeInformer.Informer().HasSynced

	nc.daemonSetStore = daemonSetInformer.Lister()
	nc.daemonSetInformerSynced = daemonSetInformer.Informer().HasSynced

	return nc, nil
}

// Run starts an asynchronous loop that monitors the status of cluster nodes.
func (nc *NodeController) Run() {
	go func() {
		defer utilruntime.HandleCrash()

		if !cache.WaitForCacheSync(wait.NeverStop, nc.nodeInformerSynced, nc.podInformerSynced, nc.daemonSetInformerSynced) {
			utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
			return
		}

		// Incorporate the results of node status pushed from kubelet to master.
		go wait.Until(func() {
			if err := nc.monitorNodeStatus(); err != nil {
				glog.Errorf("Error monitoring node status: %v", err)
			}
		}, nc.nodeMonitorPeriod, wait.NeverStop)

		if nc.runTaintManager {
			go nc.taintManager.Run(wait.NeverStop)
		}

		if nc.useTaintBasedEvictions {
			// Handling taint based evictions. Because we don't want a dedicated logic in TaintManager for NC-originated
			// taints and we normally don't rate limit evictions caused by taints, we need to rate limit adding taints.
			go wait.Until(func() {
				nc.evictorLock.Lock()
				defer nc.evictorLock.Unlock()
				for k := range nc.zoneNotReadyOrUnreachableTainer {
					// Function should return 'false' and a time after which it should be retried, or 'true' if it shouldn't (it succeeded).
					nc.zoneNotReadyOrUnreachableTainer[k].Try(func(value TimedValue) (bool, time.Duration) {
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
							EvictionsNumber.WithLabelValues(zone).Inc()
						}
						_, condition := v1.GetNodeCondition(&node.Status, v1.NodeReady)
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

						taintToAdd.TimeAdded = metav1.Now()
						err = controller.AddOrUpdateTaintOnNode(nc.kubeClient, value.Value, &taintToAdd)
						if err != nil {
							utilruntime.HandleError(
								fmt.Errorf(
									"unable to taint %v unresponsive Node %q: %v",
									taintToAdd.Key,
									value.Value,
									err))
							return false, 0
						} else {
							glog.V(4).Info("Added %v Taint to Node %v", taintToAdd, value.Value)
						}
						err = controller.RemoveTaintOffNode(nc.kubeClient, value.Value, &oppositeTaint, node)
						if err != nil {
							utilruntime.HandleError(
								fmt.Errorf(
									"unable to remove %v unneeded taint from unresponsive Node %q: %v",
									oppositeTaint.Key,
									value.Value,
									err))
							return false, 0
						} else {
							glog.V(4).Info("Made sure that Node %v has no %v Taint", value.Value, oppositeTaint)
						}
						return true, 0
					})
				}
			}, nodeEvictionPeriod, wait.NeverStop)
		} else {
			// Managing eviction of nodes:
			// When we delete pods off a node, if the node was not empty at the time we then
			// queue an eviction watcher. If we hit an error, retry deletion.
			go wait.Until(func() {
				nc.evictorLock.Lock()
				defer nc.evictorLock.Unlock()
				for k := range nc.zonePodEvictor {
					// Function should return 'false' and a time after which it should be retried, or 'true' if it shouldn't (it succeeded).
					nc.zonePodEvictor[k].Try(func(value TimedValue) (bool, time.Duration) {
						node, err := nc.nodeLister.Get(value.Value)
						if apierrors.IsNotFound(err) {
							glog.Warningf("Node %v no longer present in nodeLister!", value.Value)
						} else if err != nil {
							glog.Warningf("Failed to get Node %v from the nodeLister: %v", value.Value, err)
						} else {
							zone := utilnode.GetZoneKey(node)
							EvictionsNumber.WithLabelValues(zone).Inc()
						}
						nodeUid, _ := value.UID.(string)
						remaining, err := deletePods(nc.kubeClient, nc.recorder, value.Value, nodeUid, nc.daemonSetStore)
						if err != nil {
							utilruntime.HandleError(fmt.Errorf("unable to evict node %q: %v", value.Value, err))
							return false, 0
						}
						if remaining {
							glog.Infof("Pods awaiting deletion due to NodeController eviction")
						}
						return true, 0
					})
				}
			}, nodeEvictionPeriod, wait.NeverStop)
		}
	}()
}

// monitorNodeStatus verifies node status are constantly updated by kubelet, and if not,
// post "NodeReady==ConditionUnknown". It also evicts all pods if node is not ready or
// not reachable for a long period of time.
func (nc *NodeController) monitorNodeStatus() error {
	// We are listing nodes from local cache as we can tolerate some small delays
	// comparing to state from etcd and there is eventual consistency anyway.
	nodes, err := nc.nodeLister.List(labels.Everything())
	if err != nil {
		return err
	}
	added, deleted := nc.checkForNodeAddedDeleted(nodes)
	for i := range added {
		glog.V(1).Infof("NodeController observed a new Node: %#v", added[i].Name)
		recordNodeEvent(nc.recorder, added[i].Name, string(added[i].UID), v1.EventTypeNormal, "RegisteredNode", fmt.Sprintf("Registered Node %v in NodeController", added[i].Name))
		nc.knownNodeSet[added[i].Name] = added[i]
		// When adding new Nodes we need to check if new zone appeared, and if so add new evictor.
		zone := utilnode.GetZoneKey(added[i])
		if _, found := nc.zoneStates[zone]; !found {
			nc.zoneStates[zone] = stateInitial
			if !nc.useTaintBasedEvictions {
				nc.zonePodEvictor[zone] =
					NewRateLimitedTimedQueue(
						flowcontrol.NewTokenBucketRateLimiter(nc.evictionLimiterQPS, evictionRateLimiterBurst))
			} else {
				nc.zoneNotReadyOrUnreachableTainer[zone] =
					NewRateLimitedTimedQueue(
						flowcontrol.NewTokenBucketRateLimiter(nc.evictionLimiterQPS, evictionRateLimiterBurst))
			}
			// Init the metric for the new zone.
			glog.Infof("Initializing eviction metric for zone: %v", zone)
			EvictionsNumber.WithLabelValues(zone).Add(0)
		}
		if nc.useTaintBasedEvictions {
			nc.markNodeAsHealthy(added[i])
		} else {
			nc.cancelPodEviction(added[i])
		}
	}

	for i := range deleted {
		glog.V(1).Infof("NodeController observed a Node deletion: %v", deleted[i].Name)
		recordNodeEvent(nc.recorder, deleted[i].Name, string(deleted[i].UID), v1.EventTypeNormal, "RemovingNode", fmt.Sprintf("Removing Node %v from NodeController", deleted[i].Name))
		delete(nc.knownNodeSet, deleted[i].Name)
	}

	zoneToNodeConditions := map[string][]*v1.NodeCondition{}
	for i := range nodes {
		var gracePeriod time.Duration
		var observedReadyCondition v1.NodeCondition
		var currentReadyCondition *v1.NodeCondition
		nodeCopy, err := api.Scheme.DeepCopy(nodes[i])
		if err != nil {
			utilruntime.HandleError(err)
			continue
		}
		node := nodeCopy.(*v1.Node)
		if err := wait.PollImmediate(retrySleepTime, retrySleepTime*nodeStatusUpdateRetry, func() (bool, error) {
			gracePeriod, observedReadyCondition, currentReadyCondition, err = nc.tryUpdateNodeStatus(node)
			if err == nil {
				return true, nil
			}
			name := node.Name
			node, err = nc.kubeClient.Core().Nodes().Get(name, metav1.GetOptions{})
			if err != nil {
				glog.Errorf("Failed while getting a Node to retry updating NodeStatus. Probably Node %s was deleted.", name)
				return false, err
			}
			return false, nil
		}); err != nil {
			glog.Errorf("Update status  of Node %v from NodeController error : %v. "+
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
					if nc.markNodeForTainting(node) {
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
					if nc.markNodeForTainting(node) {
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
					removed, err := nc.markNodeAsHealthy(node)
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
			}

			// Report node event.
			if currentReadyCondition.Status != v1.ConditionTrue && observedReadyCondition.Status == v1.ConditionTrue {
				recordNodeStatusChange(nc.recorder, node, "NodeNotReady")
				if err = markAllPodsNotReady(nc.kubeClient, node); err != nil {
					utilruntime.HandleError(fmt.Errorf("Unable to mark all pods NotReady on node %v: %v", node.Name, err))
				}
			}

			// Check with the cloud provider to see if the node still exists. If it
			// doesn't, delete the node immediately.
			if currentReadyCondition.Status != v1.ConditionTrue && nc.cloud != nil {
				exists, err := nc.nodeExistsInCloudProvider(types.NodeName(node.Name))
				if err != nil {
					glog.Errorf("Error determining if node %v exists in cloud: %v", node.Name, err)
					continue
				}
				if !exists {
					glog.V(2).Infof("Deleting node (no longer present in cloud provider): %s", node.Name)
					recordNodeEvent(nc.recorder, node.Name, string(node.UID), v1.EventTypeNormal, "DeletingNode", fmt.Sprintf("Deleting Node %v because it's not present according to cloud provider", node.Name))
					go func(nodeName string) {
						defer utilruntime.HandleCrash()
						// Kubelet is not reporting and Cloud Provider says node
						// is gone. Delete it without worrying about grace
						// periods.
						if err := forcefullyDeleteNode(nc.kubeClient, nodeName); err != nil {
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

func (nc *NodeController) handleDisruption(zoneToNodeConditions map[string][]*v1.NodeCondition, nodes []*v1.Node) {
	newZoneStates := map[string]zoneState{}
	allAreFullyDisrupted := true
	for k, v := range zoneToNodeConditions {
		ZoneSize.WithLabelValues(k).Set(float64(len(v)))
		unhealthy, newState := nc.computeZoneStateFunc(v)
		ZoneHealth.WithLabelValues(k).Set(float64(100*(len(v)-unhealthy)) / float64(len(v)))
		UnhealthyNodes.WithLabelValues(k).Set(float64(unhealthy))
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
			ZoneSize.WithLabelValues(k).Set(0)
			ZoneHealth.WithLabelValues(k).Set(100)
			UnhealthyNodes.WithLabelValues(k).Set(0)
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
			glog.V(0).Info("NodeController detected that all Nodes are not-Ready. Entering master disruption mode.")
			for i := range nodes {
				if nc.useTaintBasedEvictions {
					_, err := nc.markNodeAsHealthy(nodes[i])
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
					nc.zoneNotReadyOrUnreachableTainer[k].SwapLimiter(0)
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
			glog.V(0).Info("NodeController detected that some Nodes are Ready. Exiting master disruption mode.")
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
			glog.V(0).Infof("NodeController detected that zone %v is now in state %v.", k, newState)
			nc.setLimiterInZone(k, len(zoneToNodeConditions[k]), newState)
			nc.zoneStates[k] = newState
		}
	}
}

func (nc *NodeController) setLimiterInZone(zone string, zoneSize int, state zoneState) {
	switch state {
	case stateNormal:
		if nc.useTaintBasedEvictions {
			nc.zoneNotReadyOrUnreachableTainer[zone].SwapLimiter(nc.evictionLimiterQPS)
		} else {
			nc.zonePodEvictor[zone].SwapLimiter(nc.evictionLimiterQPS)
		}
	case statePartialDisruption:
		if nc.useTaintBasedEvictions {
			nc.zoneNotReadyOrUnreachableTainer[zone].SwapLimiter(
				nc.enterPartialDisruptionFunc(zoneSize))
		} else {
			nc.zonePodEvictor[zone].SwapLimiter(
				nc.enterPartialDisruptionFunc(zoneSize))
		}
	case stateFullDisruption:
		if nc.useTaintBasedEvictions {
			nc.zoneNotReadyOrUnreachableTainer[zone].SwapLimiter(
				nc.enterFullDisruptionFunc(zoneSize))
		} else {
			nc.zonePodEvictor[zone].SwapLimiter(
				nc.enterFullDisruptionFunc(zoneSize))
		}
	}
}

// For a given node checks its conditions and tries to update it. Returns grace period to which given node
// is entitled, state of current and last observed Ready Condition, and an error if it occurred.
func (nc *NodeController) tryUpdateNodeStatus(node *v1.Node) (time.Duration, v1.NodeCondition, *v1.NodeCondition, error) {
	var err error
	var gracePeriod time.Duration
	var observedReadyCondition v1.NodeCondition
	_, currentReadyCondition := v1.GetNodeCondition(&node.Status, v1.NodeReady)
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
	// - saved status have no Ready Condition, but current one does - NodeController was restarted with Node data already present in etcd,
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
		_, savedCondition = v1.GetNodeCondition(&savedNodeStatus.status, v1.NodeReady)
	}
	_, observedCondition := v1.GetNodeCondition(&node.Status, v1.NodeReady)
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
			glog.V(3).Infof("ReadyCondition for Node %s transitioned from %v to %v", node.Name, savedCondition.Status, observedCondition)

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
		remainingNodeConditionTypes := []v1.NodeConditionType{v1.NodeOutOfDisk, v1.NodeMemoryPressure, v1.NodeDiskPressure}
		nowTimestamp := nc.now()
		for _, nodeConditionType := range remainingNodeConditionTypes {
			_, currentCondition := v1.GetNodeCondition(&node.Status, nodeConditionType)
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

		_, currentCondition := v1.GetNodeCondition(&node.Status, v1.NodeReady)
		if !apiequality.Semantic.DeepEqual(currentCondition, &observedReadyCondition) {
			if _, err = nc.kubeClient.Core().Nodes().UpdateStatus(node); err != nil {
				glog.Errorf("Error updating node %s: %v", node.Name, err)
				return gracePeriod, observedReadyCondition, currentReadyCondition, err
			} else {
				nc.nodeStatusMap[node.Name] = nodeStatusData{
					status:                   node.Status,
					probeTimestamp:           nc.nodeStatusMap[node.Name].probeTimestamp,
					readyTransitionTimestamp: nc.now(),
				}
				return gracePeriod, observedReadyCondition, currentReadyCondition, nil
			}
		}
	}

	return gracePeriod, observedReadyCondition, currentReadyCondition, err
}

func (nc *NodeController) checkForNodeAddedDeleted(nodes []*v1.Node) (added, deleted []*v1.Node) {
	for i := range nodes {
		if _, has := nc.knownNodeSet[nodes[i].Name]; !has {
			added = append(added, nodes[i])
		}
	}
	// If there's a difference between lengths of known Nodes and observed nodes
	// we must have removed some Node.
	if len(nc.knownNodeSet)+len(added) != len(nodes) {
		knowSetCopy := map[string]*v1.Node{}
		for k, v := range nc.knownNodeSet {
			knowSetCopy[k] = v
		}
		for i := range nodes {
			delete(knowSetCopy, nodes[i].Name)
		}
		for i := range knowSetCopy {
			deleted = append(deleted, knowSetCopy[i])
		}
	}
	return
}

// cancelPodEviction removes any queued evictions, typically because the node is available again. It
// returns true if an eviction was queued.
func (nc *NodeController) cancelPodEviction(node *v1.Node) bool {
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
func (nc *NodeController) evictPods(node *v1.Node) bool {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	return nc.zonePodEvictor[utilnode.GetZoneKey(node)].Add(node.Name, string(node.UID))
}

func (nc *NodeController) markNodeForTainting(node *v1.Node) bool {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	return nc.zoneNotReadyOrUnreachableTainer[utilnode.GetZoneKey(node)].Add(node.Name, string(node.UID))
}

func (nc *NodeController) markNodeAsHealthy(node *v1.Node) (bool, error) {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	err := controller.RemoveTaintOffNode(nc.kubeClient, node.Name, UnreachableTaintTemplate, node)
	if err != nil {
		glog.Errorf("Failed to remove taint from node %v: %v", node.Name, err)
		return false, err
	}
	err = controller.RemoveTaintOffNode(nc.kubeClient, node.Name, NotReadyTaintTemplate, node)
	if err != nil {
		glog.Errorf("Failed to remove taint from node %v: %v", node.Name, err)
		return false, err
	}
	return nc.zoneNotReadyOrUnreachableTainer[utilnode.GetZoneKey(node)].Remove(node.Name), nil
}

// Default value for cluster eviction rate - we take nodeNum for consistency with ReducedQPSFunc.
func (nc *NodeController) HealthyQPSFunc(nodeNum int) float32 {
	return nc.evictionLimiterQPS
}

// If the cluster is large make evictions slower, if they're small stop evictions altogether.
func (nc *NodeController) ReducedQPSFunc(nodeNum int) float32 {
	if int32(nodeNum) > nc.largeClusterThreshold {
		return nc.secondaryEvictionLimiterQPS
	}
	return 0
}

// This function is expected to get a slice of NodeReadyConditions for all Nodes in a given zone.
// The zone is considered:
// - fullyDisrupted if there're no Ready Nodes,
// - partiallyDisrupted if at least than nc.unhealthyZoneThreshold percent of Nodes are not Ready,
// - normal otherwise
func (nc *NodeController) ComputeZoneState(nodeReadyConditions []*v1.NodeCondition) (int, zoneState) {
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
