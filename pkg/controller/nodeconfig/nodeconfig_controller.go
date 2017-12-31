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

package nodeconfig

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/wait"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	// TODO(mtaufen): import as apiv1
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller"
	// TODO(mtaufen): import as nodeutil
	"k8s.io/kubernetes/pkg/controller/node/util"

	// TODO(mtaufen): update this note to generate in a new place
	// mtaufen note to self:
	// run ./generate-groups.sh all k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go k8s.io/kubernetes/pkg/controller/node "nodeconfig:v1alpha1"
	// to generate the client, listers, informers (from code-generator repo)
	nodeconfigv1alpha1listers "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/listers/nodeconfig/v1alpha1"
	nodeconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/node/nodeconfig/v1alpha1"

	nodeconfigclientset "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/clientset/versioned"
	nodeconfiginformers "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/informers/externalversions"
)

// Controller is the controller that reconciles config to nodes
type Controller struct {
	kubeClient clientset.Interface

	// TODO(mtaufen): not sure we need this with watch (we should at least rename)
	nodeMonitorPeriod time.Duration

	nodeLister         corelisters.NodeLister
	nodeInformerSynced cache.InformerSynced

	// NodeConfigSourcePool stuff
	ncspInformerFactory nodeconfiginformers.SharedInformerFactory
	ncspLister          nodeconfigv1alpha1listers.NodeConfigSourcePoolLister
	ncspInformerSynced  cache.InformerSynced
	// TODO(mtaufen): rename to ncspMustSync
	pendingNodeConfigSourcePoolSync chan bool
}

// NewNodeConfigController returns a new node config controller to sync config from the
// NodeConfigSourcePool resource to selected nodes. It handles config updates via the
// CRD and autoscaling of nodes.
// TODO(mtaufen): consider a periodic check too to aid self-healing.
func NewNodeConfigController(
	nodeInformer coreinformers.NodeInformer,
	// TODO(mtaufen): See if we can construct this informer internally, so we can take the factory
	// out of ControllerContext
	// nodeConfigSourcePoolInformer nodeconfigv1alpha1informers.NodeConfigSourcePoolInformer,
	kubeClient clientset.Interface,
	jsonClientConfig *restclient.Config, // a client config with json as content type, we'll construct ncsp informer from it
	// TODO(mtaufen): give this var a more accurate name
	nodeMonitorPeriod time.Duration,
	resyncPeriod time.Duration) (*Controller, error) {

	if kubeClient == nil {
		glog.Fatalf("kubeClient is nil when starting Controller")
	}

	ncspClient := nodeconfigclientset.NewForConfigOrDie(jsonClientConfig)
	ncspInformerFactory := nodeconfiginformers.NewSharedInformerFactory(ncspClient, resyncPeriod)
	ncspInformer := ncspInformerFactory.Nodeconfig().V1alpha1().NodeConfigSourcePools()

	nc := &Controller{
		kubeClient:                      kubeClient,
		nodeMonitorPeriod:               nodeMonitorPeriod,
		ncspInformerFactory:             ncspInformerFactory,
		pendingNodeConfigSourcePoolSync: make(chan bool, 1),
	}

	// Event handlers for reconciling config to added nodes
	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: util.CreateAddNodeHandler(func(node *v1.Node) error {
			nc.needsNodeConfigSourcePoolSync()
			return nil
		}),
		UpdateFunc: util.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			// TODO(mtaufen): healing config to updated nodes (will need to check if things differ from what we expect)
			// compare config source on old vs new node, if changed, then trigger event (same as kubelet)
			return nil
		}),
	})

	nc.nodeLister = nodeInformer.Lister()
	nc.nodeInformerSynced = nodeInformer.Informer().HasSynced

	// TODO(mtaufen): event handlers on the informer
	ncspInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: CreateAddNodeConfigSourcePoolHandler(func(pool *nodeconfigv1alpha1.NodeConfigSourcePool) error {
			glog.Infof("added NodeConfigSourcePool")
			nc.needsNodeConfigSourcePoolSync()
			return nil
		}),
		UpdateFunc: CreateUpdateNodeConfigSourcePoolHandler(func(oldPool, newPool *nodeconfigv1alpha1.NodeConfigSourcePool) error {
			glog.Infof("updated NodeConfigSourcePool")
			nc.needsNodeConfigSourcePoolSync()
			return nil
		}),
		// TODO(mtaufen): consider whether deletion should have any side-effects on the nodes
	})

	// TODO(mtaufen): might not need these two lines
	nc.ncspLister = ncspInformer.Lister()
	nc.ncspInformerSynced = ncspInformer.Informer().HasSynced

	return nc, nil
}

// Run starts an asynchronous loop that monitors the status of cluster nodes.
func (nc *Controller) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Infof("Starting nodeconfig controller")
	defer glog.Infof("Shutting down nodeconfig controller")

	// TODO(mtaufen): wait for the cache to sync on the ncsp informer
	if !controller.WaitForCacheSync("nodeconfig", stopCh, nc.nodeInformerSynced /*nc.nodeConfigSourcePoolInformerSynced*/) {
		return
	}

	// TODO(mtaufen): start ncsp informer here, after the other caches have been synced
	// (the main shared informer is started after all the controllers are set up)
	// though it's probably safe to use the ncsp shared informer here since nobody else does
	nc.ncspInformerFactory.Start(stopCh)

	// TODO(mtauefn): wait for ncsp informer cache sync here
	// if !controller.WaitForCacheSync("nodeconfig", nc.ncspInformerSynced) {
	// 	return
	// }

	// initial poke for the sync worker thread, so that we always get a resync when the node controller starts up
	glog.Infof("about to make the magic happen")
	nc.needsNodeConfigSourcePoolSync()
	// periodically check whether we need to sync
	go wait.Until(nc.syncNodeConfigSourcePools, nc.nodeMonitorPeriod, wait.NeverStop)

	<-stopCh
}

func CreateAddNodeConfigSourcePoolHandler(f func(pool *nodeconfigv1alpha1.NodeConfigSourcePool) error) func(obj interface{}) {
	return func(obj interface{}) {
		pool := obj.(*nodeconfigv1alpha1.NodeConfigSourcePool).DeepCopy()
		if err := f(pool); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing NodeConfigSourcePool Add: %v", err))
		}
	}
}

func CreateUpdateNodeConfigSourcePoolHandler(f func(oldPool, newPool *nodeconfigv1alpha1.NodeConfigSourcePool) error) func(oldObj, newObj interface{}) {
	return func(oldObj, newObj interface{}) {
		oldPool := oldObj.(*nodeconfigv1alpha1.NodeConfigSourcePool).DeepCopy()
		newPool := newObj.(*nodeconfigv1alpha1.NodeConfigSourcePool).DeepCopy()
		if err := f(oldPool, newPool); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing NodeConfigSourcePool Update: %v", err))
		}
	}
}

func (nc *Controller) needsNodeConfigSourcePoolSync() {
	select {
	case nc.pendingNodeConfigSourcePoolSync <- true:
	default:
	}
}

// TODO(mtaufen): set log levels appropriately
// checks if work needs to be done to sync configurations, does work if necessary
func (nc *Controller) syncNodeConfigSourcePools() {
	// TODO(mtaufen): actually rely on the watch trigger
	// glog.Errorf("will sync NodeConfigSourcePools if necessary")
	// select {
	// case <-nc.pendingNodeConfigSourcePoolSync:
	// default:
	//  // no work to be done, return
	//  return
	// }

	glog.Infof("will definitely sync NodeConfigSourcePools")

	// if the sync fails, we want to retry
	var syncerr error
	defer func() {
		if syncerr != nil {
			// TODO(mtaufen): Use utilruntime.HandleError for errors
			glog.Errorf(syncerr.Error())
			nc.needsNodeConfigSourcePoolSync()
		}
	}()

	// get the nodeConfigSourcePools from the local cache
	pools, err := nc.ncspLister.List(labels.Everything())
	if err != nil {
		syncerr = err
		return
	}

	glog.Infof("NodeConfigSourcePools: %#v", pools)

	// pools is a list of pointers, so this doesn't copy the objects
	for _, pool := range pools {
		// validate percentNew, skip this pool if invalid
		// TODO(mtaufen): add this validation to the custom resource as well
		if pool.Spec.PercentNew < 0 || pool.Spec.PercentNew > 100 {
			utilruntime.HandleError(fmt.Errorf("invalid PercentNew for NodeConfigSourcePool %s/%s, must be between 0 and 100", pool.Namespace, pool.Name))
			continue
		}

		var oldSource, newSource *v1.NodeConfigSource
		// note this strategy will force default config on old-configured nodes for an empty or 1 element history
		n := len(pool.Spec.History)
		if n == 1 {
			newSource = pool.Spec.History[0]
		} else if n > 1 {
			newSource = pool.Spec.History[n-1]
			oldSource = pool.Spec.History[n-2]
		}

		newSourceNodes := []*v1.Node{}
		oldSourceNodes := []*v1.Node{}

		glog.Infof("length of pool history is %d", len(pool.Spec.History))

		// select the nodes in the pool from the local cache
		if pool.Spec.LabelSelector == nil {
			glog.Infof("pool %s has nil label selector", pool.Name)
		} else {
			glog.Infof("pool %s has label selector %#v", pool.Name, *pool.Spec.LabelSelector)
		}
		nodeSelector, err := metav1.LabelSelectorAsSelector(pool.Spec.LabelSelector)
		if err != nil {
			syncerr = err
			return
		}

		// TODO(mtaufen): Do I need to make copies of the nodes before modifying them? I probably shouldn't be modifying stuff directly in the store
		glog.Infof("listing nodes in pool %s with selector %#v", pool.Name, nodeSelector)
		nodes, err := nc.nodeLister.List(nodeSelector)
		if err != nil {
			syncerr = err
			return
		} else if len(nodes) == 0 {
			glog.Infof("no nodes in NodeConfigSourcePool %s", pool.Name)
			continue
		}

		// nodes is a list of pointers, so this doesn't copy the objects
		for _, node := range nodes {
			if ConfigSourceEq(newSource, node.Spec.ConfigSource) {
				newSourceNodes = append(newSourceNodes, node)
				continue
			}
			// if not on the new source, assume on an old source
			// all will shortly be forced to conform
			oldSourceNodes = append(oldSourceNodes, node)
		}

		glog.Infof("old source: %#v", oldSource)
		glog.Infof("new source: %#v", newSource)
		glog.Infof("desired percent: %d", pool.Spec.PercentNew)
		glog.Infof("current old source nodes: %#v", oldSourceNodes)
		glog.Infof("current new source nodes: %#v", newSourceNodes)

		// TODO(mtaufen): double check math
		// TODO(mtaufen): unit test the hell out of this

		// If you have a cluster with a small number of nodes, you'll obviously need to be careful to set a % that has a real effect
		// e.g. with 3 nodes, 34% will result in 1 node on the new config source

		totalNodes := len(oldSourceNodes) + len(newSourceNodes)

		targetNewSourceNodes := (pool.Spec.PercentNew * totalNodes) / 100

		if len(newSourceNodes) < targetNewSourceNodes {
			// move nodes from old to new
			n := targetNewSourceNodes - len(newSourceNodes)
			newSourceNodes = append(newSourceNodes, oldSourceNodes[:n]...)
			oldSourceNodes = oldSourceNodes[n:]
		} else if len(newSourceNodes) > targetNewSourceNodes {
			// move nodes from new to old
			n := len(newSourceNodes) - targetNewSourceNodes
			oldSourceNodes = append(oldSourceNodes, newSourceNodes[:n]...)
			newSourceNodes = newSourceNodes[n:]
		}

		glog.Infof("new old source nodes: %#v", oldSourceNodes)
		glog.Infof("new new source nodes: %#v", newSourceNodes)

		// map of node names to patches
		patches := map[string][]byte{}

		// generate all of the necessary patches
		for _, node := range oldSourceNodes {
			if !ConfigSourceEq(oldSource, node.Spec.ConfigSource) {
				patch := map[string]interface{}{
					// TOOD(mtaufen): copied this from adapter.go, is there a reason all this metadata is necessary?
					"apiVersion": node.APIVersion,
					"kind":       node.Kind,
					"metadata":   map[string]interface{}{"name": node.Name},
					"spec":       map[string]interface{}{"configSource": oldSource},
				}
				bytes, err := json.Marshal(patch)
				if err != nil {
					// if there is an error, just log it and skip to the next node
					utilruntime.HandleError(fmt.Errorf("could not generate ConfigSource patch for node %s in pool %s source %#v, error: %v", node.Name, pool.Name, oldSource, err))
					continue
				}
				patches[node.Name] = bytes
			}
		}
		for _, node := range newSourceNodes {
			if !ConfigSourceEq(newSource, node.Spec.ConfigSource) {
				patch := map[string]interface{}{
					// TOOD(mtaufen): copied this from adapter.go, is there a reason all this metadata is necessary?
					"apiVersion": node.APIVersion,
					"kind":       node.Kind,
					"metadata":   map[string]interface{}{"name": node.Name},
					"spec":       map[string]interface{}{"configSource": newSource},
				}
				bytes, err := json.Marshal(patch)
				if err != nil {
					// if there is an error, just log it and skip to the next node
					utilruntime.HandleError(fmt.Errorf("could not generate ConfigSource patch for node %s in pool %s source %#v, error: %v", node.Name, pool.Name, newSource, err))
					continue
				}
				patches[node.Name] = bytes
			}
		}

		// patch the nodes
		for name, patch := range patches {
			glog.Infof("will apply patch to %s: %s", name, string(patch))
			if _, err := nc.kubeClient.CoreV1().Nodes().Patch(name, types.StrategicMergePatchType, patch); err != nil {
				utilruntime.HandleError(err)
			}
		}
	}
}

// TODO(mtaufen): should put these eq helpers in a centralized place, at present just copypasta from the Kubelet
// TODO(mtaufen): for now these can probably be private

// ConfigSourceEq returns true if the two config sources are semantically equivalent in the context of dynamic config
func ConfigSourceEq(a, b *v1.NodeConfigSource) bool {
	if a == b {
		return true
	} else if a == nil || b == nil {
		// not equal, and one is nil
		return false
	}
	// check equality of config source subifelds
	if a.ConfigMapRef != b.ConfigMapRef {
		return ObjectRefEq(a.ConfigMapRef, b.ConfigMapRef)
	}
	// all internal subfields of the config soruce are equal
	return true
}

// ObjectRefEq returns true if the two object references are semantically equivalent in the context of dynamic config
func ObjectRefEq(a, b *v1.ObjectReference) bool {
	if a == b {
		return true
	} else if a == nil || b == nil {
		// not equal, and one is nil
		return false
	}
	return a.UID == b.UID && a.Namespace == b.Namespace && a.Name == b.Name
}
