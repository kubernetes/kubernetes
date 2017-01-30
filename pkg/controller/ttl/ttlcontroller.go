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

// The TTLController sets ttl annotations on nodes, based on cluster size.
// The annotations are consumed by Kubelets as suggestions for how long
// it can cache objects (e.g. secrets or config maps) before refetching
// from apiserver again.
//
// TODO: This is a temporary workaround for the Kubelet not being able to
// send "watch secrets attached to pods from my node" request. Once
// sending such request will be possible, we will modify Kubelet to
// use it and get rid of this controller completely.

package ttl

import (
	"fmt"
	"strconv"
	"sync/atomic"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	listers "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"

	"github.com/golang/glog"
)

type TTLController struct {
	kubeClient clientset.Interface

	// nodeStore is a local cache of nodes.
	nodeStore listers.NodeLister

	// Nodes that need to be synced.
	queue workqueue.RateLimitingInterface

	// Returns true if all underlying informers are synced.
	hasSynced func() bool

	// Number of nodes in the cluster.
	// Should be used via atomic operations on it.
	nodeCount int32
}

func NewTTLController(nodeInformer informers.NodeInformer, kubeClient clientset.Interface) *TTLController {
	ttlc := &TTLController{
		kubeClient: kubeClient,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "ttlcontroller"),
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    ttlc.addNode,
		UpdateFunc: ttlc.updateNode,
		DeleteFunc: ttlc.deleteNode,
	})

	ttlc.nodeStore = listers.NewNodeLister(nodeInformer.Informer().GetIndexer())
	ttlc.hasSynced = nodeInformer.Informer().HasSynced

	return ttlc
}

func (ttlc *TTLController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer ttlc.queue.ShutDown()

	glog.Infof("Starting TTL controller")
	defer glog.Infof("Shutting down TTL controller")
	if !cache.WaitForCacheSync(stopCh, ttlc.hasSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(ttlc.worker, time.Second, stopCh)
	}

	<-stopCh
}

func (ttlc *TTLController) addNode(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object types: %v", obj))
		return
	}
	atomic.AddInt32(&ttlc.nodeCount, 1)
	ttlc.enqueueNode(node)
}

func (ttlc *TTLController) updateNode(_, newObj interface{}) {
	node, ok := newObj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object types: %v", newObj))
		return
	}
	// Processing all updates of nodes guarantees that we will update
	// the ttl annotation, when cluster size changes.
	// We are relying on the fact that Kubelet is updating node status
	// every 10s (or generaly every X seconds), which means that whenever
	// required, its ttl annotation should be updated within that period.
	ttlc.enqueueNode(node)
}

func (ttlc *TTLController) deleteNode(obj interface{}) {
	_, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object types: %v", obj))
		return
	}
	atomic.AddInt32(&ttlc.nodeCount, -1)
	// We are not processing the node, as it no longer exists.
}

func (ttlc *TTLController) enqueueNode(node *v1.Node) {
	key, err := controller.KeyFunc(node)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v", node)
		return
	}
	ttlc.queue.Add(key)
}

func (ttlc *TTLController) worker() {
	for ttlc.processItem() {
		if quit := ttlc.processItem(); quit {
			glog.Infof("TTL controller worker shutting down")
			return
		}
	}
}

func (ttlc *TTLController) processItem() bool {
	key, quit := ttlc.queue.Get()
	if quit {
		return false
	}
	defer ttlc.queue.Done(key)

	err := ttlc.updateNodeIfNeeded(key.(string))
	if err == nil {
		ttlc.queue.Forget(key)
		return true
	}

	ttlc.queue.AddRateLimited(key)
	utilruntime.HandleError(err)
	return true
}

type ttlBoundary struct {
	nodeCount  int32
	ttlSeconds int
}

var (
	ttlBoundaries = []ttlBoundary{
		{nodeCount: 100, ttlSeconds: 0},
		{nodeCount: 500, ttlSeconds: 15},
		{nodeCount: 1000, ttlSeconds: 30},
		{nodeCount: 2000, ttlSeconds: 60},
		{nodeCount: 5000, ttlSeconds: 300},
	}
)

func (ttlc *TTLController) getDesiredTTLSeconds(currentTTL int) int {
	nodeCount := atomic.LoadInt32(&ttlc.nodeCount)
	// To avoid situation when we are changing the annotation all the time,
	// because the cluster size is flapping between X and X+1 nodes (e.g.
	// due to cluster autoscaler), we use the following algorithm:
	// - we compute the desired ttl first based on current cluster size.
	// - if we are supposed to lower the current ttl, the current cluster
	//   size must be at most boundary -1
	// Note that with this algorith, to have flapping ttls on nodes, the cluster
	// has to be flapping constantly betwenn <=X-1 and X+1<=.
	// WARNING: you can't make this "1" a higher constant, as it may result
	// in different nodes having different ttls over long period of time, e.g.
	// if we swap the 1 with say 5, then in the scenario:
	// - the cluster was of size 1500 => all nodes have ttl=60
	// - cluster resized to 996 => all nodes still have ttl=60
	// - one new node added => it will have ttl=30
	// And nothing would change over time.
	step := len(ttlBoundaries) - 1
	for ; step > 0 && nodeCount <= ttlBoundaries[step-1].nodeCount; step-- {
	}
	desiredTTL := ttlBoundaries[step].ttlSeconds
	for {
		if step == len(ttlBoundaries)-1 {
			break
		}
		if currentTTL <= desiredTTL || nodeCount <= ttlBoundaries[step].nodeCount-1 {
			break
		}
		step++
		desiredTTL = ttlBoundaries[step].ttlSeconds
	}
	return desiredTTL
}

func getIntFromAnnotation(node *v1.Node, annotationKey string) (int, bool) {
	if node.Annotations == nil {
		return 0, false
	}
	annotationValue, ok := node.Annotations[annotationKey]
	if !ok {
		return 0, false
	}
	intValue, err := strconv.Atoi(annotationValue)
	if err != nil {
		glog.Warningf("Cannot convert the value %q with annotation key %q for the node %q",
			annotationValue, annotationKey, node.Name)
		return 0, false
	}
	return intValue, true
}

func setIntAnnotation(node *v1.Node, annotationKey string, value int) {
	if node.Annotations == nil {
		node.Annotations = make(map[string]string)
	}
	node.Annotations[annotationKey] = strconv.Itoa(value)
}

func (ttlc *TTLController) patchNodeWithAnnotation(node *v1.Node, annotationKey string, value int) error {
	oldData, err := json.Marshal(node)
	if err != nil {
		return err
	}
	setIntAnnotation(node, annotationKey, value)
	newData, err := json.Marshal(node)
	if err != nil {
		return err
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &v1.Node{})
	if err != nil {
		return err
	}
	_, err = ttlc.kubeClient.Core().Nodes().Patch(node.Name, types.StrategicMergePatchType, patchBytes)
	if err != nil {
		glog.V(2).Infof("Failed to change ttl annotation for node %s: %v", node.Name, err)
		return err
	}
	glog.V(2).Infof("Changed ttl annotation for node %s: %v", node.Name, err)
	return nil
}

func (ttlc *TTLController) updateNodeIfNeeded(key string) error {
	node, err := ttlc.nodeStore.Get(key)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	currentTTL, ok := getIntFromAnnotation(node, v1.ObjectTTLAnnotationKey)
	desiredTTL := ttlc.getDesiredTTLSeconds(currentTTL)
	if ok && currentTTL == desiredTTL {
		return nil
	}

	objCopy, err := api.Scheme.DeepCopy(node)
	if err != nil {
		return err
	}
	return ttlc.patchNodeWithAnnotation(objCopy.(*v1.Node), v1.ObjectTTLAnnotationKey, desiredTTL)
}
