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
	"context"
	"fmt"
	"math"
	"strconv"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"

	"k8s.io/klog/v2"
)

// Controller sets ttl annotations on nodes, based on cluster size.
type Controller struct {
	kubeClient clientset.Interface

	// nodeStore is a local cache of nodes.
	nodeStore listers.NodeLister

	// Nodes that need to be synced.
	queue workqueue.TypedRateLimitingInterface[string]

	// Returns true if all underlying informers are synced.
	hasSynced func() bool

	lock sync.RWMutex

	// Number of nodes in the cluster.
	nodeCount int

	// Desired TTL for all nodes in the cluster.
	desiredTTLSeconds int

	// In which interval of cluster size we currently are.
	boundaryStep int
}

// NewTTLController creates a new TTLController
func NewTTLController(ctx context.Context, nodeInformer informers.NodeInformer, kubeClient clientset.Interface) *Controller {
	ttlc := &Controller{
		kubeClient: kubeClient,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "ttlcontroller"},
		),
	}
	logger := klog.FromContext(ctx)
	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			ttlc.addNode(logger, obj)
		},
		UpdateFunc: func(old, newObj interface{}) {
			ttlc.updateNode(logger, old, newObj)
		},
		DeleteFunc: ttlc.deleteNode,
	})

	ttlc.nodeStore = listers.NewNodeLister(nodeInformer.Informer().GetIndexer())
	ttlc.hasSynced = nodeInformer.Informer().HasSynced

	return ttlc
}

type ttlBoundary struct {
	sizeMin    int
	sizeMax    int
	ttlSeconds int
}

var (
	ttlBoundaries = []ttlBoundary{
		{sizeMin: 0, sizeMax: 100, ttlSeconds: 0},
		{sizeMin: 90, sizeMax: 500, ttlSeconds: 15},
		{sizeMin: 450, sizeMax: 1000, ttlSeconds: 30},
		{sizeMin: 900, sizeMax: 2000, ttlSeconds: 60},
		{sizeMin: 1800, sizeMax: math.MaxInt32, ttlSeconds: 300},
	}
)

// Run begins watching and syncing.
func (ttlc *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer ttlc.queue.ShutDown()
	logger := klog.FromContext(ctx)
	logger.Info("Starting TTL controller")
	defer logger.Info("Shutting down TTL controller")

	if !cache.WaitForNamedCacheSync("TTL", ctx.Done(), ttlc.hasSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, ttlc.worker, time.Second)
	}

	<-ctx.Done()
}

func (ttlc *Controller) addNode(logger klog.Logger, obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
		return
	}

	func() {
		ttlc.lock.Lock()
		defer ttlc.lock.Unlock()
		ttlc.nodeCount++
		if ttlc.nodeCount > ttlBoundaries[ttlc.boundaryStep].sizeMax {
			ttlc.boundaryStep++
			ttlc.desiredTTLSeconds = ttlBoundaries[ttlc.boundaryStep].ttlSeconds
		}
	}()
	ttlc.enqueueNode(logger, node)
}

func (ttlc *Controller) updateNode(logger klog.Logger, _, newObj interface{}) {
	node, ok := newObj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}
	// Processing all updates of nodes guarantees that we will update
	// the ttl annotation, when cluster size changes.
	// We are relying on the fact that Kubelet is updating node status
	// every 10s (or generally every X seconds), which means that whenever
	// required, its ttl annotation should be updated within that period.
	ttlc.enqueueNode(logger, node)
}

func (ttlc *Controller) deleteNode(obj interface{}) {
	_, ok := obj.(*v1.Node)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", obj))
			return
		}
		_, ok = tombstone.Obj.(*v1.Node)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unexpected object types: %v", obj))
			return
		}
	}

	func() {
		ttlc.lock.Lock()
		defer ttlc.lock.Unlock()
		ttlc.nodeCount--
		if ttlc.nodeCount < ttlBoundaries[ttlc.boundaryStep].sizeMin {
			ttlc.boundaryStep--
			ttlc.desiredTTLSeconds = ttlBoundaries[ttlc.boundaryStep].ttlSeconds
		}
	}()
	// We are not processing the node, as it no longer exists.
}

func (ttlc *Controller) enqueueNode(logger klog.Logger, node *v1.Node) {
	key, err := controller.KeyFunc(node)
	if err != nil {
		logger.Error(nil, "Couldn't get key for object", "object", klog.KObj(node))
		return
	}
	ttlc.queue.Add(key)
}

func (ttlc *Controller) worker(ctx context.Context) {
	for ttlc.processItem(ctx) {
	}
}

func (ttlc *Controller) processItem(ctx context.Context) bool {
	key, quit := ttlc.queue.Get()
	if quit {
		return false
	}
	defer ttlc.queue.Done(key)

	err := ttlc.updateNodeIfNeeded(ctx, key)
	if err == nil {
		ttlc.queue.Forget(key)
		return true
	}

	ttlc.queue.AddRateLimited(key)
	utilruntime.HandleError(err)
	return true
}

func (ttlc *Controller) getDesiredTTLSeconds() int {
	ttlc.lock.RLock()
	defer ttlc.lock.RUnlock()
	return ttlc.desiredTTLSeconds
}

func getIntFromAnnotation(ctx context.Context, node *v1.Node, annotationKey string) (int, bool) {
	if node.Annotations == nil {
		return 0, false
	}
	annotationValue, ok := node.Annotations[annotationKey]
	if !ok {
		return 0, false
	}
	intValue, err := strconv.Atoi(annotationValue)
	if err != nil {
		logger := klog.FromContext(ctx)
		logger.Info("Could not convert the value with annotation key for the node", "annotationValue",
			annotationValue, "annotationKey", annotationKey, "node", klog.KObj(node))
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

func (ttlc *Controller) patchNodeWithAnnotation(ctx context.Context, node *v1.Node, annotationKey string, value int) error {
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
	_, err = ttlc.kubeClient.CoreV1().Nodes().Patch(ctx, node.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.V(2).Info("Failed to change ttl annotation for node", "node", klog.KObj(node), "err", err)
		return err
	}
	logger.V(2).Info("Changed ttl annotation", "node", klog.KObj(node), "TTL", time.Duration(value)*time.Second)
	return nil
}

func (ttlc *Controller) updateNodeIfNeeded(ctx context.Context, key string) error {
	node, err := ttlc.nodeStore.Get(key)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	desiredTTL := ttlc.getDesiredTTLSeconds()
	currentTTL, ok := getIntFromAnnotation(ctx, node, v1.ObjectTTLAnnotationKey)
	if ok && currentTTL == desiredTTL {
		return nil
	}

	return ttlc.patchNodeWithAnnotation(ctx, node.DeepCopy(), v1.ObjectTTLAnnotationKey, desiredTTL)
}
