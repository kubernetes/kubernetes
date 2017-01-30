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

package ttl

import (
	"strconv"
	"sync/atomic"
	"time"

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
	listers "k8s.io/kubernetes/pkg/client/legacylisters"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"

	"github.com/golang/glog"
)

const (
	// ObjectTTLAnnotation is a suggestion for kubelet for how long it can cache
	// an object (e.g. secret, config map) before fetching from apiserver again.
	ObjectTTLAnnotation = "node.kubernetes.io/ttl"
)

type TTLController struct {
	kubeClient clientset.Interface

	// nodeStore is a local cache of nodes.
	nodeStore listers.StoreToNodeLister

	// Secrets that need to be synced.
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

	ttlc.nodeStore = *nodeInformer.Lister()
	ttlc.hasSynced = nodeInformer.Informer().HasSynced

	return ttlc
}

func (ttlc *TTLController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer ttlc.queue.ShutDown()

	glog.Infof("Starting TTL controller")
	if !cache.WaitForCacheSync(stopCh, ttlc.hasSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(ttlc.worker, time.Second, stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down TTL controller")
}

func (ttlc *TTLController) addNode(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		glog.Errorf("Unexpected object types: %v", obj)
		return
	}
	atomic.AddInt32(&ttlc.nodeCount, 1)
	ttlc.enqueueNode(node)
}

func (ttlc *TTLController) updateNode(_, newObj interface{}) {
	node, ok := newObj.(*v1.Node)
	if !ok {
		glog.Errorf("Unexpected object type: %v", newObj)
		return
	}
	ttlc.enqueueNode(node)
}

func (ttlc *TTLController) deleteNode(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		glog.Errorf("Unexpected object types: %v", obj)
		return
	}
	atomic.AddInt32(&ttlc.nodeCount, -1)
	ttlc.enqueueNode(node)
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
	for {
		if quit := ttlc.processItem(); quit {
			glog.Infof("TTL controller worker shutting down")
			return
		}
	}
}

func (ttlc *TTLController) processItem() bool {
	key, quit := ttlc.queue.Get()
	if quit {
		return true
	}
	defer ttlc.queue.Done(key)

	err := ttlc.updateNodeIfNeeded(key.(string))
	if err == nil {
		ttlc.queue.Forget(key)
		return false
	}

	ttlc.queue.AddRateLimited(key)
	utilruntime.HandleError(err)
	return false
}

func (ttlc *TTLController) getDesiredTTLSeconds() int {
	nodeCount := atomic.LoadInt32(&ttlc.nodeCount)
	switch {
	case nodeCount <= 100:
		return 0
	case nodeCount <= 1000:
		return 30
	case nodeCount <= 2000:
		return 60
	default:
		return 300
	}
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
	return err
}

func (ttlc *TTLController) updateNodeIfNeeded(key string) error {
	obj, exists, err := ttlc.nodeStore.GetByKey(key)
	if err != nil {
		return err
	}
	if !exists {
		glog.Infof("Node %s no longer exists", key)
		return nil
	}
	node := obj.(*v1.Node)
	desiredTTL := ttlc.getDesiredTTLSeconds()

	value, ok := getIntFromAnnotation(node, ObjectTTLAnnotation)
	if ok && value == desiredTTL {
		return nil
	}
	objCopy, err := api.Scheme.DeepCopy(node)
	if err != nil {
		return err
	}
	return ttlc.patchNodeWithAnnotation(objCopy.(*v1.Node), ObjectTTLAnnotation, desiredTTL)
}
