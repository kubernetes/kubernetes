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

package kubeletconfig

import (
	"math/rand"
	"time"

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

// newSharedNodeInformer returns a shared informer that uses `client` to watch the Node with
// `nodeName` for changes and respond with `addFunc`, `updateFunc`, and `deleteFunc`.
func newSharedNodeInformer(client clientset.Interface, nodeName string,
	addFunc func(newObj interface{}),
	updateFunc func(oldObj interface{}, newObj interface{}),
	deleteFunc func(deletedObj interface{})) cache.SharedInformer {
	// select nodes by name
	fieldSelector := fields.OneTermEqualSelector("metadata.name", nodeName)

	// add some randomness to resync period, which can help avoid controllers falling into lock-step
	minResyncPeriod := 15 * time.Minute
	factor := rand.Float64() + 1
	resyncPeriod := time.Duration(float64(minResyncPeriod.Nanoseconds()) * factor)

	lw := cache.NewListWatchFromClient(client.CoreV1().RESTClient(), "nodes", metav1.NamespaceAll, fieldSelector)

	handler := cache.ResourceEventHandlerFuncs{
		AddFunc:    addFunc,
		UpdateFunc: updateFunc,
		DeleteFunc: deleteFunc,
	}

	informer := cache.NewSharedInformer(lw, &apiv1.Node{}, resyncPeriod)
	informer.AddEventHandler(handler)

	return informer
}

// onAddNodeEvent calls onUpdateNodeEvent with the new object and a nil old object
func (cc *Controller) onAddNodeEvent(newObj interface{}) {
	cc.onUpdateNodeEvent(nil, newObj)
}

// onUpdateNodeEvent checks whether the configSource changed between oldObj and newObj, and pokes the
// configuration sync worker if there was a change
func (cc *Controller) onUpdateNodeEvent(oldObj interface{}, newObj interface{}) {
	newNode, ok := newObj.(*apiv1.Node)
	if !ok {
		utillog.Errorf("failed to cast new object to Node, couldn't handle event")
		return
	}
	if oldObj == nil {
		// Node was just added, need to sync
		utillog.Infof("initial Node watch event")
		cc.pokeConfigSourceWorker()
		return
	}
	oldNode, ok := oldObj.(*apiv1.Node)
	if !ok {
		utillog.Errorf("failed to cast old object to Node, couldn't handle event")
		return
	}
	if !apiequality.Semantic.DeepEqual(oldNode.Spec.ConfigSource, newNode.Spec.ConfigSource) {
		utillog.Infof("Node.Spec.ConfigSource was updated")
		cc.pokeConfigSourceWorker()
	}
}

// onDeleteNodeEvent logs a message if the Node was deleted
// We allow the sync-loop to continue, because it is possible that the Kubelet detected
// a Node with unexpected externalID and is attempting to delete and re-create the Node
// (see pkg/kubelet/kubelet_node_status.go), or that someone accidentally deleted the Node
// (the Kubelet will re-create it).
func (cc *Controller) onDeleteNodeEvent(deletedObj interface{}) {
	// For this case, we just log the event.
	// We don't want to poke the worker, because a temporary deletion isn't worth reporting an error for.
	// If the Node is deleted because the VM is being deleted, then the Kubelet has nothing to do.
	utillog.Infof("Node was deleted")
}

// onAddRemoteConfigSourceEvent calls onUpdateConfigMapEvent with the new object and a nil old object
func (cc *Controller) onAddRemoteConfigSourceEvent(newObj interface{}) {
	cc.onUpdateRemoteConfigSourceEvent(nil, newObj)
}

// onUpdateRemoteConfigSourceEvent checks whether the configSource changed between oldObj and newObj,
// and pokes the sync worker if there was a change
func (cc *Controller) onUpdateRemoteConfigSourceEvent(oldObj interface{}, newObj interface{}) {
	// since ConfigMap is currently the only source type, we handle that here
	newConfigMap, ok := newObj.(*apiv1.ConfigMap)
	if !ok {
		utillog.Errorf("failed to cast new object to ConfigMap, couldn't handle event")
		return
	}
	if oldObj == nil {
		// ConfigMap was just added, need to sync
		utillog.Infof("initial ConfigMap watch event")
		cc.pokeConfigSourceWorker()
		return
	}
	oldConfigMap, ok := oldObj.(*apiv1.ConfigMap)
	if !ok {
		utillog.Errorf("failed to cast old object to ConfigMap, couldn't handle event")
		return
	}
	if !apiequality.Semantic.DeepEqual(oldConfigMap, newConfigMap) {
		utillog.Infof("assigned ConfigMap was updated")
		cc.pokeConfigSourceWorker()
	}
}

// onDeleteRemoteConfigSourceEvent logs a message if the ConfigMap was deleted and pokes the sync worker
func (cc *Controller) onDeleteRemoteConfigSourceEvent(deletedObj interface{}) {
	// If the ConfigMap we're watching is deleted, we log the event and poke the sync worker.
	// This requires a sync, because if the Node is still configured to use the deleted ConfigMap,
	// the Kubelet should report a DownloadError.
	utillog.Infof("assigned ConfigMap was deleted")
	cc.pokeConfigSourceWorker()
}
