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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	utilequal "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/equal"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

// newSharedNodeInformer returns a shared informer that uses `client` to watch the Node with
// `nodeName` for changes and respond with `addFunc`, `updateFunc`, and `deleteFunc`.
func newSharedNodeInformer(client clientset.Interface, nodeName string,
	addFunc func(newObj interface{}),
	updateFunc func(oldObj interface{}, newObj interface{}),
	deleteFunc func(deletedObj interface{})) cache.SharedInformer {
	// select nodes by name
	fieldselector := fields.OneTermEqualSelector("metadata.name", nodeName)

	// add some randomness to resync period, which can help avoid controllers falling into lock-step
	minResyncPeriod := 15 * time.Minute
	factor := rand.Float64() + 1
	resyncPeriod := time.Duration(float64(minResyncPeriod.Nanoseconds()) * factor)

	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (kuberuntime.Object, error) {
			return client.Core().Nodes().List(metav1.ListOptions{
				FieldSelector: fieldselector.String(),
			})
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return client.Core().Nodes().Watch(metav1.ListOptions{
				FieldSelector:   fieldselector.String(),
				ResourceVersion: options.ResourceVersion,
			})
		},
	}

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
		cc.pokeConfigSourceWorker()
		return
	}
	oldNode, ok := oldObj.(*apiv1.Node)
	if !ok {
		utillog.Errorf("failed to cast old object to Node, couldn't handle event")
		return
	}
	if !utilequal.ConfigSourceEq(oldNode.Spec.ConfigSource, newNode.Spec.ConfigSource) {
		cc.pokeConfigSourceWorker()
	}
}

// onDeleteNodeEvent logs a message if the Node was deleted and may log errors
// if an unexpected DeletedFinalStateUnknown was received.
// We allow the sync-loop to continue, because it is possible that the Kubelet detected
// a Node with unexpected externalID and is attempting to delete and re-create the Node
// (see pkg/kubelet/kubelet_node_status.go), or that someone accidently deleted the Node
// (the Kubelet will re-create it).
func (cc *Controller) onDeleteNodeEvent(deletedObj interface{}) {
	node, ok := deletedObj.(*apiv1.Node)
	if !ok {
		tombstone, ok := deletedObj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utillog.Errorf("couldn't cast deleted object to DeletedFinalStateUnknown, object: %+v", deletedObj)
			return
		}
		node, ok = tombstone.Obj.(*apiv1.Node)
		if !ok {
			utillog.Errorf("received DeletedFinalStateUnknown object but it did not contain a Node, object: %+v", deletedObj)
			return
		}
		utillog.Infof("Node was deleted (DeletedFinalStateUnknown), sync-loop will continue because the Kubelet might recreate the Node, node: %+v", node)
		return
	}
	utillog.Infof("Node was deleted, sync-loop will continue because the Kubelet might recreate the Node, node: %+v", node)
}
