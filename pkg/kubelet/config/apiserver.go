/*
Copyright 2015 The Kubernetes Authors.

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

package config

import (
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// WaitForAPIServerSyncPeriod is the period between checks for the node list/watch initial sync
const WaitForAPIServerSyncPeriod = 1 * time.Second

// NewSourceApiserver creates a config source that watches and pulls from the apiserver.
func NewSourceApiserver(c clientset.Interface, nodeName types.NodeName, nodeHasSynced func() bool, updates chan<- interface{}) {
	lw := cache.NewListWatchFromClient(c.CoreV1().RESTClient(), "pods", metav1.NamespaceAll, fields.OneTermEqualSelector("spec.nodeName", string(nodeName)))

	// The Reflector responsible for watching pods at the apiserver should be run only after
	// the node sync with the apiserver has completed.
	klog.InfoS("Waiting for node sync before watching apiserver pods")
	go func() {
		for {
			if nodeHasSynced() {
				klog.V(4).InfoS("node sync completed")
				break
			}
			time.Sleep(WaitForAPIServerSyncPeriod)
			klog.V(4).InfoS("node sync has not completed yet")
		}
		klog.InfoS("Watching apiserver")
		newSourceApiserverFromLW(lw, updates)
	}()
}

// newSourceApiserverFromLW holds creates a config source that watches and pulls from the apiserver.
func newSourceApiserverFromLW(lw cache.ListerWatcher, updates chan<- interface{}) {
	send := func(objs []interface{}) {
		var pods []*v1.Pod
		for _, o := range objs {
			pods = append(pods, o.(*v1.Pod))
		}
		updates <- kubetypes.PodUpdate{Pods: pods, Op: kubetypes.SET, Source: kubetypes.ApiserverSource}
	}
	r := cache.NewReflector(lw, &v1.Pod{}, cache.NewUndeltaStore(send, cache.MetaNamespaceKeyFunc), 0)
	go r.Run(wait.NeverStop)
}
