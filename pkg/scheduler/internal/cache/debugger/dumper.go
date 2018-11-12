/*
Copyright 2018 The Kubernetes Authors.

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

package debugger

import (
	"fmt"
	"strings"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/cache"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	"k8s.io/kubernetes/pkg/scheduler/internal/queue"
)

// CacheDumper writes some information from the scheduler cache and the scheduling queue to the
// scheduler logs for debugging purposes.
type CacheDumper struct {
	cache    internalcache.Cache
	podQueue queue.SchedulingQueue
}

// DumpAll writes cached nodes and scheduling queue information to the scheduler logs.
func (d *CacheDumper) DumpAll() {
	d.dumpNodes()
	d.dumpSchedulingQueue()
}

// dumpNodes writes NodeInfo to the scheduler logs.
func (d *CacheDumper) dumpNodes() {
	snapshot := d.cache.Snapshot()
	klog.Info("Dump of cached NodeInfo")
	for _, nodeInfo := range snapshot.Nodes {
		klog.Info(printNodeInfo(nodeInfo))
	}
}

// dumpSchedulingQueue writes pods in the scheduling queue to the scheduler logs.
func (d *CacheDumper) dumpSchedulingQueue() {
	waitingPods := d.podQueue.WaitingPods()
	var podData strings.Builder
	for _, p := range waitingPods {
		podData.WriteString(printPod(p))
	}
	klog.Infof("Dump of scheduling queue:\n%s", podData.String())
}

// printNodeInfo writes parts of NodeInfo to a string.
func printNodeInfo(n *cache.NodeInfo) string {
	var nodeData strings.Builder
	nodeData.WriteString(fmt.Sprintf("\nNode name: %+v\nRequested Resources: %+v\nAllocatable Resources:%+v\nNumber of Pods: %v\nPods:\n",
		n.Node().Name, n.RequestedResource(), n.AllocatableResource(), len(n.Pods())))
	// Dumping Pod Info
	for _, p := range n.Pods() {
		nodeData.WriteString(printPod(p))
	}
	return nodeData.String()
}

// printPod writes parts of a Pod object to a string.
func printPod(p *v1.Pod) string {
	return fmt.Sprintf("name: %v, namespace: %v, uid: %v, phase: %v, nominated node: %v\n", p.Name, p.Namespace, p.UID, p.Status.Phase, p.Status.NominatedNodeName)
}
