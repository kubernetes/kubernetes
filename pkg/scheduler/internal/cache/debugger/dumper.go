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

	"k8s.io/klog/v2"

	"k8s.io/api/core/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
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
	dump := d.cache.Dump()
	klog.Info("Dump of cached NodeInfo")
	for _, nodeInfo := range dump.Nodes {
		klog.Info(d.printNodeInfo(nodeInfo))
	}
}

// dumpSchedulingQueue writes pods in the scheduling queue to the scheduler logs.
func (d *CacheDumper) dumpSchedulingQueue() {
	pendingPods := d.podQueue.PendingPods()
	var podData strings.Builder
	for _, p := range pendingPods {
		podData.WriteString(printPod(p))
	}
	klog.Infof("Dump of scheduling queue:\n%s", podData.String())
}

// printNodeInfo writes parts of NodeInfo to a string.
func (d *CacheDumper) printNodeInfo(n *framework.NodeInfo) string {
	var nodeData strings.Builder
	nodeData.WriteString(fmt.Sprintf("\nNode name: %+v\nRequested Resources: %+v\nAllocatable Resources:%+v\nScheduled Pods(number: %v):\n",
		n.Node().Name, n.Requested, n.Allocatable, len(n.Pods)))
	// Dumping Pod Info
	for _, p := range n.Pods {
		nodeData.WriteString(printPod(p.Pod))
	}
	// Dumping nominated pods info on the node
	nominatedPods := d.podQueue.NominatedPodsForNode(n.Node().Name)
	if len(nominatedPods) != 0 {
		nodeData.WriteString(fmt.Sprintf("Nominated Pods(number: %v):\n", len(nominatedPods)))
		for _, p := range nominatedPods {
			nodeData.WriteString(printPod(p))
		}
	}
	return nodeData.String()
}

// printPod writes parts of a Pod object to a string.
func printPod(p *v1.Pod) string {
	return fmt.Sprintf("name: %v, namespace: %v, uid: %v, phase: %v, nominated node: %v\n", p.Name, p.Namespace, p.UID, p.Status.Phase, p.Status.NominatedNodeName)
}
