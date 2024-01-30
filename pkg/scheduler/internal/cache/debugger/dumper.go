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
	"k8s.io/kubernetes/pkg/scheduler/framework"
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
func (d *CacheDumper) DumpAll(logger klog.Logger) {
	d.dumpNodes(logger)
	d.dumpSchedulingQueue(logger)
}

// dumpNodes writes NodeInfo to the scheduler logs.
func (d *CacheDumper) dumpNodes(logger klog.Logger) {
	dump := d.cache.Dump()
	nodeInfos := make([]string, 0, len(dump.Nodes))
	for name, nodeInfo := range dump.Nodes {
		nodeInfos = append(nodeInfos, d.printNodeInfo(name, nodeInfo))
	}
	// Extra blank line added between node entries for readability.
	logger.Info("Dump of cached NodeInfo", "nodes", strings.Join(nodeInfos, "\n\n"))
}

// dumpSchedulingQueue writes pods in the scheduling queue to the scheduler logs.
func (d *CacheDumper) dumpSchedulingQueue(logger klog.Logger) {
	pendingPods, s := d.podQueue.PendingPods()
	var podData strings.Builder
	for _, p := range pendingPods {
		podData.WriteString(printPod(p))
	}
	logger.Info("Dump of scheduling queue", "summary", s, "pods", podData.String())
}

// printNodeInfo writes parts of NodeInfo to a string.
func (d *CacheDumper) printNodeInfo(name string, n *framework.NodeInfo) string {
	var nodeData strings.Builder
	nodeData.WriteString(fmt.Sprintf("Node name: %s\nDeleted: %t\nRequested Resources: %+v\nAllocatable Resources:%+v\nScheduled Pods(number: %v):\n",
		name, n.Node() == nil, n.Requested, n.Allocatable, len(n.Pods)))
	// Dumping Pod Info
	for _, p := range n.Pods {
		nodeData.WriteString(printPod(p.Pod))
	}
	// Dumping nominated pods info on the node
	nominatedPodInfos := d.podQueue.NominatedPodsForNode(name)
	if len(nominatedPodInfos) != 0 {
		nodeData.WriteString(fmt.Sprintf("Nominated Pods(number: %v):\n", len(nominatedPodInfos)))
		for _, pi := range nominatedPodInfos {
			nodeData.WriteString(printPod(pi.Pod))
		}
	}
	return nodeData.String()
}

// printPod writes parts of a Pod object to a string.
func printPod(p *v1.Pod) string {
	return fmt.Sprintf("name: %v, namespace: %v, uid: %v, phase: %v, nominated node: %v\n", p.Name, p.Namespace, p.UID, p.Status.Phase, p.Status.NominatedNodeName)
}
