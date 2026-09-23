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

	v1 "k8s.io/api/core/v1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
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
	nodes := make(nodeDumps, 0, len(dump.Nodes))
	for name, nodeInfo := range dump.Nodes {
		nodes = append(nodes, d.dumpNodeInfo(name, nodeInfo))
	}
	logger.Info("Dump of cached NodeInfo", "nodes", nodes)
}

// dumpSchedulingQueue writes pods in the scheduling queue to the scheduler logs.
func (d *CacheDumper) dumpSchedulingQueue(logger klog.Logger) {
	pendingPods, s := d.podQueue.PendingPods()
	pods := make(podDumps, 0, len(pendingPods))
	for _, p := range pendingPods {
		pods = append(pods, dumpPod(p))
	}
	logger.Info("Dump of scheduling queue", "summary", s, "pods", pods)
}

// dumpNodeInfo collects the parts of NodeInfo that are useful for debugging.
func (d *CacheDumper) dumpNodeInfo(name string, n *framework.NodeInfo) nodeDump {
	pods := make(podDumps, 0, len(n.Pods))
	for _, p := range n.Pods {
		pods = append(pods, dumpPod(p.GetPod()))
	}
	nominated := d.podQueue.NominatedPodsForNode(name)
	nominatedPods := make(podDumps, 0, len(nominated))
	for _, pi := range nominated {
		nominatedPods = append(nominatedPods, dumpPod(pi.GetPod()))
	}
	return nodeDump{
		Name:          name,
		Deleted:       n.Node() == nil,
		Requested:     newResourceDump(n.Requested),
		Allocatable:   newResourceDump(n.Allocatable),
		Pods:          pods,
		NominatedPods: nominatedPods,
	}
}

// dumpPod collects the parts of a Pod object that are useful for debugging.
func dumpPod(p *v1.Pod) podDump {
	return podDump{
		Name:          p.Name,
		Namespace:     p.Namespace,
		UID:           string(p.UID),
		Phase:         p.Status.Phase,
		NominatedNode: p.Status.NominatedNodeName,
	}
}

// resourceDump is a loggable view of framework.Resource.
type resourceDump struct {
	MilliCPU         int64                     `json:"milliCPU"`
	Memory           int64                     `json:"memory"`
	EphemeralStorage int64                     `json:"ephemeralStorage"`
	AllowedPodNumber int                       `json:"allowedPodNumber"`
	ScalarResources  map[v1.ResourceName]int64 `json:"scalarResources,omitempty"`
}

func newResourceDump(r *framework.Resource) resourceDump {
	if r == nil {
		return resourceDump{}
	}
	return resourceDump{
		MilliCPU:         r.MilliCPU,
		Memory:           r.Memory,
		EphemeralStorage: r.EphemeralStorage,
		AllowedPodNumber: r.AllowedPodNumber,
		ScalarResources:  r.ScalarResources,
	}
}

// podDump is a loggable view of a Pod.
type podDump struct {
	Name          string      `json:"name"`
	Namespace     string      `json:"namespace"`
	UID           string      `json:"uid"`
	Phase         v1.PodPhase `json:"phase"`
	NominatedNode string      `json:"nominatedNode,omitempty"`
}

func (p podDump) String() string {
	return fmt.Sprintf("name: %v, namespace: %v, uid: %v, phase: %v, nominated node: %v",
		p.Name, p.Namespace, p.UID, p.Phase, p.NominatedNode)
}

// nodeDump is a loggable view of framework.NodeInfo.
type nodeDump struct {
	Name          string       `json:"name"`
	Deleted       bool         `json:"deleted"`
	Requested     resourceDump `json:"requested"`
	Allocatable   resourceDump `json:"allocatable"`
	Pods          podDumps     `json:"pods"`
	NominatedPods podDumps     `json:"nominatedPods,omitempty"`
}

func (n nodeDump) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Node name: %s\nDeleted: %t\nRequested Resources: %+v\nAllocatable Resources: %+v\nScheduled Pods(number: %v):\n",
		n.Name, n.Deleted, n.Requested, n.Allocatable, len(n.Pods))
	for _, p := range n.Pods {
		fmt.Fprintf(&b, "%s\n", p.String())
	}
	if len(n.NominatedPods) != 0 {
		fmt.Fprintf(&b, "Nominated Pods(number: %v):\n", len(n.NominatedPods))
		for _, p := range n.NominatedPods {
			fmt.Fprintf(&b, "%s\n", p.String())
		}
	}
	return b.String()
}

// podDumps renders as readable multi-line text when logged in text format
// (via String) and as a JSON array of objects in JSON format (via MarshalLog).
type podDumps []podDump

func (ps podDumps) String() string {
	var b strings.Builder
	for _, p := range ps {
		fmt.Fprintf(&b, "%s\n", p.String())
	}
	return b.String()
}

// MarshalLog returns a plain slice so that the JSON logging backend encodes it
// as an array of structs instead of as the String() representation.
func (ps podDumps) MarshalLog() any {
	return []podDump(ps)
}

// nodeDumps renders as readable multi-line text when logged in text format
// (via String) and as a JSON array of objects in JSON format (via MarshalLog).
type nodeDumps []nodeDump

func (ns nodeDumps) String() string {
	entries := make([]string, 0, len(ns))
	for _, n := range ns {
		entries = append(entries, n.String())
	}
	// Extra blank line added between node entries for readability.
	return strings.Join(entries, "\n\n")
}

// MarshalLog returns a plain slice so that the JSON logging backend encodes it
// as an array of structs instead of as the String() representation.
func (ns nodeDumps) MarshalLog() any {
	return []nodeDump(ns)
}
