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
	"sort"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
)

// CacheComparer is an implementation of the Scheduler's cache comparer.
type CacheComparer struct {
	NodeLister corelisters.NodeLister
	PodLister  corelisters.PodLister
	Cache      internalcache.Cache
	PodQueue   internalqueue.SchedulingQueue
}

// Compare compares the nodes and pods of NodeLister with Cache.Snapshot.
func (c *CacheComparer) Compare() error {
	klog.V(3).InfoS("Cache comparer started")
	defer klog.V(3).InfoS("Cache comparer finished")

	nodes, err := c.NodeLister.List(labels.Everything())
	if err != nil {
		return err
	}

	pods, err := c.PodLister.List(labels.Everything())
	if err != nil {
		return err
	}

	dump := c.Cache.Dump()

	pendingPods := c.PodQueue.PendingPods()

	if missed, redundant := c.CompareNodes(nodes, dump.Nodes); len(missed)+len(redundant) != 0 {
		klog.InfoS("Cache mismatch", "missedNodes", missed, "redundantNodes", redundant)
	}

	if missed, redundant := c.ComparePods(pods, pendingPods, dump.Nodes); len(missed)+len(redundant) != 0 {
		klog.InfoS("Cache mismatch", "missedPods", missed, "redundantPods", redundant)
	}

	return nil
}

// CompareNodes compares actual nodes with cached nodes.
func (c *CacheComparer) CompareNodes(nodes []*v1.Node, nodeinfos map[string]*framework.NodeInfo) (missed, redundant []string) {
	actual := []string{}
	for _, node := range nodes {
		actual = append(actual, node.Name)
	}

	cached := []string{}
	for nodeName := range nodeinfos {
		cached = append(cached, nodeName)
	}

	return compareStrings(actual, cached)
}

// ComparePods compares actual pods with cached pods.
func (c *CacheComparer) ComparePods(pods, waitingPods []*v1.Pod, nodeinfos map[string]*framework.NodeInfo) (missed, redundant []string) {
	actual := []string{}
	for _, pod := range pods {
		actual = append(actual, string(pod.UID))
	}

	cached := []string{}
	for _, nodeinfo := range nodeinfos {
		for _, p := range nodeinfo.Pods {
			cached = append(cached, string(p.Pod.UID))
		}
	}
	for _, pod := range waitingPods {
		cached = append(cached, string(pod.UID))
	}

	return compareStrings(actual, cached)
}

func compareStrings(actual, cached []string) (missed, redundant []string) {
	missed, redundant = []string{}, []string{}

	sort.Strings(actual)
	sort.Strings(cached)

	compare := func(i, j int) int {
		if i == len(actual) {
			return 1
		} else if j == len(cached) {
			return -1
		}
		return strings.Compare(actual[i], cached[j])
	}

	for i, j := 0, 0; i < len(actual) || j < len(cached); {
		switch compare(i, j) {
		case 0:
			i++
			j++
		case -1:
			missed = append(missed, actual[i])
			i++
		case 1:
			redundant = append(redundant, cached[j])
			j++
		}
	}

	return
}
