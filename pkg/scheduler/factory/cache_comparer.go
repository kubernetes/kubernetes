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

package factory

import (
	"sort"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
)

type cacheComparer struct {
	nodeLister corelisters.NodeLister
	podLister  corelisters.PodLister
	cache      schedulercache.Cache

	compareStrategy
}

func (c *cacheComparer) Compare() error {
	glog.V(3).Info("cache comparer started")
	defer glog.V(3).Info("cache comparer finished")

	nodes, err := c.nodeLister.List(labels.Everything())
	if err != nil {
		return err
	}

	snapshot := c.cache.Snapshot()

	if missed, redundant := c.CompareNodes(nodes, snapshot.Nodes); len(missed)+len(redundant) != 0 {
		glog.Warningf("cache mismatch: missed nodes: %s; redundant nodes: %s", missed, redundant)
	}

	return nil
}

type compareStrategy struct {
}

func (c compareStrategy) CompareNodes(nodes []*v1.Node, nodeinfos map[string]*schedulercache.NodeInfo) (missed, redundant []string) {
	missed, redundant = []string{}, []string{}

	actual := []string{}
	for _, node := range nodes {
		actual = append(actual, node.Name)
	}

	cached := []string{}
	for nodeName := range nodeinfos {
		cached = append(cached, nodeName)
	}

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
