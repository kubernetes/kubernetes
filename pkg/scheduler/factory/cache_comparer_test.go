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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
)

func TestCompareNodes(t *testing.T) {
	compare := compareStrategy{}

	tests := []struct {
		actual    []string
		cached    []string
		missing   []string
		redundant []string
	}{
		{
			actual:    []string{"foo", "bar"},
			cached:    []string{"bar", "foo", "foobar"},
			missing:   []string{},
			redundant: []string{"foobar"},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foo"},
			missing:   []string{"foobar"},
			redundant: []string{},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foobar", "foo"},
			missing:   []string{},
			redundant: []string{},
		},
	}

	for _, test := range tests {
		nodes := []*v1.Node{}
		for _, nodeName := range test.actual {
			node := &v1.Node{}
			node.Name = nodeName
			nodes = append(nodes, node)
		}

		nodeInfo := make(map[string]*schedulercache.NodeInfo)
		for _, nodeName := range test.cached {
			nodeInfo[nodeName] = &schedulercache.NodeInfo{}
		}

		m, r := compare.CompareNodes(nodes, nodeInfo)

		if !reflect.DeepEqual(m, test.missing) {
			t.Errorf("missing expected to be %s; got %s", test.missing, m)
		}

		if !reflect.DeepEqual(r, test.redundant) {
			t.Errorf("redundant expected to be %s; got %s", test.redundant, r)
		}
	}
}

func TestComparePods(t *testing.T) {
	compare := compareStrategy{}

	tests := []struct {
		actual    []string
		cached    []string
		queued    []string
		missing   []string
		redundant []string
	}{
		{
			actual:    []string{"foo", "bar"},
			cached:    []string{"bar", "foo", "foobar"},
			queued:    []string{},
			missing:   []string{},
			redundant: []string{"foobar"},
		},
		{
			actual:    []string{"foo", "bar"},
			cached:    []string{"foo", "foobar"},
			queued:    []string{"bar"},
			missing:   []string{},
			redundant: []string{"foobar"},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foo"},
			queued:    []string{},
			missing:   []string{"foobar"},
			redundant: []string{},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"foo"},
			queued:    []string{"bar"},
			missing:   []string{"foobar"},
			redundant: []string{},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foobar", "foo"},
			queued:    []string{},
			missing:   []string{},
			redundant: []string{},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"foobar", "foo"},
			queued:    []string{"bar"},
			missing:   []string{},
			redundant: []string{},
		},
	}

	for _, test := range tests {
		pods := []*v1.Pod{}
		for _, uid := range test.actual {
			pod := &v1.Pod{}
			pod.UID = types.UID(uid)
			pods = append(pods, pod)
		}

		queuedPods := []*v1.Pod{}
		for _, uid := range test.queued {
			pod := &v1.Pod{}
			pod.UID = types.UID(uid)
			queuedPods = append(queuedPods, pod)
		}

		nodeInfo := make(map[string]*schedulercache.NodeInfo)
		for _, uid := range test.cached {
			pod := &v1.Pod{}
			pod.UID = types.UID(uid)
			pod.Namespace = "ns"
			pod.Name = uid

			nodeInfo[uid] = schedulercache.NewNodeInfo(pod)
		}

		m, r := compare.ComparePods(pods, queuedPods, nodeInfo)

		if !reflect.DeepEqual(m, test.missing) {
			t.Errorf("missing expected to be %s; got %s", test.missing, m)
		}

		if !reflect.DeepEqual(r, test.redundant) {
			t.Errorf("redundant expected to be %s; got %s", test.redundant, r)
		}
	}
}

func TestComparePdbs(t *testing.T) {
	compare := compareStrategy{}

	tests := []struct {
		actual    []string
		cached    []string
		missing   []string
		redundant []string
	}{
		{
			actual:    []string{"foo", "bar"},
			cached:    []string{"bar", "foo", "foobar"},
			missing:   []string{},
			redundant: []string{"foobar"},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foo"},
			missing:   []string{"foobar"},
			redundant: []string{},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foobar", "foo"},
			missing:   []string{},
			redundant: []string{},
		},
	}

	for _, test := range tests {
		pdbs := []*policy.PodDisruptionBudget{}
		for _, uid := range test.actual {
			pdb := &policy.PodDisruptionBudget{}
			pdb.UID = types.UID(uid)
			pdbs = append(pdbs, pdb)
		}

		cache := make(map[string]*policy.PodDisruptionBudget)
		for _, uid := range test.cached {
			pdb := &policy.PodDisruptionBudget{}
			pdb.UID = types.UID(uid)
			cache[uid] = pdb
		}

		m, r := compare.ComparePdbs(pdbs, cache)

		if !reflect.DeepEqual(m, test.missing) {
			t.Errorf("missing expected to be %s; got %s", test.missing, m)
		}

		if !reflect.DeepEqual(r, test.redundant) {
			t.Errorf("redundant expected to be %s; got %s", test.redundant, r)
		}
	}
}
