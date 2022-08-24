/*
Copyright 2019 The Kubernetes Authors.

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

package selectorspread

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/pointer"
)

var (
	rcKind = v1.SchemeGroupVersion.WithKind("ReplicationController")
	rsKind = apps.SchemeGroupVersion.WithKind("ReplicaSet")
	ssKind = apps.SchemeGroupVersion.WithKind("StatefulSet")
)

func controllerRef(name string, gvk schema.GroupVersionKind) []metav1.OwnerReference {
	return []metav1.OwnerReference{
		{
			APIVersion: gvk.GroupVersion().String(),
			Kind:       gvk.Kind,
			Name:       name,
			Controller: pointer.Bool(true),
		},
	}
}

func TestSelectorSpreadScore(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	zone1Spec := v1.PodSpec{
		NodeName: "node1",
	}
	zone2Spec := v1.PodSpec{
		NodeName: "node2",
	}
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []string
		rcs          []*v1.ReplicationController
		rss          []*apps.ReplicaSet
		services     []*v1.Service
		sss          []*apps.StatefulSet
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			pod:          new(v1.Pod),
			nodes:        []string{"node1", "node2"},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: framework.MaxNodeScore}},
			name:         "nothing scheduled",
		},
		{
			pod:          st.MakePod().Labels(labels1).Obj(),
			pods:         []*v1.Pod{st.MakePod().Node("node1").Obj()},
			nodes:        []string{"node1", "node2"},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: framework.MaxNodeScore}},
			name:         "no services",
		},
		{
			pod:          st.MakePod().Labels(labels1).Obj(),
			pods:         []*v1.Pod{st.MakePod().Labels(labels2).Node("node1").Obj()},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: framework.MaxNodeScore}},
			name:         "different services",
		},
		{
			pod: st.MakePod().Labels(labels1).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Labels(labels2).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node2").Obj(),
			},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: 0}},
			name:         "two pods, one service pod",
		},
		{
			pod: st.MakePod().Labels(labels1).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Labels(labels2).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node1").Namespace(metav1.NamespaceDefault).Obj(),
				st.MakePod().Labels(labels1).Node("node1").Namespace("ns1").Obj(),
				st.MakePod().Labels(labels1).Node("node2").Obj(),
				st.MakePod().Labels(labels2).Node("node2").Obj(),
			},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: 0}},
			name:         "five pods, one service pod in no namespace",
		},
		{
			pod: st.MakePod().Labels(labels1).Namespace(metav1.NamespaceDefault).Obj(),
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: metav1.NamespaceDefault}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
			},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: 0}},
			name:         "four pods, one service pod in default namespace",
		},
		{
			pod: st.MakePod().Labels(labels1).Namespace("ns1").Obj(),
			pods: []*v1.Pod{
				st.MakePod().Labels(labels2).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node1").Namespace(metav1.NamespaceDefault).Obj(),
				st.MakePod().Labels(labels1).Node("node1").Namespace("ns2").Obj(),
				st.MakePod().Labels(labels1).Node("node2").Namespace("ns1").Obj(),
				st.MakePod().Labels(labels2).Node("node2").Obj(),
			},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns1"}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: 0}},
			name:         "five pods, one service pod in specific namespace",
		},
		{
			pod: st.MakePod().Labels(labels1).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Labels(labels2).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node2").Obj(),
			},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "three pods, two service pods on different nodes",
		},
		{
			pod: st.MakePod().Labels(labels1).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Labels(labels2).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node2").Obj(),
				st.MakePod().Labels(labels1).Node("node2").Obj(),
			},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 50}, {Name: "node2", Score: 0}},
			name:         "four pods, three service pods",
		},
		{
			pod: st.MakePod().Labels(labels1).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Labels(labels2).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node1").Obj(),
				st.MakePod().Labels(labels1).Node("node2").Obj(),
			},
			nodes:        []string{"node1", "node2"},
			services:     []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 50}},
			name:         "service with partial pod label matches",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Node("node1").Namespace(metav1.NamespaceDefault).Labels(labels2).Obj(),
				st.MakePod().Node("node1").Namespace(metav1.NamespaceDefault).Labels(labels1).Obj(),
				st.MakePod().Node("node2").Namespace(metav1.NamespaceDefault).Labels(labels1).Obj(),
			},
			nodes: []string{"node1", "node2"},
			rcs: []*v1.ReplicationController{
				{ObjectMeta: metav1.ObjectMeta{Name: "rc1", Namespace: metav1.NamespaceDefault}, Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}},
				{ObjectMeta: metav1.ObjectMeta{Name: "rc2", Namespace: metav1.NamespaceDefault}, Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"bar": "foo"}}},
			},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "s1", Namespace: metav1.NamespaceDefault}, Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			// "baz=blah" matches both labels1 and labels2, and "foo=bar" matches only labels 1. This means that we assume that we want to
			// do spreading pod2 and pod3 and not pod1.
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "service with partial pod label matches with service and replication controller",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Node("node1").Namespace(metav1.NamespaceDefault).Labels(labels2).Obj(),
				st.MakePod().Node("node1").Namespace(metav1.NamespaceDefault).Labels(labels1).Obj(),
				st.MakePod().Node("node2").Namespace(metav1.NamespaceDefault).Labels(labels1).Obj(),
			},
			nodes:    []string{"node1", "node2"},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "s1", Namespace: metav1.NamespaceDefault}, Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			rss: []*apps.ReplicaSet{
				{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "rs1"}, Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}},
				{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "rs2"}, Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"bar": "foo"}}}},
			},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "service with partial pod label matches with service and replica set",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Node("node1").Namespace(metav1.NamespaceDefault).Labels(labels2).Obj(),
				st.MakePod().Node("node1").Namespace(metav1.NamespaceDefault).Labels(labels1).Obj(),
				st.MakePod().Node("node2").Namespace(metav1.NamespaceDefault).Labels(labels1).Obj(),
			},
			nodes:    []string{"node1", "node2"},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "s1", Namespace: metav1.NamespaceDefault}, Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			sss: []*apps.StatefulSet{
				{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "ss1"}, Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}},
				{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "ss2"}, Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"bar": "foo"}}}},
			},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "service with partial pod label matches with service and statefulset",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(map[string]string{"foo": "bar", "bar": "foo"}).OwnerReference("rc3", rcKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("rc2", rcKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
			},
			nodes: []string{"node1", "node2"},
			rcs: []*v1.ReplicationController{{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "rc3"},
				Spec:       v1.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "s1", Namespace: metav1.NamespaceDefault}, Spec: v1.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			// Taken together Service and Replication Controller should match no pods.
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: framework.MaxNodeScore}},
			name:         "disjoined service and replication controller matches no pods",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(map[string]string{"foo": "bar", "bar": "foo"}).OwnerReference("rs3", rsKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("rs2", rsKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
			},
			nodes:    []string{"node1", "node2"},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "s1", Namespace: metav1.NamespaceDefault}, Spec: v1.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			rss: []*apps.ReplicaSet{
				{ObjectMeta: metav1.ObjectMeta{Name: "rs3", Namespace: metav1.NamespaceDefault}, Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: framework.MaxNodeScore}},
			name:         "disjoined service and replica set matches no pods",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(map[string]string{"foo": "bar", "bar": "foo"}).OwnerReference("ss3", ssKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("ss2", ssKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
			},
			nodes:    []string{"node1", "node2"},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "s1", Namespace: metav1.NamespaceDefault}, Spec: v1.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			sss: []*apps.StatefulSet{
				{ObjectMeta: metav1.ObjectMeta{Name: "ss3", Namespace: metav1.NamespaceDefault}, Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: framework.MaxNodeScore}, {Name: "node2", Score: framework.MaxNodeScore}},
			name:         "disjoined service and stateful set matches no pods",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("rc2", rcKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
			},
			nodes: []string{"node1", "node2"},
			rcs:   []*v1.ReplicationController{{ObjectMeta: metav1.ObjectMeta{Name: "rc1", Namespace: metav1.NamespaceDefault}, Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			// Both Nodes have one pod from the given RC, hence both get 0 score.
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "Replication controller with partial pod label matches",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("rs2", rsKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
			},
			nodes: []string{"node1", "node2"},
			rss:   []*apps.ReplicaSet{{ObjectMeta: metav1.ObjectMeta{Name: "rs1", Namespace: metav1.NamespaceDefault}, Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "Replica set with partial pod label matches",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("ss2", ssKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
			},
			nodes: []string{"node1", "node2"},
			sss:   []*apps.StatefulSet{{ObjectMeta: metav1.ObjectMeta{Name: "ss1", Namespace: metav1.NamespaceDefault}, Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use StatefulSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "StatefulSet with partial pod label matches",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("rc3", rcKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("rc2", rcKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("rc1", rcKind).Obj(),
			},
			nodes:        []string{"node1", "node2"},
			rcs:          []*v1.ReplicationController{{ObjectMeta: metav1.ObjectMeta{Name: "rc3", Namespace: metav1.NamespaceDefault}, Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 50}},
			name:         "Another replication controller with partial pod label matches",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("rs3", rsKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("rs2", rsKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("rs1", rsKind).Obj(),
			},
			nodes: []string{"node1", "node2"},
			rss:   []*apps.ReplicaSet{{ObjectMeta: metav1.ObjectMeta{Name: "rs3", Namespace: metav1.NamespaceDefault}, Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"baz": "blah"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 50}},
			name:         "Another replication set with partial pod label matches",
		},
		{
			pod: st.MakePod().Namespace(metav1.NamespaceDefault).Labels(labels1).OwnerReference("ss3", ssKind).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels2).OwnerReference("ss2", ssKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node1").Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
				st.MakePod().Namespace(metav1.NamespaceDefault).Node("node2").Labels(labels1).OwnerReference("ss1", ssKind).Obj(),
			},
			nodes: []string{"node1", "node2"},
			sss:   []*apps.StatefulSet{{ObjectMeta: metav1.ObjectMeta{Name: "ss3", Namespace: metav1.NamespaceDefault}, Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"baz": "blah"}}}}},
			// We use StatefulSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 50}},
			name:         "Another stateful set with partial pod label matches",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       metav1.NamespaceDefault,
					Labels:          labels1,
					OwnerReferences: controllerRef("ss1", ssKind),
				},
				Spec: v1.PodSpec{
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
						{
							MaxSkew:           1,
							TopologyKey:       "foo",
							WhenUnsatisfiable: v1.DoNotSchedule,
						},
					},
				},
			},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Labels: labels2, OwnerReferences: controllerRef("ss2", ssKind)}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Labels: labels1, OwnerReferences: controllerRef("ss1", ssKind)}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Labels: labels1, OwnerReferences: controllerRef("ss1", ssKind)}},
			},
			nodes:        []string{"node1", "node2"},
			sss:          []*apps.StatefulSet{{ObjectMeta: metav1.ObjectMeta{Name: "ss1", Namespace: metav1.NamespaceDefault}, Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"baz": "blah"}}}}},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 0}, {Name: "node2", Score: 0}},
			name:         "Another statefulset with TopologySpreadConstraints set in pod",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := makeNodeList(test.nodes)
			snapshot := cache.NewSnapshot(test.pods, nodes)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			informerFactory, err := populateAndStartInformers(ctx, test.rcs, test.rss, test.services, test.sss)
			if err != nil {
				t.Errorf("error creating informerFactory: %+v", err)
			}
			fh, err := frameworkruntime.NewFramework(nil, nil, ctx.Done(), frameworkruntime.WithSnapshotSharedLister(snapshot), frameworkruntime.WithInformerFactory(informerFactory))
			if err != nil {
				t.Errorf("error creating new framework handle: %+v", err)
			}

			state := framework.NewCycleState()

			pl, err := New(nil, fh)
			if err != nil {
				t.Fatal(err)
			}
			plugin := pl.(*SelectorSpread)

			status := plugin.PreScore(ctx, state, test.pod, nodes)
			if !status.IsSuccess() {
				t.Fatalf("unexpected error: %v", status)
			}

			var gotList framework.NodeScoreList
			for _, nodeName := range test.nodes {
				score, status := plugin.Score(ctx, state, test.pod, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}

			status = plugin.ScoreExtensions().NormalizeScore(ctx, state, test.pod, gotList)
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}

			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedList, gotList)
			}
		})
	}
}

func buildPod(nodeName string, labels map[string]string, ownerRefs []metav1.OwnerReference) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Labels: labels, OwnerReferences: ownerRefs},
		Spec:       v1.PodSpec{NodeName: nodeName},
	}
}

func TestZoneSelectorSpreadPriority(t *testing.T) {
	labels1 := map[string]string{
		"label1": "l1",
		"baz":    "blah",
	}
	labels2 := map[string]string{
		"label2": "l2",
		"baz":    "blah",
	}

	const nodeMachine1Zone1 = "node1.zone1"
	const nodeMachine1Zone2 = "node1.zone2"
	const nodeMachine2Zone2 = "node2.zone2"
	const nodeMachine1Zone3 = "node1.zone3"
	const nodeMachine2Zone3 = "node2.zone3"
	const nodeMachine3Zone3 = "node3.zone3"

	buildNodeLabels := func(failureDomain string) map[string]string {
		labels := map[string]string{
			v1.LabelTopologyZone: failureDomain,
		}
		return labels
	}
	labeledNodes := map[string]map[string]string{
		nodeMachine1Zone1: buildNodeLabels("zone1"),
		nodeMachine1Zone2: buildNodeLabels("zone2"),
		nodeMachine2Zone2: buildNodeLabels("zone2"),
		nodeMachine1Zone3: buildNodeLabels("zone3"),
		nodeMachine2Zone3: buildNodeLabels("zone3"),
		nodeMachine3Zone3: buildNodeLabels("zone3"),
	}

	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		rcs          []*v1.ReplicationController
		rss          []*apps.ReplicaSet
		services     []*v1.Service
		sss          []*apps.StatefulSet
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			pod: new(v1.Pod),
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "nothing scheduled",
		},
		{
			pod:  buildPod("", labels1, nil),
			pods: []*v1.Pod{buildPod(nodeMachine1Zone1, nil, nil)},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "no services",
		},
		{
			pod:      buildPod("", labels1, nil),
			pods:     []*v1.Pod{buildPod(nodeMachine1Zone1, labels2, nil)},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "different services",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels2, nil),
			},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "two pods, 0 matching",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
			},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: 0},  // Already have pod on node
				{Name: nodeMachine2Zone2, Score: 33}, // Already have pod in zone
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "two pods, 1 matching (in z2)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels1, nil),
				buildPod(nodeMachine1Zone3, labels2, nil),
				buildPod(nodeMachine2Zone3, labels1, nil),
			},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine1Zone3, Score: 66}, // Pod in zone
				{Name: nodeMachine2Zone3, Score: 33}, // Pod on node
				{Name: nodeMachine3Zone3, Score: 66}, // Pod in zone
			},
			name: "five pods, 3 matching (z2=2, z3=1)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels1, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels2, nil),
				buildPod(nodeMachine1Zone3, labels1, nil),
			},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: 0},  // Pod on node
				{Name: nodeMachine1Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone2, Score: 33}, // Pod in zone
				{Name: nodeMachine1Zone3, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone3, Score: 33}, // Pod in zone
				{Name: nodeMachine3Zone3, Score: 33}, // Pod in zone
			},
			name: "four pods, 3 matching (z1=1, z2=1, z3=1)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels1, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels2, nil),
				buildPod(nodeMachine1Zone3, labels1, nil),
			},
			services: []*v1.Service{{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "s1"}, Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: 33}, // Pod on node
				{Name: nodeMachine1Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone2, Score: 0},  // Pod in zone
				{Name: nodeMachine1Zone3, Score: 33}, // Pod on node
				{Name: nodeMachine2Zone3, Score: 66}, // Pod in zone
				{Name: nodeMachine3Zone3, Score: 66}, // Pod in zone
			},
			name: "five pods, 4 matching (z1=1, z2=2, z3=1)",
		},
		{
			pod: buildPod("", labels1, controllerRef("rc1", rcKind)),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone3, labels1, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine1Zone3, labels1, nil),
			},
			rcs: []*v1.ReplicationController{
				{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "rc1"}, Spec: v1.ReplicationControllerSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				// Note that because we put two pods on the same node (nodeMachine1Zone3),
				// the values here are questionable for zone2, in particular for nodeMachine1Zone2.
				// However they kind of make sense; zone1 is still most-highly favored.
				// zone3 is in general least favored, and m1.z3 particularly low priority.
				// We would probably prefer to see a bigger gap between putting a second
				// pod on m1.z2 and putting a pod on m2.z2, but the ordering is correct.
				// This is also consistent with what we have already.
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore}, // No pods in zone
				{Name: nodeMachine1Zone2, Score: 50},                     // Pod on node
				{Name: nodeMachine2Zone2, Score: 66},                     // Pod in zone
				{Name: nodeMachine1Zone3, Score: 0},                      // Two pods on node
				{Name: nodeMachine2Zone3, Score: 33},                     // Pod in zone
				{Name: nodeMachine3Zone3, Score: 33},                     // Pod in zone
			},
			name: "Replication controller spreading (z1=0, z2=1, z3=2)",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := makeLabeledNodeList(labeledNodes)
			snapshot := cache.NewSnapshot(test.pods, nodes)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			informerFactory, err := populateAndStartInformers(ctx, test.rcs, test.rss, test.services, test.sss)
			if err != nil {
				t.Errorf("error creating informerFactory: %+v", err)
			}
			fh, err := frameworkruntime.NewFramework(nil, nil, wait.NeverStop, frameworkruntime.WithSnapshotSharedLister(snapshot), frameworkruntime.WithInformerFactory(informerFactory))
			if err != nil {
				t.Errorf("error creating new framework handle: %+v", err)
			}

			pl, err := New(nil, fh)
			if err != nil {
				t.Fatal(err)
			}
			plugin := pl.(*SelectorSpread)

			state := framework.NewCycleState()
			status := plugin.PreScore(ctx, state, test.pod, nodes)
			if !status.IsSuccess() {
				t.Fatalf("unexpected error: %v", status)
			}

			var gotList framework.NodeScoreList
			for _, n := range nodes {
				nodeName := n.ObjectMeta.Name
				score, status := plugin.Score(ctx, state, test.pod, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}

			status = plugin.ScoreExtensions().NormalizeScore(ctx, state, test.pod, gotList)
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}

			sortNodeScoreList(test.expectedList)
			sortNodeScoreList(gotList)
			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedList, gotList)
			}
		})
	}
}

func populateAndStartInformers(ctx context.Context, rcs []*v1.ReplicationController, rss []*apps.ReplicaSet, services []*v1.Service, sss []*apps.StatefulSet) (informers.SharedInformerFactory, error) {
	objects := make([]runtime.Object, 0, len(rcs)+len(rss)+len(services)+len(sss))
	for _, rc := range rcs {
		objects = append(objects, rc.DeepCopyObject())
	}
	for _, rs := range rss {
		objects = append(objects, rs.DeepCopyObject())
	}
	for _, service := range services {
		objects = append(objects, service.DeepCopyObject())
	}
	for _, ss := range sss {
		objects = append(objects, ss.DeepCopyObject())
	}
	client := clientsetfake.NewSimpleClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	// Because we use an informer factory, we need to make requests for the specific informers we want before calling Start()
	_ = informerFactory.Core().V1().Services().Lister()
	_ = informerFactory.Core().V1().ReplicationControllers().Lister()
	_ = informerFactory.Apps().V1().ReplicaSets().Lister()
	_ = informerFactory.Apps().V1().StatefulSets().Lister()
	informerFactory.Start(ctx.Done())
	caches := informerFactory.WaitForCacheSync(ctx.Done())
	for _, synced := range caches {
		if !synced {
			return nil, fmt.Errorf("error waiting for informer cache sync")
		}
	}
	return informerFactory, nil
}

func makeLabeledNodeList(nodeMap map[string]map[string]string) []*v1.Node {
	nodes := make([]*v1.Node, 0, len(nodeMap))
	for nodeName, labels := range nodeMap {
		nodes = append(nodes, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName, Labels: labels}})
	}
	return nodes
}

func makeNodeList(nodeNames []string) []*v1.Node {
	nodes := make([]*v1.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		nodes = append(nodes, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}
	return nodes
}

func sortNodeScoreList(out framework.NodeScoreList) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].Name < out[j].Name
		}
		return out[i].Score < out[j].Score
	})
}
