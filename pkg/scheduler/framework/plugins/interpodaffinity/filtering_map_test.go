/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package interpodaffinity

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestFilteringExistingPodCacheProxyPodEvents(t *testing.T) {
	// 创建测试环境
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	podInformer := informerFactory.Core().V1().Pods()
	namespaceInformer := informerFactory.Core().V1().Namespaces()
	nodeInformer := informerFactory.Core().V1().Nodes()

	// 创建测试节点
	node := st.MakeNode().Name("node1").Label("region", "r1").Label("zone", "z1").Obj()
	_, err := client.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建节点失败: %v", err)
	}

	// 创建测试命名空间
	ns := st.MakeNamespace().Name("testns").Obj()
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建命名空间失败: %v", err)
	}

	// 创建一个模拟的SharedLister
	snapshot := cache.NewSnapshot(nil, []*v1.Node{node})

	// 创建测试用的proxy
	proxy := NewFilteringExistingPodCacheProxy(context.TODO(),
		podInformer,
		podInformer.Lister(),
		namespaceInformer.Lister(),
		namespaceInformer,
		nodeInformer,
		snapshot,
	)

	// 启动informer工厂
	_, cancel := startInformerFactory(informerFactory)
	defer cancel()
	// 创建带有反亲和性的Pod
	podWithAntiAffinity := st.MakePod().Namespace("testns").Name("pod1").Label("app", "test").
		PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).Node("node1").Obj()
	podWithAntiAffinity, err = client.CoreV1().Pods(podWithAntiAffinity.Namespace).Create(context.TODO(), podWithAntiAffinity, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建Pod失败: %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	proxy.impl.Write(namespacedLabels{labels: podWithAntiAffinity.Labels, namespace: podWithAntiAffinity.Namespace}, &FilteringExistingPodAffinityTermDetailedState{
		preCalRes: topologyToMatchedTermCount{
			topologyPair{"zone", "z2"}: 1,
		},
		cachedPods: map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount{
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "mocked-pod",
			}: topologyToMatchedTermCount{
				topologyPair{"zone", "z2"}: 1,
			},
		},
		namespace:       "testns",
		labels:          map[string]string{"app": "test"},
		namespaceLabels: map[string]string{},
	})
	proxy.impl.ProcessReservePod(podWithAntiAffinity, node.Name)

	// 等待处理完成
	time.Sleep(10 * time.Millisecond)

	stateEqual := func(a, b *FilteringExistingPodAffinityTermDetailedState) bool {
		return cmp.Equal(a.cachedPods, b.cachedPods) &&
			cmp.Equal(a.preCalRes, b.preCalRes) &&
			cmp.Equal(a.labels, b.labels) &&
			cmp.Equal(a.namespace, b.namespace) &&
			cmp.Equal(a.namespaceLabels, b.namespaceLabels)
	}
	data := proxy.impl.Read(namespacedLabels{labels: podWithAntiAffinity.Labels, namespace: podWithAntiAffinity.Namespace})
	if diff := cmp.Diff(&FilteringExistingPodAffinityTermDetailedState{
		preCalRes: topologyToMatchedTermCount{
			topologyPair{"zone", "z2"}: 1,
			topologyPair{"zone", "z1"}: 1,
		},
		cachedPods: map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount{
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "mocked-pod",
			}: topologyToMatchedTermCount{
				topologyPair{"zone", "z2"}: 1,
			},
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "pod1",
			}: topologyToMatchedTermCount{
				topologyPair{"zone", "z1"}: 1,
			},
		},
		namespace:       "testns",
		labels:          map[string]string{"app": "test"},
		namespaceLabels: map[string]string{},
	}, data, cmp.Comparer(stateEqual)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}

	proxy.impl.ProcessUnreservePod(st.MakePod().Namespace("testns").Name("mocked-pod").Obj(), "")
	data = proxy.impl.Read(namespacedLabels{labels: podWithAntiAffinity.Labels, namespace: podWithAntiAffinity.Namespace})
	if diff := cmp.Diff(&FilteringExistingPodAffinityTermDetailedState{
		preCalRes: topologyToMatchedTermCount{
			topologyPair{"zone", "z2"}: 0,
			topologyPair{"zone", "z1"}: 1,
		},
		cachedPods: map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount{
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "pod1",
			}: topologyToMatchedTermCount{
				topologyPair{"zone", "z1"}: 1,
			},
		},
		namespace:       "testns",
		labels:          map[string]string{"app": "test"},
		namespaceLabels: map[string]string{},
	}, data, cmp.Comparer(stateEqual)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}
}

func TestFilteringIncomingPodCacheProxyPodEvents(t *testing.T) {
	// 创建测试环境
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	podInformer := informerFactory.Core().V1().Pods()
	namespaceInformer := informerFactory.Core().V1().Namespaces()
	nodeInformer := informerFactory.Core().V1().Nodes()

	// 创建测试节点
	node := st.MakeNode().Name("node1").Label("region", "r1").Label("zone", "z1").Obj()
	_, err := client.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建节点失败: %v", err)
	}

	// 创建测试命名空间
	ns := st.MakeNamespace().Name("testns").Obj()
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建命名空间失败: %v", err)
	}

	// 创建一个模拟的SharedLister
	snapshot := cache.NewSnapshot(nil, []*v1.Node{node})

	// 创建测试用的proxy
	proxy := NewFilteringIncomingPodCacheProxy(context.TODO(),
		podInformer.Lister(),
		podInformer,
		namespaceInformer,
		nodeInformer,
		snapshot,
	)

	// 启动informer工厂
	_, cancel := startInformerFactory(informerFactory)
	defer cancel()
	// 创建带有反亲和性的Pod
	pth := "test"
	podWithAntiAffinity := st.MakePod().Namespace("testns").Name("pod1").Label("app", "test").Label(apps.DefaultDeploymentUniqueLabelKey, pth).
		PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).Node("node1").Obj()
	podWithAntiAffinity, err = client.CoreV1().Pods(podWithAntiAffinity.Namespace).Create(context.TODO(), podWithAntiAffinity, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建Pod失败: %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	selector := labels.NewSelector()
	requirement, err := labels.NewRequirement("app", selection.Exists, []string{})
	if err != nil {
		t.Fatalf("创建requirement失败: %v", err)
	}
	selector = selector.Add(*requirement)
	proxy.impl.Write(pth, &FilteringIncomingPodAffinityTermDetailedState{
		cachedPods: map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount{
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "mocked-pod",
			}: []topologyToMatchedTermCount{
				{},
				{topologyPair{"zone", "z2"}: 1},
			},
		},
		affinityCounts:     topologyToMatchedTermCount{},
		antiAffinityCounts: topologyToMatchedTermCount{topologyPair{"zone", "z2"}: 1},
		affinity:           []fwk.AffinityTerm{},
		antiaffinity:       []fwk.AffinityTerm{{Namespaces: sets.Set[string]{"testns": struct{}{}}, TopologyKey: "zone", Selector: selector}},
	})
	proxy.impl.ProcessReservePod(podWithAntiAffinity, node.Name)

	// 等待处理完成
	time.Sleep(10 * time.Millisecond)

	stateEqual := func(a, b *FilteringIncomingPodAffinityTermDetailedState) bool {
		return cmp.Equal(a.cachedPods, b.cachedPods) &&
			cmp.Equal(a.affinityCounts, b.affinityCounts) &&
			cmp.Equal(a.antiAffinityCounts, b.antiAffinityCounts) &&
			cmp.Equal(a.affinity, b.affinity) &&
			cmp.Equal(a.antiaffinity, b.antiaffinity)
	}
	data := proxy.impl.Read(pth)
	if diff := cmp.Diff(&FilteringIncomingPodAffinityTermDetailedState{
		cachedPods: map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount{
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "mocked-pod",
			}: []topologyToMatchedTermCount{
				{},
				{topologyPair{"zone", "z2"}: 1},
			},
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "pod1",
			}: []topologyToMatchedTermCount{
				{},
				{topologyPair{"zone", "z1"}: 1},
			},
		},
		affinityCounts:     topologyToMatchedTermCount{},
		antiAffinityCounts: topologyToMatchedTermCount{topologyPair{"zone", "z2"}: 1, topologyPair{"zone", "z1"}: 1},
		affinity:           []fwk.AffinityTerm{},
		antiaffinity:       []fwk.AffinityTerm{{Namespaces: sets.Set[string]{"testns": struct{}{}}, TopologyKey: "zone", Selector: selector}},
	}, data, cmp.Comparer(stateEqual)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}

	proxy.impl.ProcessUnreservePod(st.MakePod().Namespace("testns").Name("mocked-pod").Obj(), "")
	data = proxy.impl.Read(pth)
	if diff := cmp.Diff(&FilteringIncomingPodAffinityTermDetailedState{
		cachedPods: map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount{
			cacheplugin.NamespaceedNameNode{
				Namespace: "testns",
				Name:      "pod1",
			}: []topologyToMatchedTermCount{
				{},
				{topologyPair{"zone", "z1"}: 1},
			},
		},
		affinityCounts:     topologyToMatchedTermCount{},
		antiAffinityCounts: topologyToMatchedTermCount{topologyPair{"zone", "z1"}: 1},
		affinity:           []fwk.AffinityTerm{},
		antiaffinity:       []fwk.AffinityTerm{{Namespaces: sets.Set[string]{"testns": struct{}{}}, TopologyKey: "zone", Selector: selector}},
	}, data, cmp.Comparer(stateEqual)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}
}

// 辅助函数：启动informer工厂
func startInformerFactory(informerFactory informers.SharedInformerFactory) (<-chan struct{}, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	return ctx.Done(), cancel
}

// TestGetExistingAntiAffinityCountsTopoMapsByPod tests the getExistingAntiAffinityCounts function
// to ensure that topoMapsByPod is correctly populated.
func TestGetExistingAntiAffinityCountsTopoMapsByPod(t *testing.T) {
	tests := []struct {
		name              string
		pod               *v1.Pod
		existingPods      []*v1.Pod
		nodes             []*v1.Node
		expectedCounts    topologyToMatchedTermCount
		expectedByPodKeys []cacheplugin.NamespaceedNameNode
	}{
		{
			name: "single existing pod with anti-affinity matching incoming pod",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Obj(),
			},
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}: 1,
			},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{
				{Namespace: "testns", Name: "existing1"},
			},
		},
		{
			name: "multiple existing pods with anti-affinity on different nodes",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
				st.MakePod().Namespace("testns").Name("existing2").Label("service", "cache").
					PodAntiAffinityExists("app", "region", st.PodAntiAffinityWithRequiredReq).
					Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Label("region", "r1").Obj(),
				st.MakeNode().Name("node2").Label("zone", "z2").Label("region", "r1").Obj(),
			},
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}:   1,
				topologyPair{key: "region", value: "r1"}: 1,
			},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{
				{Namespace: "testns", Name: "existing1"},
				{Namespace: "testns", Name: "existing2"},
			},
		},
		{
			name: "existing pod anti-affinity does not match incoming pod labels",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db").
					PodAntiAffinityExists("nonexistent-label", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Obj(),
			},
			expectedCounts:    topologyToMatchedTermCount{},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{},
		},
		{
			name: "multiple anti-affinity terms on same existing pod",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Label("tier", "frontend").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					PodAntiAffinityExists("tier", "region", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Label("region", "r1").Obj(),
			},
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}:   1,
				topologyPair{key: "region", value: "r1"}: 1,
			},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{
				{Namespace: "testns", Name: "existing1"},
			},
		},
		{
			name: "same topology value on multiple nodes",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db1").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
				st.MakePod().Namespace("testns").Name("existing2").Label("service", "db2").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Obj(),
				st.MakeNode().Name("node2").Label("zone", "z1").Obj(), // same zone
			},
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}: 2,
			},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{
				{Namespace: "testns", Name: "existing1"},
				{Namespace: "testns", Name: "existing2"},
			},
		},
		{
			name: "anti-affinity with specific label value",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db").
					PodAntiAffinityIn("app", "zone", []string{"web", "api"}, st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Obj(),
			},
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}: 1,
			},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{
				{Namespace: "testns", Name: "existing1"},
			},
		},
		{
			name: "anti-affinity with label value not matching incoming pod",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db").
					PodAntiAffinityIn("app", "zone", []string{"cache", "api"}, st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Obj(),
			},
			expectedCounts:    topologyToMatchedTermCount{},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{},
		},
		{
			name: "namespace aware anti-affinity",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").Label("service", "db").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
				st.MakePod().Namespace("otherns").Name("existing2").Label("service", "cache").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Obj(),
				st.MakeNode().Name("node2").Label("zone", "z2").Obj(),
			},
			// only testns pod matches since default namespace selector matches pods in same namespace
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}: 1,
			},
			expectedByPodKeys: []cacheplugin.NamespaceedNameNode{
				{Namespace: "testns", Name: "existing1"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			// Create snapshot with existing pods and nodes
			snapshot := cache.NewSnapshot(tt.existingPods, tt.nodes)

			// Get nodes with required anti-affinity pods
			nodesWithAntiAffinity, err := snapshot.NodeInfos().HavePodsWithRequiredAntiAffinityList()
			if err != nil {
				t.Fatalf("Failed to get nodes with anti-affinity: %v", err)
			}

			// Create InterPodAffinity plugin with minimal setup
			pl := &InterPodAffinity{
				parallelizer: parallelize.NewParallelizer(parallelize.DefaultParallelism),
				sharedLister: snapshot,
			}

			// Get namespace labels (empty for test)
			nsLabels := labels.Set{}

			// Call getExistingAntiAffinityCounts
			result, topoMapsByPod := pl.getExistingAntiAffinityCounts(ctx, tt.pod, nsLabels, nodesWithAntiAffinity)

			// Verify the result counts
			if diff := cmp.Diff(tt.expectedCounts, result); diff != "" {
				t.Errorf("getExistingAntiAffinityCounts() result mismatch (-want +got):\n%s", diff)
			}

			// Verify topoMapsByPod contains expected pod keys
			if len(tt.expectedByPodKeys) == 0 {
				if len(topoMapsByPod) != 0 {
					t.Errorf("expected empty topoMapsByPod, got %d entries", len(topoMapsByPod))
				}
			} else {
				// Check that all expected keys are present
				for _, expectedKey := range tt.expectedByPodKeys {
					if _, found := topoMapsByPod[expectedKey]; !found {
						t.Errorf("expected topoMapsByPod to contain key %v, but it was not found", expectedKey)
					}
				}

				// Check that no unexpected keys are present
				expectedKeySet := make(map[cacheplugin.NamespaceedNameNode]struct{})
				for _, k := range tt.expectedByPodKeys {
					expectedKeySet[k] = struct{}{}
				}
				for gotKey := range topoMapsByPod {
					if _, expected := expectedKeySet[gotKey]; !expected {
						t.Errorf("unexpected key in topoMapsByPod: %v", gotKey)
					}
				}
			}

			// Verify each pod's topology count is correct
			for key, topoCount := range topoMapsByPod {
				// Verify that each topoCount entry has non-zero counts
				if len(topoCount) == 0 {
					t.Errorf("topoMapsByPod[%v] has empty topology count", key)
				}
				for tp, count := range topoCount {
					if count <= 0 {
						t.Errorf("topoMapsByPod[%v][%v] has non-positive count: %d", key, tp, count)
					}
				}
			}
		})
	}
}

// TestGetExistingAntiAffinityCountsTopoMapsByPodDetailedValues provides detailed verification
// of the values in topoMapsByPod
func TestGetExistingAntiAffinityCountsTopoMapsByPodDetailedValues(t *testing.T) {
	tests := []struct {
		name           string
		pod            *v1.Pod
		existingPods   []*v1.Pod
		nodes          []*v1.Node
		expectedByPod  map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount
		expectedCounts topologyToMatchedTermCount
	}{
		{
			name: "verify detailed topology values per pod",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Label("tier", "frontend").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
				st.MakePod().Namespace("testns").Name("existing2").
					PodAntiAffinityExists("tier", "region", st.PodAntiAffinityWithRequiredReq).
					Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Label("region", "r1").Obj(),
				st.MakeNode().Name("node2").Label("zone", "z2").Label("region", "r1").Obj(),
			},
			expectedByPod: map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount{
				{Namespace: "testns", Name: "existing1"}: {
					topologyPair{key: "zone", value: "z1"}: 1,
				},
				{Namespace: "testns", Name: "existing2"}: {
					topologyPair{key: "region", value: "r1"}: 1,
				},
			},
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}:   1,
				topologyPair{key: "region", value: "r1"}: 1,
			},
		},
		{
			name: "pod with multiple matching anti-affinity terms",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Label("env", "prod").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					PodAntiAffinityExists("env", "region", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Label("region", "r1").Obj(),
			},
			expectedByPod: map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount{
				{Namespace: "testns", Name: "existing1"}: {
					topologyPair{key: "zone", value: "z1"}:   1,
					topologyPair{key: "region", value: "r1"}: 1,
				},
			},
			expectedCounts: topologyToMatchedTermCount{
				topologyPair{key: "zone", value: "z1"}:   1,
				topologyPair{key: "region", value: "r1"}: 1,
			},
		},
		{
			name: "pods on same topology domain aggregate counts correctly",
			pod:  st.MakePod().Namespace("testns").Name("incoming").Label("app", "web").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace("testns").Name("existing1").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node1").Obj(),
				st.MakePod().Namespace("testns").Name("existing2").
					PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).
					Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("zone", "z1").Obj(),
				st.MakeNode().Name("node2").Label("zone", "z1").Obj(), // same zone
			},
			expectedByPod: map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount{
				{Namespace: "testns", Name: "existing1"}: {
					topologyPair{key: "zone", value: "z1"}: 1,
				},
				{Namespace: "testns", Name: "existing2"}: {
					topologyPair{key: "zone", value: "z1"}: 1,
				},
			},
			expectedCounts: topologyToMatchedTermCount{
				// aggregated count is 2 since both pods contribute
				topologyPair{key: "zone", value: "z1"}: 2,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			snapshot := cache.NewSnapshot(tt.existingPods, tt.nodes)
			nodesWithAntiAffinity, err := snapshot.NodeInfos().HavePodsWithRequiredAntiAffinityList()
			if err != nil {
				t.Fatalf("Failed to get nodes with anti-affinity: %v", err)
			}

			pl := &InterPodAffinity{
				parallelizer: parallelize.NewParallelizer(parallelize.DefaultParallelism),
				sharedLister: snapshot,
			}

			nsLabels := labels.Set{}
			result, topoMapsByPod := pl.getExistingAntiAffinityCounts(ctx, tt.pod, nsLabels, nodesWithAntiAffinity)

			// Verify overall counts
			if diff := cmp.Diff(tt.expectedCounts, result); diff != "" {
				t.Errorf("result counts mismatch (-want +got):\n%s", diff)
			}

			// Verify detailed per-pod topology counts
			if diff := cmp.Diff(tt.expectedByPod, topoMapsByPod); diff != "" {
				t.Errorf("topoMapsByPod mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
