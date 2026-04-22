package interpodaffinity

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestIncomingPodCacheProxyPodEvents(t *testing.T) {
	// 创建测试环境
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	_ = informerFactory.Core().V1().Pods()
	_ = informerFactory.Core().V1().Namespaces()
	_ = informerFactory.Core().V1().Nodes()

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
	pth := "test"
	podWithAntiAffinity := st.MakePod().Namespace("testns").Name("pod1").Label("app", "test").Label("app1", "test").Label(apps.DefaultDeploymentUniqueLabelKey, pth).
		PodAffinityExists("app1", "zone", st.PodAntiAffinityWithRequiredPreferredReq).
		PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredPreferredReq).Node("node1").Obj()
	podWithAntiAffinity, err = client.CoreV1().Pods(podWithAntiAffinity.Namespace).Create(context.TODO(), podWithAntiAffinity, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建Pod失败: %v", err)
	}
	// 创建一个模拟的SharedLister
	snapshot := cache.NewSnapshot([]*v1.Pod{podWithAntiAffinity}, []*v1.Node{node})
	fw, err := runtime.NewFramework(context.Background(), nil, nil,
		runtime.WithClientSet(client), runtime.WithInformerFactory(informerFactory), runtime.WithSnapshotSharedLister(snapshot))
	if err != nil {
		t.Fatalf("创建framework失败: %v", err)
	}

	_ = os.Setenv("EnableInterPodAffinityCache", "true")
	plugin, err := New(context.Background(), &config.InterPodAffinityArgs{}, fw, feature.Features{})
	if err != nil {
		t.Fatalf("创建插件失败: %v", err)
	}

	// 启动informer工厂
	_, cancel := startInformerFactory(informerFactory)
	defer cancel()
	// 创建带有反亲和性的Pod
	mockedPod := st.MakePod().Namespace("testns").Name("mocked-pod").Label("app", "test").Label("app1", "test").
		Label(apps.DefaultDeploymentUniqueLabelKey, pth).
		PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredPreferredReq).Obj()
	mockedPod, err = client.CoreV1().Pods(mockedPod.Namespace).Create(context.TODO(), mockedPod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("创建Pod失败: %v", err)
	}

	nodes, err := fw.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		t.Fatalf("error listing nodes: %v", err)
	}
	proxy := plugin.(*InterPodAffinity).incomingPodCache
	proxy1 := plugin.(*InterPodAffinity).existingPodCache
	plugin.(*InterPodAffinity).PreScore(context.Background(), framework.NewCycleState(), mockedPod, nodes)

	stateEqual := func(a, b *IncomingPodAffinityTermDetailedState) bool {
		ja := cmp.Diff(a.cachedPods, b.cachedPods)
		jb := cmp.Diff(a.preCalRes, b.preCalRes)
		jc := cmp.Diff(a.affinity, b.affinity, cmpopts.EquateEmpty())
		jd := cmp.Diff(a.antiaffinity, b.antiaffinity)
		if ja != "" {
			t.Log(ja)
		}
		if jb != "" {
			t.Log(jb)
		}
		if jc != "" {
			t.Log(jc)
		}
		if jd != "" {
			t.Log(jd)
		}
		return ja == "" && jb == "" && jc == "" && jd == ""
	}
	stateEqual1 := func(a, b *ExistingPodAffinityTermDetailedState) bool {
		ja := cmp.Diff(a.cachedPods, b.cachedPods)
		jb := cmp.Diff(a.preCalRes, b.preCalRes)
		jc := cmp.Diff(a.labels, b.labels, cmpopts.EquateEmpty())
		jd := cmp.Diff(a.namespaceLabels, b.namespaceLabels)
		if ja != "" {
			t.Log(ja)
		}
		if jb != "" {
			t.Log(jb)
		}
		if jc != "" {
			t.Log(jc)
		}
		if jd != "" {
			t.Log(jd)
		}
		return ja == "" && jb == "" && jc == "" && jd == ""
	}
	data := proxy.impl.Read(pth)
	data1 := proxy1.impl.Read(namespacedLabels{namespace: podWithAntiAffinity.Namespace, labels: podWithAntiAffinity.Labels})

	t.Log("---------------------------------- Before Add Pod -------------------------------------")
	selector := labels.NewSelector()
	req, err := labels.NewRequirement("app", selection.Exists, nil)
	if err != nil {
		t.Fatal(err)
	}
	selector = selector.Add(*req)
	if diff := cmp.Diff(&IncomingPodAffinityTermDetailedState{
		cachedPods: cachedPodsMap{cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "pod1"}: scoreMap{"zone": map[string]int64{"z1": -1}}},
		preCalRes:  scoreMap{"zone": map[string]int64{"z1": -1}},
		affinity:   []fwk.WeightedAffinityTerm{},
		antiaffinity: []fwk.WeightedAffinityTerm{{
			AffinityTerm: fwk.AffinityTerm{Namespaces: sets.Set[string]{"testns": struct{}{}}, TopologyKey: "zone", Selector: selector, NamespaceSelector: labels.Nothing()},
			Weight:       1}},
	}, data, cmp.Comparer(stateEqual)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}
	if diff := cmp.Diff(&ExistingPodAffinityTermDetailedState{
		cachedPods:      cachedPodsMap{cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "pod1"}: scoreMap{"zone": map[string]int64{"z1": -1}}},
		preCalRes:       scoreMap{"zone": map[string]int64{"z1": -1}},
		namespace:       "testns",
		labels:          podWithAntiAffinity.Labels,
		namespaceLabels: map[string]string{},
	}, data1, cmp.Comparer(stateEqual1)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}

	proxy.impl.ProcessReservePod(mockedPod, "node1")
	proxy1.impl.ProcessReservePod(mockedPod, "node1")
	time.Sleep(time.Millisecond * 10)

	t.Log("---------------------------------- After Add Pod -------------------------------------")
	data = proxy.impl.Read(pth)
	data1 = proxy1.impl.Read(namespacedLabels{namespace: podWithAntiAffinity.Namespace, labels: podWithAntiAffinity.Labels})
	if diff := cmp.Diff(&IncomingPodAffinityTermDetailedState{
		cachedPods: cachedPodsMap{cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "pod1"}: scoreMap{"zone": map[string]int64{"z1": -1}},
			cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "mocked-pod"}: scoreMap{"zone": map[string]int64{"z1": -1}}},
		preCalRes: scoreMap{"zone": map[string]int64{"z1": -2}},
		affinity:  []fwk.WeightedAffinityTerm{},
		antiaffinity: []fwk.WeightedAffinityTerm{{
			AffinityTerm: fwk.AffinityTerm{Namespaces: sets.Set[string]{"testns": struct{}{}}, TopologyKey: "zone", Selector: selector, NamespaceSelector: labels.Nothing()},
			Weight:       1}},
	}, data, cmp.Comparer(stateEqual)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}
	if diff := cmp.Diff(&ExistingPodAffinityTermDetailedState{
		cachedPods: cachedPodsMap{
			cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "pod1"}:       scoreMap{"zone": map[string]int64{"z1": -1}},
			cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "mocked-pod"}: scoreMap{"zone": map[string]int64{"z1": -1}},
		},
		preCalRes:       scoreMap{"zone": map[string]int64{"z1": -2}},
		namespace:       "testns",
		labels:          podWithAntiAffinity.Labels,
		namespaceLabels: map[string]string{},
	}, data1, cmp.Comparer(stateEqual1)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}

	time.Sleep(time.Millisecond * 10)
	t.Log("---------------------------------- After Unreserve Pod -------------------------------------")

	proxy.impl.ProcessUnreservePod(st.MakePod().Namespace("testns").Name("mocked-pod").Obj(), "")
	proxy1.impl.ProcessUnreservePod(st.MakePod().Namespace("testns").Name("mocked-pod").Obj(), "")
	data = proxy.impl.Read(pth)
	data1 = proxy1.impl.Read(namespacedLabels{namespace: podWithAntiAffinity.Namespace, labels: podWithAntiAffinity.Labels})
	if diff := cmp.Diff(&IncomingPodAffinityTermDetailedState{
		cachedPods: cachedPodsMap{cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "pod1"}: scoreMap{"zone": map[string]int64{"z1": -1}}},
		preCalRes:  scoreMap{"zone": map[string]int64{"z1": -1}},
		affinity:   []fwk.WeightedAffinityTerm{},
		antiaffinity: []fwk.WeightedAffinityTerm{{
			AffinityTerm: fwk.AffinityTerm{Namespaces: sets.Set[string]{"testns": struct{}{}}, TopologyKey: "zone", Selector: selector, NamespaceSelector: labels.Nothing()},
			Weight:       1}},
	}, data, cmp.Comparer(stateEqual)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}
	if diff := cmp.Diff(&ExistingPodAffinityTermDetailedState{
		cachedPods:      cachedPodsMap{cacheplugin.NamespaceedNameNode{Namespace: "testns", Name: "pod1"}: scoreMap{"zone": map[string]int64{"z1": -1}}},
		preCalRes:       scoreMap{"zone": map[string]int64{"z1": -1}},
		namespace:       "testns",
		labels:          podWithAntiAffinity.Labels,
		namespaceLabels: map[string]string{},
	}, data1, cmp.Comparer(stateEqual1)); diff != "" {
		t.Errorf("unexpected diff: %s", diff)
	}
}
