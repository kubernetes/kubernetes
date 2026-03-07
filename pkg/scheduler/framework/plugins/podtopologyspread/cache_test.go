package podtopologyspread

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	kubefake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

func Test_podEvtHandle(t *testing.T) {
	podSelector := labels.SelectorFromValidatedSet(labels.Set{"app": "test"})
	selector := st.MakeLabelSelector().Label("app", "test").Obj()

	testPod1 := st.MakePod().Name("test1").Namespace("default").UID("test1").Label("app", "test").NodeAffinityIn("zone", []string{"a"}, st.NodeSelectorTypeMatchExpressions).SpreadConstraint(
		1, "kubernetes.io/hostname", v1.ScheduleAnyway, selector, nil,
		ptr.To(v1.NodeInclusionPolicyHonor), ptr.To(v1.NodeInclusionPolicyIgnore), nil,
	).Node("test-node").Obj()
	testPod2 := st.MakePod().Name("test2").Namespace("default").UID("test2").Label("app", "test").NodeAffinityIn("zone", []string{"a"}, st.NodeSelectorTypeMatchExpressions).SpreadConstraint(
		1, "kubernetes.io/hostname", v1.ScheduleAnyway, selector, nil,
		ptr.To(v1.NodeInclusionPolicyHonor), ptr.To(v1.NodeInclusionPolicyIgnore), nil,
	).Node("test-node").Obj()
	testPod3 := st.MakePod().Name("test3").Namespace("default").UID("test3").Label("app", "test-not-match").NodeAffinityIn("zone", []string{"a"}, st.NodeSelectorTypeMatchExpressions).SpreadConstraint(
		1, "kubernetes.io/hostname", v1.ScheduleAnyway, selector, nil,
		ptr.To(v1.NodeInclusionPolicyHonor), ptr.To(v1.NodeInclusionPolicyIgnore), nil,
	).Node("test-node").Obj()
	testPod4 := st.MakePod().Name("test4").Namespace("default").UID("test4").Label("app", "test").SpreadConstraint(
		1, "kubernetes.io/hostname", v1.ScheduleAnyway, selector, nil,
		ptr.To(v1.NodeInclusionPolicyHonor), ptr.To(v1.NodeInclusionPolicyIgnore), nil,
	).Node("test-node1").Obj()

	testnode := st.MakeNode().Name("test-node").Label("kubernetes.io/hostname", "test-node").Label("zone", "a").Obj()
	testnode1 := st.MakeNode().Name("test-node1").Label("kubernetes.io/hostname", "test-node1").Label("zone", "b").Obj()

	cli := kubefake.NewSimpleClientset(testPod1, testPod2, testPod3, testPod4, testnode, testnode1)
	factory := informers.NewSharedInformerFactory(cli, 0)
	podinformer := factory.Core().V1().Pods()
	nodeinformer := factory.Core().V1().Nodes()
	podlister := factory.Core().V1().Pods().Lister()

	snap := cache.NewSnapshot([]*v1.Pod{testPod1, testPod2, testPod3, testPod4}, []*v1.Node{testnode, testnode1})

	evtHandle := buildPodEvtHandle(&PodTopologySpread{}, podlister, podinformer, nodeinformer, snap)

	factory.Start(nil)
	factory.WaitForCacheSync(nil)

	affinity := nodeaffinity.GetRequiredNodeAffinity(testPod1)
	originState := &PodTopologySpreadState{
		cachedPods: make(cachedPodsMap),
		constraints: []topologySpreadConstraint{
			{
				MaxSkew:            1,
				TopologyKey:        "kubernetes.io/hostname",
				Selector:           podSelector,
				NodeAffinityPolicy: v1.NodeInclusionPolicyHonor,
				NodeTaintsPolicy:   v1.NodeInclusionPolicyIgnore,
			},
		},
		namespace:            "default",
		preCalRes:            []map[string]*int64{},
		requireAllTopologies: false,
		requiredNodeAffinity: affinity,
	}

	evtHandle(cacheplugin.NamespaceedNameNode{Namespace: "default", Name: "test1", ReservedNode: "test-node"}, originState, klog.Background())

	calRes := make([]map[string]int64, 1)
	for i, v := range originState.preCalRes {
		mp := map[string]int64{}
		for k, vv := range v {
			mp[k] = *vv
		}
		calRes[i] = mp
	}
	assert.Equal(t, []map[string]int64{{
		"test-node": 1,
	}}, calRes)

	evtHandle(cacheplugin.NamespaceedNameNode{Namespace: "default", Name: "test2", ReservedNode: "test-node"}, originState, klog.Background())

	for i, v := range originState.preCalRes {
		mp := map[string]int64{}
		for k, vv := range v {
			mp[k] = *vv
		}
		calRes[i] = mp
	}
	assert.Equal(t, []map[string]int64{{
		"test-node": 2,
	}}, calRes)

	evtHandle(cacheplugin.NamespaceedNameNode{Namespace: "default", Name: "test3", ReservedNode: "test-node"}, originState, klog.Background())

	for i, v := range originState.preCalRes {
		mp := map[string]int64{}
		for k, vv := range v {
			mp[k] = *vv
		}
		calRes[i] = mp
	}
	assert.Equal(t, []map[string]int64{{
		"test-node": 2,
	}}, calRes)

	evtHandle(cacheplugin.NamespaceedNameNode{Namespace: "default", Name: "test4", ReservedNode: "test-node1"}, originState, klog.Background())

	for i, v := range originState.preCalRes {
		mp := map[string]int64{}
		for k, vv := range v {
			mp[k] = *vv
		}
		calRes[i] = mp
	}
	assert.Equal(t, []map[string]int64{{
		"test-node": 2,
	}}, calRes)

	_ = cli.CoreV1().Pods("default").Delete(context.Background(), "test1", metav1.DeleteOptions{})
	time.Sleep(time.Millisecond * 20)
	evtHandle(cacheplugin.NamespaceedNameNode{Namespace: "default", Name: "test1", ReservedNode: "test-node"}, originState, klog.Background())
	for i, v := range originState.preCalRes {
		mp := map[string]int64{}
		for k, vv := range v {
			mp[k] = *vv
		}
		calRes[i] = mp
	}
	assert.Equal(t, []map[string]int64{{
		"test-node": 1,
	}}, calRes)
}
