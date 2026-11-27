/*
Copyright The Kubernetes Authors.

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

package endpointslice

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

type topologyHintsFixture struct {
	c      *endpointSliceController
	client *fake.Clientset
	nodes  []*v1.Node
	logger klog.Logger
	ctx    context.Context
	pods   int
}

func newTopologyHintsFixture(t *testing.T, pods int, configure func(*v1.Service, *[]*v1.Node)) *topologyHintsFixture {
	t.Helper()
	client, c := newController(t, nil, 0)
	logger, ctx := ktesting.NewTestContext(t)
	f := &topologyHintsFixture{c: c, client: client, logger: logger, ctx: ctx, pods: pods}
	for i, zone := range []string{"zone-a", "zone-b"} {
		f.nodes = append(f.nodes, &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("node-%d", i+1), Labels: map[string]string{v1.LabelTopologyZone: zone}},
			Status:     v1.NodeStatus{Allocatable: v1.ResourceList{v1.ResourceCPU: resource.MustParse("4")}, Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}},
		})
	}
	service := createService(t, c, "ns", "svc").DeepCopy()
	service.Annotations = map[string]string{v1.AnnotationTopologyMode: "Auto"}
	if configure != nil {
		configure(service, &f.nodes)
	}
	for _, node := range f.nodes {
		require.NoError(t, c.nodeStore.Add(node))
	}
	c.addNode()
	c.processNextTopologyWorkItem(logger)
	for i := 0; i < pods; i++ {
		pod := newPod(i, "ns", true, 0, false)
		pod.Spec.NodeName = f.nodes[i*2/pods].Name
		require.NoError(t, c.podStore.Add(pod))
	}
	require.NoError(t, c.serviceStore.Update(service))
	require.NoError(t, c.syncService(logger, "ns/svc"))
	return f
}

func (f *topologyHintsFixture) hints(t *testing.T) map[string]int {
	t.Helper()
	slices, err := f.client.DiscoveryV1().EndpointSlices("ns").List(f.ctx, metav1.ListOptions{})
	require.NoError(t, err)
	counts := map[string]int{}
	total := 0
	for _, slice := range slices.Items {
		for _, ep := range slice.Endpoints {
			total++
			if ep.Hints != nil {
				require.Len(t, ep.Hints.ForZones, 1)
				counts[ep.Hints.ForZones[0].Name]++
			}
		}
	}
	require.Equal(t, f.pods, total)
	return counts
}

func (f *topologyHintsFixture) updateNode(t *testing.T, index int, mutate func(*v1.Node)) int {
	t.Helper()
	old := f.nodes[index]
	updated := old.DeepCopy()
	mutate(updated)
	require.NoError(t, f.c.nodeStore.Update(updated))
	f.c.updateNode(old, updated)
	if f.c.topologyQueue.Len() > 0 {
		f.c.processNextTopologyWorkItem(f.logger)
	}
	queued := f.c.serviceQueue.Len()
	for i := 0; f.c.serviceQueue.Len() > 0; i++ {
		require.Less(t, i, 10, "unexpected queue loop")
		f.c.processNextServiceWorkItem(f.logger)
	}
	f.nodes[index] = updated
	return queued
}

func TestNodeCPUUpdatePreservesHintPolicies(t *testing.T) {
	balanced := map[string]int{"zone-a": 3, "zone-b": 3}
	for _, tc := range []struct {
		name      string
		configure func(*v1.Service, *[]*v1.Node)
		target    int
		pods      int
		want      map[string]int
	}{
		{name: "control-plane remains excluded", target: 2, pods: 6, want: balanced, configure: func(_ *v1.Service, nodes *[]*v1.Node) {
			extra := (*nodes)[0].DeepCopy()
			extra.Name = "excluded"
			extra.Labels["node-role.kubernetes.io/control-plane"] = ""
			*nodes = append(*nodes, extra)
		}},
		{name: "legacy master remains excluded", target: 2, pods: 6, want: balanced, configure: func(_ *v1.Service, nodes *[]*v1.Node) {
			extra := (*nodes)[0].DeepCopy()
			extra.Name = "excluded"
			extra.Labels["node-role.kubernetes.io/master"] = "true"
			*nodes = append(*nodes, extra)
		}},
		{name: "NotReady remains excluded", target: 2, pods: 6, want: balanced, configure: func(_ *v1.Service, nodes *[]*v1.Node) {
			extra := (*nodes)[0].DeepCopy()
			extra.Name = "notready"
			extra.Status.Conditions[0].Status = v1.ConditionFalse
			*nodes = append(*nodes, extra)
		}},
		{name: "disabled topology remains disabled", pods: 6, want: map[string]int{}, configure: func(s *v1.Service, _ *[]*v1.Node) { s.Annotations[v1.AnnotationTopologyMode] = "Disabled" }},
		{name: "no Auto annotation remains unhinted", pods: 6, want: map[string]int{}, configure: func(s *v1.Service, _ *[]*v1.Node) { s.Annotations = nil }},
		{name: "PreferSameZone retains its own distribution", pods: 6, want: balanced, configure: func(s *v1.Service, _ *[]*v1.Node) {
			s.Annotations = nil
			s.Spec.TrafficDistribution = ptr.To(v1.ServiceTrafficDistributionPreferSameZone)
		}},
		{name: "one zone remains unhinted", pods: 6, want: map[string]int{}, configure: func(_ *v1.Service, nodes *[]*v1.Node) { (*nodes)[1].Labels[v1.LabelTopologyZone] = "zone-a" }},
		{name: "insufficient endpoints remain unhinted", pods: 1, want: map[string]int{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newTopologyHintsFixture(t, tc.pods, tc.configure)
			require.Equal(t, tc.want, f.hints(t))
			f.updateNode(t, tc.target, func(n *v1.Node) { n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse("8") })
			require.Equal(t, tc.want, f.hints(t))
		})
	}
}

func TestTopologyHintsRecoverAfterNodeUpdate(t *testing.T) {
	for _, tc := range []struct {
		name    string
		pods    int
		disable func(*v1.Node)
		restore func(*v1.Node)
	}{
		{name: "CPU zero then restored", pods: 6, disable: func(n *v1.Node) { n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse("0") }},
		{name: "CPU missing then restored", pods: 6, disable: func(n *v1.Node) { delete(n.Status.Allocatable, v1.ResourceCPU) }},
		{name: "overload fallback then balanced again", pods: 2, disable: func(n *v1.Node) { n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse("8") }},
		{name: "existing readiness fallback then Ready again", pods: 6, disable: func(n *v1.Node) { n.Status.Conditions[0].Status = v1.ConditionFalse }, restore: func(n *v1.Node) { n.Status.Conditions[0].Status = v1.ConditionTrue }},
		{name: "missing zone then restored", pods: 6, disable: func(n *v1.Node) { delete(n.Labels, v1.LabelTopologyZone) }, restore: func(n *v1.Node) { n.Labels[v1.LabelTopologyZone] = "zone-a" }},
		{name: "one zone then two again", pods: 6, disable: func(n *v1.Node) { n.Labels[v1.LabelTopologyZone] = "zone-b" }, restore: func(n *v1.Node) { n.Labels[v1.LabelTopologyZone] = "zone-a" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newTopologyHintsFixture(t, tc.pods, nil)
			want := map[string]int{"zone-a": tc.pods / 2, "zone-b": tc.pods / 2}
			require.Equal(t, want, f.hints(t))
			f.updateNode(t, 0, tc.disable)
			require.Empty(t, f.hints(t), "fallback did not remove hints")
			f.updateNode(t, 0, func(n *v1.Node) {
				n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse("4")
				if tc.restore != nil {
					tc.restore(n)
				}
			})
			require.Equal(t, want, f.hints(t), "capacity recovered but hints did not return")
		})
	}
}

func TestTopologyHintsForgetUnmanagedServices(t *testing.T) {
	for _, tc := range []struct {
		name   string
		update func(*v1.Service)
		delete bool
		want   map[string]int
	}{
		{name: "explicitly disabled", update: func(s *v1.Service) { s.Annotations[v1.AnnotationTopologyMode] = "Disabled" }, want: map[string]int{}},
		{name: "Auto annotation removed", update: func(s *v1.Service) { s.Annotations = nil }, want: map[string]int{}},
		{name: "PreferSameZone replaces Auto", update: func(s *v1.Service) {
			s.Annotations = nil
			s.Spec.TrafficDistribution = ptr.To(v1.ServiceTrafficDistributionPreferSameZone)
		}, want: map[string]int{"zone-a": 3, "zone-b": 3}},
		{name: "selector removed", update: func(s *v1.Service) { s.Spec.Selector = nil }, want: map[string]int{}},
		{name: "ExternalName", update: func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeExternalName
			s.Spec.ExternalName = "example.test"
			s.Spec.IPFamilies = nil
		}, want: map[string]int{}},
		{name: "deleted", delete: true, want: map[string]int{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newTopologyHintsFixture(t, 6, nil)
			f.updateNode(t, 0, func(n *v1.Node) { n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse("0") })
			require.Empty(t, f.hints(t))
			obj, exists, err := f.c.serviceStore.GetByKey("ns/svc")
			require.NoError(t, err)
			require.True(t, exists)
			service := obj.(*v1.Service).DeepCopy()
			if tc.delete {
				require.NoError(t, f.c.serviceStore.Delete(service))
			} else {
				tc.update(service)
				require.NoError(t, f.c.serviceStore.Update(service))
			}
			require.NoError(t, f.c.syncService(f.logger, "ns/svc"))
			queued := f.updateNode(t, 0, func(n *v1.Node) { n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse("4") })
			require.Zero(t, queued, "unmanaged Service was retained as a pending Auto allocation")
			require.Equal(t, tc.want, f.hints(t))
		})
	}
}
