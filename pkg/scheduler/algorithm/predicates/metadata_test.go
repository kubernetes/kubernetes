/*
Copyright 2017 The Kubernetes Authors.

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

package predicates

import (
	"fmt"
	"reflect"
	"sort"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// sortablePods lets us to sort pods.
type sortablePods []*v1.Pod

func (s sortablePods) Less(i, j int) bool {
	return s[i].Namespace < s[j].Namespace ||
		(s[i].Namespace == s[j].Namespace && s[i].Name < s[j].Name)
}
func (s sortablePods) Len() int      { return len(s) }
func (s sortablePods) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

var _ = sort.Interface(&sortablePods{})

// sortableServices allows us to sort services.
type sortableServices []*v1.Service

func (s sortableServices) Less(i, j int) bool {
	return s[i].Namespace < s[j].Namespace ||
		(s[i].Namespace == s[j].Namespace && s[i].Name < s[j].Name)
}
func (s sortableServices) Len() int      { return len(s) }
func (s sortableServices) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

var _ = sort.Interface(&sortableServices{})

// predicateMetadataEquivalent returns true if the two metadata are equivalent.
// Note: this function does not compare podRequest.
func predicateMetadataEquivalent(meta1, meta2 *predicateMetadata) error {
	if !reflect.DeepEqual(meta1.pod, meta2.pod) {
		return fmt.Errorf("pods are not the same")
	}
	if meta1.podBestEffort != meta2.podBestEffort {
		return fmt.Errorf("podBestEfforts are not equal")
	}
	if meta1.serviceAffinityInUse != meta1.serviceAffinityInUse {
		return fmt.Errorf("serviceAffinityInUses are not equal")
	}
	if len(meta1.podPorts) != len(meta2.podPorts) {
		return fmt.Errorf("podPorts are not equal")
	}
	for !reflect.DeepEqual(meta1.podPorts, meta2.podPorts) {
		return fmt.Errorf("podPorts are not equal")
	}
	if !reflect.DeepEqual(meta1.topologyPairsPotentialAffinityPods, meta2.topologyPairsPotentialAffinityPods) {
		return fmt.Errorf("topologyPairsPotentialAffinityPods are not equal")
	}
	if !reflect.DeepEqual(meta1.topologyPairsPotentialAntiAffinityPods, meta2.topologyPairsPotentialAntiAffinityPods) {
		return fmt.Errorf("topologyPairsPotentialAntiAffinityPods are not equal")
	}
	if !reflect.DeepEqual(meta1.topologyPairsAntiAffinityPodsMap.podToTopologyPairs,
		meta2.topologyPairsAntiAffinityPodsMap.podToTopologyPairs) {
		return fmt.Errorf("topologyPairsAntiAffinityPodsMap.podToTopologyPairs are not equal")
	}
	if !reflect.DeepEqual(meta1.topologyPairsAntiAffinityPodsMap.topologyPairToPods,
		meta2.topologyPairsAntiAffinityPodsMap.topologyPairToPods) {
		return fmt.Errorf("topologyPairsAntiAffinityPodsMap.topologyPairToPods are not equal")
	}
	if meta1.serviceAffinityInUse {
		sortablePods1 := sortablePods(meta1.serviceAffinityMatchingPodList)
		sort.Sort(sortablePods1)
		sortablePods2 := sortablePods(meta2.serviceAffinityMatchingPodList)
		sort.Sort(sortablePods2)
		if !reflect.DeepEqual(sortablePods1, sortablePods2) {
			return fmt.Errorf("serviceAffinityMatchingPodLists are not euqal")
		}

		sortableServices1 := sortableServices(meta1.serviceAffinityMatchingPodServices)
		sort.Sort(sortableServices1)
		sortableServices2 := sortableServices(meta2.serviceAffinityMatchingPodServices)
		sort.Sort(sortableServices2)
		if !reflect.DeepEqual(sortableServices1, sortableServices2) {
			return fmt.Errorf("serviceAffinityMatchingPodServices are not euqal")
		}
	}
	return nil
}

func TestPredicateMetadata_AddRemovePod(t *testing.T) {
	var label1 = map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	var label2 = map[string]string{
		"region": "r1",
		"zone":   "z12",
	}
	var label3 = map[string]string{
		"region": "r2",
		"zone":   "z21",
	}
	selector1 := map[string]string{"foo": "bar"}
	antiAffinityFooBar := &v1.PodAntiAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
			{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"bar"},
						},
					},
				},
				TopologyKey: "region",
			},
		},
	}
	antiAffinityComplex := &v1.PodAntiAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
			{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"bar", "buzz"},
						},
					},
				},
				TopologyKey: "region",
			},
			{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "service",
							Operator: metav1.LabelSelectorOpNotIn,
							Values:   []string{"bar", "security", "test"},
						},
					},
				},
				TopologyKey: "zone",
			},
		},
	}
	affinityComplex := &v1.PodAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
			{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"bar", "buzz"},
						},
					},
				},
				TopologyKey: "region",
			},
			{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "service",
							Operator: metav1.LabelSelectorOpNotIn,
							Values:   []string{"bar", "security", "test"},
						},
					},
				},
				TopologyKey: "zone",
			},
		},
	}

	tests := []struct {
		name         string
		pendingPod   *v1.Pod
		addedPod     *v1.Pod
		existingPods []*v1.Pod
		nodes        []*v1.Node
		services     []*v1.Service
	}{
		{
			name: "no anti-affinity or service affinity exist",
			pendingPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pending", Labels: selector1},
			},
			existingPods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{NodeName: "nodeC"},
				},
			},
			addedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "addedPod", Labels: selector1},
				Spec:       v1.PodSpec{NodeName: "nodeB"},
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: label3}},
			},
		},
		{
			name: "metadata anti-affinity terms are updated correctly after adding and removing a pod",
			pendingPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pending", Labels: selector1},
			},
			existingPods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{
						NodeName: "nodeC",
						Affinity: &v1.Affinity{
							PodAntiAffinity: antiAffinityFooBar,
						},
					},
				},
			},
			addedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "addedPod", Labels: selector1},
				Spec: v1.PodSpec{
					NodeName: "nodeB",
					Affinity: &v1.Affinity{
						PodAntiAffinity: antiAffinityFooBar,
					},
				},
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: label3}},
			},
		},
		{
			name: "metadata service-affinity data are updated correctly after adding and removing a pod",
			pendingPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pending", Labels: selector1},
			},
			existingPods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{NodeName: "nodeC"},
				},
			},
			addedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "addedPod", Labels: selector1},
				Spec:       v1.PodSpec{NodeName: "nodeB"},
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector1}}},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: label3}},
			},
		},
		{
			name: "metadata anti-affinity terms and service affinity data are updated correctly after adding and removing a pod",
			pendingPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pending", Labels: selector1},
			},
			existingPods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{
						NodeName: "nodeC",
						Affinity: &v1.Affinity{
							PodAntiAffinity: antiAffinityFooBar,
						},
					},
				},
			},
			addedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "addedPod", Labels: selector1},
				Spec: v1.PodSpec{
					NodeName: "nodeA",
					Affinity: &v1.Affinity{
						PodAntiAffinity: antiAffinityComplex,
					},
				},
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector1}}},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: label3}},
			},
		},
		{
			name: "metadata matching pod affinity and anti-affinity are updated correctly after adding and removing a pod",
			pendingPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pending", Labels: selector1},
			},
			existingPods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{
						NodeName: "nodeC",
						Affinity: &v1.Affinity{
							PodAntiAffinity: antiAffinityFooBar,
							PodAffinity:     affinityComplex,
						},
					},
				},
			},
			addedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "addedPod", Labels: selector1},
				Spec: v1.PodSpec{
					NodeName: "nodeA",
					Affinity: &v1.Affinity{
						PodAntiAffinity: antiAffinityComplex,
					},
				},
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector1}}},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: label3}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allPodLister := st.FakePodLister(append(test.existingPods, test.addedPod))
			// getMeta creates predicate meta data given the list of pods.
			getMeta := func(lister st.FakePodLister) (*predicateMetadata, map[string]*schedulernodeinfo.NodeInfo) {
				nodeInfoMap := schedulernodeinfo.CreateNodeNameToInfoMap(lister, test.nodes)
				// nodeList is a list of non-pointer nodes to feed to FakeNodeListInfo.
				nodeList := []v1.Node{}
				for _, n := range test.nodes {
					nodeList = append(nodeList, *n)
				}
				_, precompute := NewServiceAffinityPredicate(lister, st.FakeServiceLister(test.services), FakeNodeListInfo(nodeList), nil)
				RegisterPredicateMetadataProducer("ServiceAffinityMetaProducer", precompute)
				pmf := PredicateMetadataFactory{lister}
				meta := pmf.GetMetadata(test.pendingPod, nodeInfoMap)
				return meta.(*predicateMetadata), nodeInfoMap
			}

			// allPodsMeta is meta data produced when all pods, including test.addedPod
			// are given to the metadata producer.
			allPodsMeta, _ := getMeta(allPodLister)
			// existingPodsMeta1 is meta data produced for test.existingPods (without test.addedPod).
			existingPodsMeta1, nodeInfoMap := getMeta(st.FakePodLister(test.existingPods))
			// Add test.addedPod to existingPodsMeta1 and make sure meta is equal to allPodsMeta
			nodeInfo := nodeInfoMap[test.addedPod.Spec.NodeName]
			if err := existingPodsMeta1.AddPod(test.addedPod, nodeInfo); err != nil {
				t.Errorf("error adding pod to meta: %v", err)
			}
			if err := predicateMetadataEquivalent(allPodsMeta, existingPodsMeta1); err != nil {
				t.Errorf("meta data are not equivalent: %v", err)
			}
			// Remove the added pod and from existingPodsMeta1 an make sure it is equal
			// to meta generated for existing pods.
			existingPodsMeta2, _ := getMeta(st.FakePodLister(test.existingPods))
			if err := existingPodsMeta1.RemovePod(test.addedPod); err != nil {
				t.Errorf("error removing pod from meta: %v", err)
			}
			if err := predicateMetadataEquivalent(existingPodsMeta1, existingPodsMeta2); err != nil {
				t.Errorf("meta data are not equivalent: %v", err)
			}
		})
	}
}

// TestPredicateMetadata_ShallowCopy tests the ShallowCopy function. It is based
// on the idea that shallow-copy should produce an object that is deep-equal to the original
// object.
func TestPredicateMetadata_ShallowCopy(t *testing.T) {
	selector1 := map[string]string{"foo": "bar"}
	source := predicateMetadata{
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test",
				Namespace: "testns",
			},
		},
		podBestEffort: true,
		podRequest: &schedulernodeinfo.Resource{
			MilliCPU:         1000,
			Memory:           300,
			AllowedPodNumber: 4,
		},
		podPorts: []*v1.ContainerPort{
			{
				Name:          "name",
				HostPort:      10,
				ContainerPort: 20,
				Protocol:      "TCP",
				HostIP:        "1.2.3.4",
			},
		},
		topologyPairsAntiAffinityPodsMap: &topologyPairsMaps{
			topologyPairToPods: map[topologyPair]podSet{
				{key: "name", value: "machine1"}: {
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p2", Labels: selector1},
						Spec: v1.PodSpec{NodeName: "nodeC"},
					}: struct{}{},
				},
				{key: "name", value: "machine2"}: {
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
						Spec: v1.PodSpec{NodeName: "nodeA"},
					}: struct{}{},
				},
			},
			podToTopologyPairs: map[string]topologyPairSet{
				"p2_": {
					topologyPair{key: "name", value: "machine1"}: struct{}{},
				},
				"p1_": {
					topologyPair{key: "name", value: "machine2"}: struct{}{},
				},
			},
		},
		topologyPairsPotentialAffinityPods: &topologyPairsMaps{
			topologyPairToPods: map[topologyPair]podSet{
				{key: "name", value: "nodeA"}: {
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
						Spec: v1.PodSpec{NodeName: "nodeA"},
					}: struct{}{},
				},
				{key: "name", value: "nodeC"}: {
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
						Spec: v1.PodSpec{
							NodeName: "nodeC",
						},
					}: struct{}{},
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p6", Labels: selector1},
						Spec: v1.PodSpec{NodeName: "nodeC"},
					}: struct{}{},
				},
			},
			podToTopologyPairs: map[string]topologyPairSet{
				"p1_": {
					topologyPair{key: "name", value: "nodeA"}: struct{}{},
				},
				"p2_": {
					topologyPair{key: "name", value: "nodeC"}: struct{}{},
				},
				"p6_": {
					topologyPair{key: "name", value: "nodeC"}: struct{}{},
				},
			},
		},
		topologyPairsPotentialAntiAffinityPods: &topologyPairsMaps{
			topologyPairToPods: map[topologyPair]podSet{
				{key: "name", value: "nodeN"}: {
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
						Spec: v1.PodSpec{NodeName: "nodeN"},
					}: struct{}{},
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
						Spec: v1.PodSpec{
							NodeName: "nodeM",
						},
					}: struct{}{},
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p3"},
						Spec: v1.PodSpec{
							NodeName: "nodeM",
						},
					}: struct{}{},
				},
				{key: "name", value: "nodeM"}: {
					&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p6", Labels: selector1},
						Spec: v1.PodSpec{NodeName: "nodeM"},
					}: struct{}{},
				},
			},
			podToTopologyPairs: map[string]topologyPairSet{
				"p1_": {
					topologyPair{key: "name", value: "nodeN"}: struct{}{},
				},
				"p2_": {
					topologyPair{key: "name", value: "nodeN"}: struct{}{},
				},
				"p3_": {
					topologyPair{key: "name", value: "nodeN"}: struct{}{},
				},
				"p6_": {
					topologyPair{key: "name", value: "nodeM"}: struct{}{},
				},
			},
		},
		topologyPairsPodSpreadMap: &topologyPairsPodSpreadMap{
			topologyKeyToMinPodsMap: map[string]int32{"name": 1},
			topologyPairsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "name", value: "nodeA"}: {
						&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
							Spec: v1.PodSpec{NodeName: "nodeA"},
						}: struct{}{},
					},
					{key: "name", value: "nodeC"}: {
						&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
							Spec: v1.PodSpec{
								NodeName: "nodeC",
							},
						}: struct{}{},
						&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p6", Labels: selector1},
							Spec: v1.PodSpec{NodeName: "nodeC"},
						}: struct{}{},
					},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"p1_": {
						topologyPair{key: "name", value: "nodeA"}: struct{}{},
					},
					"p2_": {
						topologyPair{key: "name", value: "nodeC"}: struct{}{},
					},
					"p6_": {
						topologyPair{key: "name", value: "nodeC"}: struct{}{},
					},
				},
			},
		},
		serviceAffinityInUse: true,
		serviceAffinityMatchingPodList: []*v1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}},
		},
		serviceAffinityMatchingPodServices: []*v1.Service{
			{ObjectMeta: metav1.ObjectMeta{Name: "service1"}},
		},
	}

	if !reflect.DeepEqual(source.ShallowCopy().(*predicateMetadata), &source) {
		t.Errorf("Copy is not equal to source!")
	}
}

// TestGetTPMapMatchingIncomingAffinityAntiAffinity tests against method getTPMapMatchingIncomingAffinityAntiAffinity
// on Anti Affinity cases
func TestGetTPMapMatchingIncomingAffinityAntiAffinity(t *testing.T) {
	newPodAffinityTerms := func(keys ...string) []v1.PodAffinityTerm {
		var terms []v1.PodAffinityTerm
		for _, key := range keys {
			terms = append(terms, v1.PodAffinityTerm{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      key,
							Operator: metav1.LabelSelectorOpExists,
						},
					},
				},
				TopologyKey: "hostname",
			})
		}
		return terms
	}
	newPod := func(labels ...string) *v1.Pod {
		labelMap := make(map[string]string)
		for _, l := range labels {
			labelMap[l] = ""
		}
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "normal", Labels: labelMap},
			Spec:       v1.PodSpec{NodeName: "nodeA"},
		}
	}
	normalPodA := newPod("aaa")
	normalPodB := newPod("bbb")
	normalPodAB := newPod("aaa", "bbb")
	nodeA := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: map[string]string{"hostname": "nodeA"}}}

	tests := []struct {
		name                     string
		existingPods             []*v1.Pod
		nodes                    []*v1.Node
		pod                      *v1.Pod
		wantAffinityPodsMaps     *topologyPairsMaps
		wantAntiAffinityPodsMaps *topologyPairsMaps
		wantErr                  bool
	}{
		{
			name:  "nil test",
			nodes: []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "aaa-normal"},
			},
			wantAffinityPodsMaps:     newTopologyPairsMaps(),
			wantAntiAffinityPodsMaps: newTopologyPairsMaps(),
		},
		{
			name:         "incoming pod without affinity/anti-affinity causes a no-op",
			existingPods: []*v1.Pod{normalPodA},
			nodes:        []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "aaa-normal"},
			},
			wantAffinityPodsMaps:     newTopologyPairsMaps(),
			wantAntiAffinityPodsMaps: newTopologyPairsMaps(),
		},
		{
			name:         "no pod has label that violates incoming pod's affinity and anti-affinity",
			existingPods: []*v1.Pod{normalPodB},
			nodes:        []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "aaa-anti"},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa"),
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa"),
						},
					},
				},
			},
			wantAffinityPodsMaps:     newTopologyPairsMaps(),
			wantAntiAffinityPodsMaps: newTopologyPairsMaps(),
		},
		{
			name:         "existing pod matches incoming pod's affinity and anti-affinity - single term case",
			existingPods: []*v1.Pod{normalPodA},
			nodes:        []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "affi-antiaffi"},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa"),
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa"),
						},
					},
				},
			},
			wantAffinityPodsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "hostname", value: "nodeA"}: {normalPodA: struct{}{}},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"normal_": {
						topologyPair{key: "hostname", value: "nodeA"}: struct{}{},
					},
				},
			},
			wantAntiAffinityPodsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "hostname", value: "nodeA"}: {normalPodA: struct{}{}},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"normal_": {
						topologyPair{key: "hostname", value: "nodeA"}: struct{}{},
					},
				},
			},
		},
		{
			name:         "existing pod matches incoming pod's affinity and anti-affinity - mutiple terms case",
			existingPods: []*v1.Pod{normalPodAB},
			nodes:        []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "affi-antiaffi"},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa", "bbb"),
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa"),
						},
					},
				},
			},
			wantAffinityPodsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "hostname", value: "nodeA"}: {normalPodAB: struct{}{}},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"normal_": {
						topologyPair{key: "hostname", value: "nodeA"}: struct{}{},
					},
				},
			},
			wantAntiAffinityPodsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "hostname", value: "nodeA"}: {normalPodAB: struct{}{}},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"normal_": {
						topologyPair{key: "hostname", value: "nodeA"}: struct{}{},
					},
				},
			},
		},
		{
			name:         "existing pod not match incoming pod's affinity but matches anti-affinity",
			existingPods: []*v1.Pod{normalPodA},
			nodes:        []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "affi-antiaffi"},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa", "bbb"),
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa", "bbb"),
						},
					},
				},
			},
			wantAffinityPodsMaps: newTopologyPairsMaps(),
			wantAntiAffinityPodsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "hostname", value: "nodeA"}: {normalPodA: struct{}{}},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"normal_": {
						topologyPair{key: "hostname", value: "nodeA"}: struct{}{},
					},
				},
			},
		},
		{
			name:         "incoming pod's anti-affinity has more than one term - existing pod violates partial term - case 1",
			existingPods: []*v1.Pod{normalPodAB},
			nodes:        []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "anaffi-antiaffiti"},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa", "ccc"),
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa", "ccc"),
						},
					},
				},
			},
			wantAffinityPodsMaps: newTopologyPairsMaps(),
			wantAntiAffinityPodsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "hostname", value: "nodeA"}: {normalPodAB: struct{}{}},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"normal_": {
						topologyPair{key: "hostname", value: "nodeA"}: struct{}{},
					},
				},
			},
		},
		{
			name:         "incoming pod's anti-affinity has more than one term - existing pod violates partial term - case 2",
			existingPods: []*v1.Pod{normalPodB},
			nodes:        []*v1.Node{nodeA},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "affi-antiaffi"},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa", "bbb"),
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: newPodAffinityTerms("aaa", "bbb"),
						},
					},
				},
			},
			wantAffinityPodsMaps: newTopologyPairsMaps(),
			wantAntiAffinityPodsMaps: &topologyPairsMaps{
				topologyPairToPods: map[topologyPair]podSet{
					{key: "hostname", value: "nodeA"}: {normalPodB: struct{}{}},
				},
				podToTopologyPairs: map[string]topologyPairSet{
					"normal_": {
						topologyPair{key: "hostname", value: "nodeA"}: struct{}{},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeInfoMap := schedulernodeinfo.CreateNodeNameToInfoMap(tt.existingPods, tt.nodes)

			gotAffinityPodsMaps, gotAntiAffinityPodsMaps, err := getTPMapMatchingIncomingAffinityAntiAffinity(tt.pod, nodeInfoMap)
			if (err != nil) != tt.wantErr {
				t.Errorf("getTPMapMatchingIncomingAffinityAntiAffinity() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotAffinityPodsMaps, tt.wantAffinityPodsMaps) {
				t.Errorf("getTPMapMatchingIncomingAffinityAntiAffinity() gotAffinityPodsMaps = %#v, want %#v", gotAffinityPodsMaps, tt.wantAffinityPodsMaps)
			}
			if !reflect.DeepEqual(gotAntiAffinityPodsMaps, tt.wantAntiAffinityPodsMaps) {
				t.Errorf("getTPMapMatchingIncomingAffinityAntiAffinity() gotAntiAffinityPodsMaps = %#v, want %#v", gotAntiAffinityPodsMaps, tt.wantAntiAffinityPodsMaps)
			}
		})
	}
}

func TestPodMatchesSpreadConstraint(t *testing.T) {
	tests := []struct {
		name       string
		podLabels  map[string]string
		constraint v1.TopologySpreadConstraint
		want       bool
		wantErr    bool
	}{
		{
			name:      "normal match",
			podLabels: map[string]string{"foo": "", "bar": ""},
			constraint: v1.TopologySpreadConstraint{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: metav1.LabelSelectorOpExists,
						},
					},
				},
			},
			want: true,
		},
		{
			name:      "normal mismatch",
			podLabels: map[string]string{"foo": "", "baz": ""},
			constraint: v1.TopologySpreadConstraint{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: metav1.LabelSelectorOpExists,
						},
						{
							Key:      "bar",
							Operator: metav1.LabelSelectorOpExists,
						},
					},
				},
			},
			want: false,
		},
		{
			name: "podLabels is nil",
			constraint: v1.TopologySpreadConstraint{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: metav1.LabelSelectorOpExists,
						},
					},
				},
			},
			want: false,
		},
		{
			name: "constraint.LabelSelector is nil",
			podLabels: map[string]string{
				"foo": "",
				"bar": "",
			},
			constraint: v1.TopologySpreadConstraint{
				MaxSkew: 1,
			},
			want: false,
		},
		{
			name: "both podLabels and constraint.LabelSelector are nil",
			constraint: v1.TopologySpreadConstraint{
				MaxSkew: 1,
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podLabelSet := labels.Set(tt.podLabels)
			got, err := PodMatchesSpreadConstraint(podLabelSet, tt.constraint)
			if (err != nil) != tt.wantErr {
				t.Errorf("PodMatchesSpreadConstraint() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("PodMatchesSpreadConstraint() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetTPMapMatchingSpreadConstraints(t *testing.T) {
	// we need to inject the exact pod pointers to want.topologyPairsMaps.topologyPairToPods
	// otherwise, *pod (as key of a map) will always fail in reflect.DeepEqual()
	tests := []struct {
		name              string
		pod               *v1.Pod
		nodes             []*v1.Node
		existingPods      []*v1.Pod
		injectPodPointers map[topologyPair][]int
		want              *topologyPairsPodSpreadMap
	}{
		{
			name: "clean cluster with one spreadConstraint",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				// denotes no existing pod is matched on this zone pair, but still needed to be
				// calculated if incoming pod matches its own spread constraints
				{key: "zone", value: "zone1"}: {},
				{key: "zone", value: "zone2"}: {},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: make(map[string]topologyPairSet),
				},
			},
		},
		{
			name: "normal case with one spreadConstraint",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				// denotes existingPods[0,1,2]
				{key: "zone", value: "zone1"}: {0, 1, 2},
				// denotes existingPods[3,4]
				{key: "zone", value: "zone2"}: {3, 4},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 2},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1"),
						"p-a2_": newPairSet("zone", "zone1"),
						"p-b1_": newPairSet("zone", "zone1"),
						"p-y1_": newPairSet("zone", "zone2"),
						"p-y2_": newPairSet("zone", "zone2"),
					},
				},
			},
		},
		{
			name: "namespace mismatch doesn't count",
			pod: st.MakePod().Name("p").Label("foo", "").SpreadConstraint(
				1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj(),
			).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Namespace("ns1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Namespace("ns2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}: {0, 2},
				{key: "zone", value: "zone2"}: {4},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1"),
						"p-b1_": newPairSet("zone", "zone1"),
						"p-y2_": newPairSet("zone", "zone2"),
					},
				},
			},
		},
		{
			name: "normal case with two spreadConstraints",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {0, 1, 2},
				{key: "zone", value: "zone2"}:  {3, 4, 5, 6},
				{key: "node", value: "node-a"}: {0, 1},
				{key: "node", value: "node-b"}: {2},
				{key: "node", value: "node-x"}: {},
				{key: "node", value: "node-y"}: {3, 4, 5, 6},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 3, "node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-a2_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-b1_": newPairSet("zone", "zone1", "node", "node-b"),
						"p-y1_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y2_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y3_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y4_": newPairSet("zone", "zone2", "node", "node-y"),
					},
				},
			},
		},
		{
			name: "soft spreadConstraints should be bypassed",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {0, 1, 2},
				{key: "zone", value: "zone2"}:  {3, 4, 5, 6},
				{key: "node", value: "node-a"}: {0, 1},
				{key: "node", value: "node-b"}: {2},
				{key: "node", value: "node-y"}: {3, 4, 5, 6},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 3, "node": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-a2_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-b1_": newPairSet("zone", "zone1", "node", "node-b"),
						"p-y1_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y2_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y3_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y4_": newPairSet("zone", "zone2", "node", "node-y"),
					},
				},
			},
		},
		{
			name: "different labelSelectors - simple version",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a").Node("node-a").Label("foo", "").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {0},
				{key: "zone", value: "zone2"}:  {},
				{key: "node", value: "node-a"}: {},
				{key: "node", value: "node-b"}: {},
				{key: "node", value: "node-y"}: {},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 0, "node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a_": newPairSet("zone", "zone1"),
					},
				},
			},
		},
		{
			name: "different labelSelectors - complex version",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Label("bar", "").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {0, 1, 2},
				{key: "zone", value: "zone2"}:  {3, 4, 5, 6},
				{key: "node", value: "node-a"}: {1},
				{key: "node", value: "node-b"}: {},
				{key: "node", value: "node-y"}: {4, 6},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 3, "node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1"),
						"p-a2_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-b1_": newPairSet("zone", "zone1"),
						"p-y1_": newPairSet("zone", "zone2"),
						"p-y2_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y3_": newPairSet("zone", "zone2"),
						"p-y4_": newPairSet("zone", "zone2", "node", "node-y"),
					},
				},
			},
		},
		{
			name: "two spreadConstraints, and with podAffinity",
			pod: st.MakePod().Name("p").Label("foo", "").
				NodeAffinityNotIn("node", []string{"node-x"}). // exclude node-x
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {0, 1, 2},
				{key: "zone", value: "zone2"}:  {3, 4, 5, 6},
				{key: "node", value: "node-a"}: {0, 1},
				{key: "node", value: "node-b"}: {2},
				{key: "node", value: "node-y"}: {3, 4, 5, 6},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 3, "node": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-a2_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-b1_": newPairSet("zone", "zone1", "node", "node-b"),
						"p-y1_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y2_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y3_": newPairSet("zone", "zone2", "node", "node-y"),
						"p-y4_": newPairSet("zone", "zone2", "node", "node-y"),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.want.topologyPairToPods = make(map[topologyPair]podSet)
			for pair, indexes := range tt.injectPodPointers {
				pSet := make(podSet)
				for _, i := range indexes {
					pSet[tt.existingPods[i]] = struct{}{}
				}
				tt.want.topologyPairToPods[pair] = pSet
			}
			nodeInfoMap := schedulernodeinfo.CreateNodeNameToInfoMap(tt.existingPods, tt.nodes)
			if got, _ := getTPMapMatchingSpreadConstraints(tt.pod, nodeInfoMap); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getTPMapMatchingSpreadConstraints() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPodSpreadMap_addPod(t *testing.T) {
	tests := []struct {
		name              string
		preemptorPod      *v1.Pod
		addedPod          *v1.Pod
		existingPods      []*v1.Pod
		nodeIdx           int // denotes which node 'addedPod' belongs to
		nodes             []*v1.Node
		injectPodPointers map[topologyPair][]int // non-negative index refers to existingPods[i], negative index refers to addedPod
		want              *topologyPairsPodSpreadMap
	}{
		{
			name: "node a and b both impact current min match",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod:     st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: nil, // it's an empty cluster
			nodeIdx:      0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "node", value: "node-a"}: {-1},
				{key: "node", value: "node-b"}: {},
			},
			want: &topologyPairsPodSpreadMap{
				// min match map shouldn't be changed b/c node-b is still on the critical path
				// determining min match
				topologyKeyToMinPodsMap: map[string]int32{"node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("node", "node-a"),
					},
				},
			},
		},
		{
			name: "only node a impacts current min match",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "node", value: "node-a"}: {-1},
				{key: "node", value: "node-b"}: {0},
			},
			want: &topologyPairsPodSpreadMap{
				// min match should be changed from 0 to 1
				topologyKeyToMinPodsMap: map[string]int32{"node": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("node", "node-a"),
						"p-b1_": newPairSet("node", "node-b"),
					},
				},
			},
		},
		{
			name: "add a pod with mis-matched namespace doesn't change topologyKeyToMinPodsMap",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Namespace("ns1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "node", value: "node-a"}: {},
				{key: "node", value: "node-b"}: {0},
			},
			want: &topologyPairsPodSpreadMap{
				// min match remains the same
				topologyKeyToMinPodsMap: map[string]int32{"node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						// "p-a1_": newPairSet("node", "node-a") shouldn't exist
						"p-b1_": newPairSet("node", "node-b"),
					},
				},
			},
		},
		{
			name: "add pod on non-critical node won't trigger re-calculation",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-b2").Node("node-b").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodeIdx: 1,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "node", value: "node-a"}: {},
				{key: "node", value: "node-b"}: {-1, 0},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-b1_": newPairSet("node", "node-b"),
						"p-b2_": newPairSet("node", "node-b"),
					},
				},
			},
		},
		{
			name: "node a and x both impact topologyKeyToMinPodsMap on zone and node",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod:     st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: nil, // it's an empty cluster
			nodeIdx:      0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {-1},
				{key: "zone", value: "zone2"}:  {},
				{key: "node", value: "node-a"}: {-1},
				{key: "node", value: "node-x"}: {},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 0, "node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
					},
				},
			},
		},
		{
			name: "only node a impacts topologyKeyToMinPodsMap on zone and node",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {-1},
				{key: "zone", value: "zone2"}:  {0},
				{key: "node", value: "node-a"}: {-1},
				{key: "node", value: "node-x"}: {0},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1, "node": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-x1_": newPairSet("zone", "zone2", "node", "node-x"),
					},
				},
			},
		},
		{
			name: "node a impacts topologyKeyToMinPodsMap on node, node x impacts topologyKeyToMinPodsMap on zone",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-b2").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {-1, 0, 1},
				{key: "zone", value: "zone2"}:  {2},
				{key: "node", value: "node-a"}: {-1},
				{key: "node", value: "node-b"}: {0, 1},
				{key: "node", value: "node-x"}: {2},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1, "node": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-b1_": newPairSet("zone", "zone1", "node", "node-b"),
						"p-b2_": newPairSet("zone", "zone1", "node", "node-b"),
						"p-x1_": newPairSet("zone", "zone2", "node", "node-x"),
					},
				},
			},
		},
		{
			name: "constraints hold different labelSelectors, node a impacts topologyKeyToMinPodsMap on node",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-x2").Node("node-x").Label("bar", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {-1, 0},
				{key: "zone", value: "zone2"}:  {1},
				{key: "node", value: "node-a"}: {},
				{key: "node", value: "node-b"}: {0},
				{key: "node", value: "node-x"}: {2},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1, "node": 0},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1"),
						"p-b1_": newPairSet("zone", "zone1", "node", "node-b"),
						"p-x1_": newPairSet("zone", "zone2"),
						"p-x2_": newPairSet("node", "node-x"),
					},
				},
			},
		},
		{
			name: "constraints hold different labelSelectors, node a impacts topologyKeyToMinPodsMap on both zone and node",
			preemptorPod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			addedPod: st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Label("bar", "").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("bar", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-x2").Node("node-x").Label("bar", "").Obj(),
			},
			nodeIdx: 0,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {-1},
				{key: "zone", value: "zone2"}:  {1},
				{key: "node", value: "node-a"}: {-1},
				{key: "node", value: "node-b"}: {0},
				{key: "node", value: "node-x"}: {2},
			},
			want: &topologyPairsPodSpreadMap{
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1, "node": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-b1_": newPairSet("node", "node-b"),
						"p-x1_": newPairSet("zone", "zone2"),
						"p-x2_": newPairSet("node", "node-x"),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.want.topologyPairToPods = make(map[topologyPair]podSet)
			for pair, indexes := range tt.injectPodPointers {
				pSet := make(podSet)
				for _, i := range indexes {
					if i >= 0 {
						pSet[tt.existingPods[i]] = struct{}{}
					} else {
						pSet[tt.addedPod] = struct{}{}
					}
				}
				tt.want.topologyPairToPods[pair] = pSet
			}

			nodeInfoMap := schedulernodeinfo.CreateNodeNameToInfoMap(tt.existingPods, tt.nodes)
			podSpreadMap, _ := getTPMapMatchingSpreadConstraints(tt.preemptorPod, nodeInfoMap)

			podSpreadMap.addPod(tt.addedPod, tt.preemptorPod, tt.nodes[tt.nodeIdx])
			if !reflect.DeepEqual(podSpreadMap, tt.want) {
				t.Errorf("podSpreadMap#addPod() = %v, want %v", podSpreadMap, tt.want)
			}
		})
	}
}

func TestPodSpreadMap_removePod(t *testing.T) {
	tests := []struct {
		name              string
		preemptor         *v1.Pod // preemptor pod
		nodes             []*v1.Node
		existingPods      []*v1.Pod
		deletedPodIdx     int     // need to reuse *Pod of existingPods[i]
		deletedPod        *v1.Pod // if deletedPodIdx is invalid, this field is bypassed
		injectPodPointers map[topologyPair][]int
		want              *topologyPairsPodSpreadMap
	}{
		{
			// A high priority pod may not be scheduled due to node taints or resource shortage.
			// So preemption is triggered.
			name: "one spreadConstraint on zone, topologyKeyToMinPodsMap unchanged",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
			},
			deletedPodIdx: 0, // remove pod "p-a1"
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}: {1},
				{key: "zone", value: "zone2"}: {2},
			},
			want: &topologyPairsPodSpreadMap{
				// topologyKeyToMinPodsMap actually doesn't change
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-b1_": newPairSet("zone", "zone1"),
						"p-x1_": newPairSet("zone", "zone2"),
					},
				},
			},
		},
		{
			name: "one spreadConstraint on node, topologyKeyToMinPodsMap changed",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			deletedPodIdx: 0, // remove pod "p-a1"
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}: {1},
				{key: "zone", value: "zone2"}: {2, 3},
			},
			want: &topologyPairsPodSpreadMap{
				// topologyKeyToMinPodsMap is expected to be re-calculated from {"zone": 2}
				// to {"zone": 1}
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-b1_": newPairSet("zone", "zone1"),
						"p-x1_": newPairSet("zone", "zone2"),
						"p-y1_": newPairSet("zone", "zone2"),
					},
				},
			},
		},
		{
			name: "delete an irrelevant pod won't help",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a0").Node("node-a").Label("bar", "").Obj(),
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			deletedPodIdx: 0, // remove pod "p-a0"
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}: {1, 2},
				{key: "zone", value: "zone2"}: {3, 4},
			},
			want: &topologyPairsPodSpreadMap{
				// topologyKeyToMinPodsMap is unchanged
				topologyKeyToMinPodsMap: map[string]int32{"zone": 2},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1"),
						"p-b1_": newPairSet("zone", "zone1"),
						"p-x1_": newPairSet("zone", "zone2"),
						"p-y1_": newPairSet("zone", "zone2"),
					},
				},
			},
		},
		{
			name: "delete a non-existing pod won't help",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
			},
			deletedPodIdx: -1,
			deletedPod:    st.MakePod().Name("p-a0").Node("node-a").Label("bar", "").Obj(),
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}: {0, 1},
				{key: "zone", value: "zone2"}: {2, 3},
			},
			want: &topologyPairsPodSpreadMap{
				// topologyKeyToMinPodsMap is unchanged
				topologyKeyToMinPodsMap: map[string]int32{"zone": 2},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1"),
						"p-b1_": newPairSet("zone", "zone1"),
						"p-x1_": newPairSet("zone", "zone2"),
						"p-y1_": newPairSet("zone", "zone2"),
					},
				},
			},
		},
		{
			name: "two spreadConstraints",
			preemptor: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-x2").Node("node-x").Label("foo", "").Obj(),
			},
			deletedPodIdx: 3, // remove pod "p-x1"
			injectPodPointers: map[topologyPair][]int{
				{key: "zone", value: "zone1"}:  {0, 1, 2},
				{key: "zone", value: "zone2"}:  {4},
				{key: "node", value: "node-a"}: {0, 1},
				{key: "node", value: "node-b"}: {2},
				{key: "node", value: "node-x"}: {4},
			},
			want: &topologyPairsPodSpreadMap{
				// topologyKeyToMinPodsMap is expected to be re-calculated from {"zone": 2, "node": 1}
				// to {"zone": 1, "node": 1}
				topologyKeyToMinPodsMap: map[string]int32{"zone": 1, "node": 1},
				topologyPairsMaps: &topologyPairsMaps{
					podToTopologyPairs: map[string]topologyPairSet{
						"p-a1_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-a2_": newPairSet("zone", "zone1", "node", "node-a"),
						"p-b1_": newPairSet("zone", "zone1", "node", "node-b"),
						"p-x2_": newPairSet("zone", "zone2", "node", "node-x"),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.want.topologyPairToPods = make(map[topologyPair]podSet)
			for pair, indexes := range tt.injectPodPointers {
				pSet := make(podSet)
				for _, i := range indexes {
					pSet[tt.existingPods[i]] = struct{}{}
				}
				tt.want.topologyPairToPods[pair] = pSet
			}

			nodeInfoMap := schedulernodeinfo.CreateNodeNameToInfoMap(tt.existingPods, tt.nodes)
			podSpreadMap, _ := getTPMapMatchingSpreadConstraints(tt.preemptor, nodeInfoMap)

			var deletedPod *v1.Pod
			if tt.deletedPodIdx < len(tt.existingPods) && tt.deletedPodIdx >= 0 {
				deletedPod = tt.existingPods[tt.deletedPodIdx]
			} else {
				deletedPod = tt.deletedPod
			}
			podSpreadMap.removePod(deletedPod)
			if !reflect.DeepEqual(podSpreadMap, tt.want) {
				t.Errorf("podSpreadMap#removePod() = %v, want %v", podSpreadMap, tt.want)
			}
		})
	}
}

var (
	hardSpread = v1.DoNotSchedule
	softSpread = v1.ScheduleAnyway
)

func newPairSet(kv ...string) topologyPairSet {
	result := make(topologyPairSet)
	for i := 0; i < len(kv); i += 2 {
		pair := topologyPair{key: kv[i], value: kv[i+1]}
		result[pair] = struct{}{}
	}
	return result
}
