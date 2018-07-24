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
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
)

// sortableAntiAffinityTerms lets us to sort anti-affinity terms.
type sortableAntiAffinityTerms []matchingPodAntiAffinityTerm

// Less establishes some ordering between two matchingPodAntiAffinityTerms for
// sorting.
func (s sortableAntiAffinityTerms) Less(i, j int) bool {
	t1, t2 := s[i], s[j]
	if t1.node.Name != t2.node.Name {
		return t1.node.Name < t2.node.Name
	}
	if len(t1.term.Namespaces) != len(t2.term.Namespaces) {
		return len(t1.term.Namespaces) < len(t2.term.Namespaces)
	}
	if t1.term.TopologyKey != t2.term.TopologyKey {
		return t1.term.TopologyKey < t2.term.TopologyKey
	}
	if len(t1.term.LabelSelector.MatchLabels) != len(t2.term.LabelSelector.MatchLabels) {
		return len(t1.term.LabelSelector.MatchLabels) < len(t2.term.LabelSelector.MatchLabels)
	}
	return false
}
func (s sortableAntiAffinityTerms) Len() int { return len(s) }
func (s sortableAntiAffinityTerms) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

var _ = sort.Interface(sortableAntiAffinityTerms{})

func sortAntiAffinityTerms(terms map[string][]matchingPodAntiAffinityTerm) {
	for k, v := range terms {
		sortableTerms := sortableAntiAffinityTerms(v)
		sort.Sort(sortableTerms)
		terms[k] = sortableTerms
	}
}

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

func sortNodePodMap(np map[string][]*v1.Pod) {
	for _, pl := range np {
		sortablePods := sortablePods(pl)
		sort.Sort(sortablePods)
	}
}

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
	sortAntiAffinityTerms(meta1.matchingAntiAffinityTerms)
	sortAntiAffinityTerms(meta2.matchingAntiAffinityTerms)
	if !reflect.DeepEqual(meta1.matchingAntiAffinityTerms, meta2.matchingAntiAffinityTerms) {
		return fmt.Errorf("matchingAntiAffinityTerms are not euqal")
	}
	sortNodePodMap(meta1.nodeNameToMatchingAffinityPods)
	sortNodePodMap(meta2.nodeNameToMatchingAffinityPods)
	if !reflect.DeepEqual(meta1.nodeNameToMatchingAffinityPods, meta2.nodeNameToMatchingAffinityPods) {
		return fmt.Errorf("nodeNameToMatchingAffinityPods are not euqal")
	}
	sortNodePodMap(meta1.nodeNameToMatchingAntiAffinityPods)
	sortNodePodMap(meta2.nodeNameToMatchingAntiAffinityPods)
	if !reflect.DeepEqual(meta1.nodeNameToMatchingAntiAffinityPods, meta2.nodeNameToMatchingAntiAffinityPods) {
		return fmt.Errorf("nodeNameToMatchingAntiAffinityPods are not euqal")
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
			allPodLister := schedulertesting.FakePodLister(append(test.existingPods, test.addedPod))
			// getMeta creates predicate meta data given the list of pods.
			getMeta := func(lister schedulertesting.FakePodLister) (*predicateMetadata, map[string]*schedulercache.NodeInfo) {
				nodeInfoMap := schedulercache.CreateNodeNameToInfoMap(lister, test.nodes)
				// nodeList is a list of non-pointer nodes to feed to FakeNodeListInfo.
				nodeList := []v1.Node{}
				for _, n := range test.nodes {
					nodeList = append(nodeList, *n)
				}
				_, precompute := NewServiceAffinityPredicate(lister, schedulertesting.FakeServiceLister(test.services), FakeNodeListInfo(nodeList), nil)
				RegisterPredicateMetadataProducer("ServiceAffinityMetaProducer", precompute)
				pmf := PredicateMetadataFactory{lister}
				meta := pmf.GetMetadata(test.pendingPod, nodeInfoMap)
				return meta.(*predicateMetadata), nodeInfoMap
			}

			// allPodsMeta is meta data produced when all pods, including test.addedPod
			// are given to the metadata producer.
			allPodsMeta, _ := getMeta(allPodLister)
			// existingPodsMeta1 is meta data produced for test.existingPods (without test.addedPod).
			existingPodsMeta1, nodeInfoMap := getMeta(schedulertesting.FakePodLister(test.existingPods))
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
			existingPodsMeta2, _ := getMeta(schedulertesting.FakePodLister(test.existingPods))
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
		podRequest: &schedulercache.Resource{
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
		matchingAntiAffinityTerms: map[string][]matchingPodAntiAffinityTerm{
			"term1": {
				{
					term: &v1.PodAffinityTerm{TopologyKey: "node"},
					node: &v1.Node{
						ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
					},
				},
			},
		},
		nodeNameToMatchingAffinityPods: map[string][]*v1.Pod{
			"nodeA": {
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
			},
			"nodeC": {
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{
						NodeName: "nodeC",
					},
				},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p6", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeC"},
				},
			},
		},
		nodeNameToMatchingAntiAffinityPods: map[string][]*v1.Pod{
			"nodeN": {
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeN"},
				},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{
						NodeName: "nodeM",
					},
				},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p3"},
					Spec: v1.PodSpec{
						NodeName: "nodeM",
					},
				},
			},
			"nodeM": {
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p6", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeM"},
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
