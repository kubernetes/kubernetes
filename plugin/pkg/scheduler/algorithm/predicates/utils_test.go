/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api/v1"
)

// ExampleUtils is a https://blog.golang.org/examples styled unit test.
func ExampleFindLabelsInSet() {
	labelSubset := labels.Set{}
	labelSubset["label1"] = "value1"
	labelSubset["label2"] = "value2"
	// Lets make believe that these pods are on the cluster.
	// Utility functions will inspect their labels, filter them, and so on.
	nsPods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod1",
				Namespace: "ns1",
				Labels: map[string]string{
					"label1": "wontSeeThis",
					"label2": "wontSeeThis",
					"label3": "will_see_this",
				},
			},
		}, // first pod which will be used via the utilities
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod2",
				Namespace: "ns1",
			},
		},

		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod3ThatWeWontSee",
			},
		},
	}
	fmt.Println(FindLabelsInSet([]string{"label1", "label2", "label3"}, nsPods[0].ObjectMeta.Labels)["label3"])
	AddUnsetLabelsToMap(labelSubset, []string{"label1", "label2", "label3"}, nsPods[0].ObjectMeta.Labels)
	fmt.Println(labelSubset)

	for _, pod := range FilterPodsByNamespace(nsPods, "ns1") {
		fmt.Print(pod.Name, ",")
	}
	// Output:
	// will_see_this
	// label1=value1,label2=value2,label3=will_see_this
	// pod1,pod2,
}

func newNamedPod(name, ns string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{Name: name, Namespace: ns},
	}
}

func TestFilterPodsByNamespace(t *testing.T) {
	tests := []struct {
		Pods      []*v1.Pod
		Namespace string
		Expected  []*v1.Pod
	}{
		{
			Pods: []*v1.Pod{
				newNamedPod("pod_1", "case1_ns"),
				newNamedPod("pod_2", "case1_ns"),
				newNamedPod("pod_3", "default"),
				newNamedPod("pod_4", "other_ns"),
			},
			Namespace: "case1_ns",
			Expected: []*v1.Pod{
				newNamedPod("pod_1", "case1_ns"),
				newNamedPod("pod_2", "case1_ns"),
			},
		},
		{
			Pods: []*v1.Pod{
				newNamedPod("pod_1", "default"),
				newNamedPod("pod_2", "default"),
				newNamedPod("pod_3", "default"),
			},
			Namespace: "case2_ns",
			Expected:  []*v1.Pod{},
		},
	}

	for _, tc := range tests {
		pods := FilterPodsByNamespace(tc.Pods, tc.Namespace)
		if pods == nil {
			t.Error("got unexpected nil pods slice")
		}
		if len(pods) != len(tc.Expected) {
			t.Errorf("expect %d pods, got %d pods", len(tc.Expected), len(pods))
		}
		for i := range pods {
			exp := tc.Expected[i]
			pod := pods[i]
			if exp.Name != pod.Name || pod.Namespace != tc.Namespace {
				t.Errorf("expect pod %v/%v, got pod %v/%v",
					exp.Namespace,
					exp.Name,
					pod.Namespace,
					pod.Name,
				)
			}
		}
	}
}
