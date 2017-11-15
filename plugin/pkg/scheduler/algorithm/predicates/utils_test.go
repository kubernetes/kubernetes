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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/diff"
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

func TestFindLabelsInSet(t *testing.T) {

	nsPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: "ns1",
			Labels: map[string]string{
				"label1": "foo",
				"label2": "foo",
				"label3": "bar",
			},
		},
	}

	expected := map[string]string{
		"label1": "foo",
		"label2": "foo",
	}

	result := FindLabelsInSet([]string{"label1", "label2"}, nsPod.ObjectMeta.Labels)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Got different result than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, result))
	}

}

func TestAddUnsetLabelsToMap(t *testing.T) {

	labelSubset := labels.Set{}
	labelSubset["label1"] = "value1"
	labelSubset["label2"] = "value2"

	nsPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: "ns1",
			Labels: map[string]string{
				"label1": "foo",
				"label2": "foo",
				"label3": "bar",
			},
		},
	}

	expected := labels.Set{
		"label1": "value1",
		"label2": "value2",
		"label3": "bar",
	}

	AddUnsetLabelsToMap(labelSubset, []string{"label1", "label2", "label3"}, nsPod.ObjectMeta.Labels)

	if !reflect.DeepEqual(labelSubset, expected) {
		t.Errorf("Got different result than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, labelSubset))
	}

}

func TestFilterPodsByNamespace(t *testing.T) {

	nsPods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod1",
				Namespace: "ns1",
				Labels: map[string]string{
					"label1": "foo",
					"label2": "foo",
					"label3": "bar",
				},
			},
		},
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

	expected := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod1",
				Namespace: "ns1",
				Labels: map[string]string{
					"label1": "foo",
					"label2": "foo",
					"label3": "bar",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod2",
				Namespace: "ns1",
			},
		},
	}

	result := FilterPodsByNamespace(nsPods, "ns1")
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Got different result than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, result))
	}

}
