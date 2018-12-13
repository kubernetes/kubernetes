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
	"k8s.io/apimachinery/pkg/util/sets"
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

func TestIntersect(t *testing.T) {
	tests := []struct {
		name    string
		strsets []sets.String
		want    sets.String
	}{
		{
			name:    "single nil",
			strsets: []sets.String{},
			want:    sets.String{},
		},
		{
			name:    "multiple nils",
			strsets: []sets.String{{}, {}, {}, {}},
			want:    sets.String{},
		},
		{
			name: "no intersaction found",
			strsets: []sets.String{
				{
					"nodeB": struct{}{},
					"nodeC": struct{}{},
				},
				{
					"nodeC": struct{}{},
					"nodeA": struct{}{},
				},
				{
					"nodeA": struct{}{},
					"nodeB": struct{}{},
					"nodeD": struct{}{},
				},
			},
			want: sets.String{},
		},
		{
			name: "intersaction found",
			strsets: []sets.String{
				{
					"nodeB": struct{}{},
					"nodeC": struct{}{},
				},
				{
					"nodeC": struct{}{},
					"nodeA": struct{}{},
					"nodeB": struct{}{},
				},
				{
					"nodeA": struct{}{},
					"nodeB": struct{}{},
					"nodeD": struct{}{},
				},
				{
					"nodeE": struct{}{},
					"nodeB": struct{}{},
				},
			},
			want: sets.String{
				"nodeB": struct{}{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Intersect(tt.strsets); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Intersect() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestUnion(t *testing.T) {
	tests := []struct {
		name    string
		strsets []sets.String
		want    sets.String
	}{
		{
			name:    "single nil",
			strsets: []sets.String{},
			want:    sets.String{},
		},
		{
			name:    "multiple nils",
			strsets: []sets.String{{}, {}, {}, {}},
			want:    sets.String{},
		},
		{
			name: "union case 1",
			strsets: []sets.String{
				{
					"nodeB": struct{}{},
					"nodeC": struct{}{},
				},
				{
					"nodeC": struct{}{},
					"nodeA": struct{}{},
				},
				{
					"nodeA": struct{}{},
					"nodeB": struct{}{},
					"nodeD": struct{}{},
				},
			},
			want: sets.String{
				"nodeA": struct{}{},
				"nodeB": struct{}{},
				"nodeC": struct{}{},
				"nodeD": struct{}{},
			},
		},
		{
			name: "union case 2",
			strsets: []sets.String{
				{
					"nodeB": struct{}{},
				},
				{
					"nodeC": struct{}{},
					"nodeA": struct{}{},
				},
				{
					"nodeA": struct{}{},
					"nodeB": struct{}{},
					"nodeD": struct{}{},
				},
				{
					"nodeE": struct{}{},
					"nodeB": struct{}{},
				},
			},
			want: sets.String{
				"nodeA": struct{}{},
				"nodeB": struct{}{},
				"nodeC": struct{}{},
				"nodeD": struct{}{},
				"nodeE": struct{}{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Union(tt.strsets); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Union() = %v, want %v", got, tt.want)
			}
		})
	}
}
