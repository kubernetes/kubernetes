/*
Copyright 2014 The Kubernetes Authors.

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

package events

import (
	"sort"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
)

func TestSortableEvents(t *testing.T) {
	// Arrange
	list := SortableEvents([]api.Event{
		{
			Source:         api.EventSource{Component: "kubelet"},
			Message:        "Item 1",
			FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
			LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
			Count:          1,
			Type:           api.EventTypeNormal,
		},
		{
			Source:         api.EventSource{Component: "scheduler"},
			Message:        "Item 2",
			FirstTimestamp: metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
			LastTimestamp:  metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
			Count:          1,
			Type:           api.EventTypeNormal,
		},
		{
			Source:         api.EventSource{Component: "kubelet"},
			Message:        "Item 3",
			FirstTimestamp: metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
			LastTimestamp:  metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
			Count:          1,
			Type:           api.EventTypeNormal,
		},
	})

	// Act
	sort.Sort(list)

	// Assert
	if list[0].Message != "Item 2" ||
		list[1].Message != "Item 3" ||
		list[2].Message != "Item 1" {
		t.Fatal("List is not sorted by time. List: ", list)
	}
}
