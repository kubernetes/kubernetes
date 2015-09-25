/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"sort"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// VerifyDatesInOrder checks the start of each line for a RFC1123Z date
// and posts error if all subsequent dates are not equal or increasing
func VerifyDatesInOrder(
	resultToTest, rowDelimiter, columnDelimiter string, t *testing.T) {
	lines := strings.Split(resultToTest, rowDelimiter)
	var previousTime time.Time
	for _, str := range lines {
		columns := strings.Split(str, columnDelimiter)
		if len(columns) > 0 {
			currentTime, err := time.Parse(time.RFC1123Z, columns[0])
			if err == nil {
				if previousTime.After(currentTime) {
					t.Errorf(
						"Output is not sorted by time. %s should be listed after %s. Complete output: %s",
						previousTime.Format(time.RFC1123Z),
						currentTime.Format(time.RFC1123Z),
						resultToTest)
				}
				previousTime = currentTime
			}
		}
	}

}

func TestSortableEvents(t *testing.T) {
	// Arrange
	list := SortableEvents([]api.Event{
		{
			Source:         api.EventSource{Component: "kubelet"},
			Message:        "Item 1",
			FirstTimestamp: unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
			LastTimestamp:  unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
			Count:          1,
		},
		{
			Source:         api.EventSource{Component: "scheduler"},
			Message:        "Item 2",
			FirstTimestamp: unversioned.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
			LastTimestamp:  unversioned.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
			Count:          1,
		},
		{
			Source:         api.EventSource{Component: "kubelet"},
			Message:        "Item 3",
			FirstTimestamp: unversioned.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
			LastTimestamp:  unversioned.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
			Count:          1,
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
