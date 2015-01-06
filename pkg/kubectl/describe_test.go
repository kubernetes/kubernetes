/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type describeClient struct {
	T         *testing.T
	Namespace string
	Err       error
	*client.Fake
}

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
}

func TestDescribePod(t *testing.T) {
	fake := &client.Fake{}
	c := &describeClient{T: t, Namespace: "foo", Fake: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "Status:") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeService(t *testing.T) {
	fake := &client.Fake{}
	c := &describeClient{T: t, Namespace: "foo", Fake: fake}
	d := ServiceDescriber{c}
	out, err := d.Describe("foo", "bar")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "Labels:") || !strings.Contains(out, "bar") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestPodDescribeResultsSorted(t *testing.T) {
	// Arrange
	fake := &client.Fake{
		EventsList: api.EventList{
			Items: []api.Event{
				{
					Source:    api.EventSource{Component: "kubelet"},
					Message:   "Item 1",
					Timestamp: util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				},
				{
					Source:    api.EventSource{Component: "scheduler"},
					Message:   "Item 2",
					Timestamp: util.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				},
				{
					Source:    api.EventSource{Component: "kubelet"},
					Message:   "Item 3",
					Timestamp: util.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				},
			},
		},
	}
	c := &describeClient{T: t, Namespace: "foo", Fake: fake}
	d := PodDescriber{c}

	// Act
	out, err := d.Describe("foo", "bar")

	// Assert
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	VerifyDatesInOrder(out, "\n" /* rowDelimiter */, "\t" /* columnDelimiter */, t)
}
