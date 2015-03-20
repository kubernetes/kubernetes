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
	"fmt"
	"reflect"
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
					Source:         api.EventSource{Component: "kubelet"},
					Message:        "Item 1",
					FirstTimestamp: util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					Count:          1,
				},
				{
					Source:         api.EventSource{Component: "scheduler"},
					Message:        "Item 2",
					FirstTimestamp: util.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  util.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
					Count:          1,
				},
				{
					Source:         api.EventSource{Component: "kubelet"},
					Message:        "Item 3",
					FirstTimestamp: util.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  util.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
					Count:          1,
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

func TestDescribers(t *testing.T) {
	first := &api.Event{}
	second := &api.Pod{}
	var third *api.Pod
	testErr := fmt.Errorf("test")
	d := Describers{}
	d.Add(
		func(e *api.Event, p *api.Pod) (string, error) {
			if e != first {
				t.Errorf("first argument not equal: %#v", e)
			}
			if p != second {
				t.Errorf("second argument not equal: %#v", p)
			}
			return "test", testErr
		},
	)
	if out, err := d.DescribeObject(first, second); out != "test" || err != testErr {
		t.Errorf("unexpected result: %s %v", out, err)
	}

	if out, err := d.DescribeObject(first, second, third); out != "" || err == nil {
		t.Errorf("unexpected result: %s %v", out, err)
	} else {
		if noDescriber, ok := err.(ErrNoDescriber); ok {
			if !reflect.DeepEqual(noDescriber.Types, []string{"*api.Event", "*api.Pod", "*api.Pod"}) {
				t.Errorf("unexpected describer: %v", err)
			}
		} else {
			t.Errorf("unexpected error type: %v", err)
		}
	}

	d.Add(
		func(e *api.Event) (string, error) {
			if e != first {
				t.Errorf("first argument not equal: %#v", e)
			}
			return "simpler", testErr
		},
	)
	if out, err := d.DescribeObject(first); out != "simpler" || err != testErr {
		t.Errorf("unexpected result: %s %v", out, err)
	}
}

func TestDefaultDescribers(t *testing.T) {
	out, err := DefaultObjectDescriber.DescribeObject(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&api.Node{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}
}
