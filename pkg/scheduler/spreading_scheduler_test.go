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

package scheduler

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestSpreadPriority(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	machine1State := api.PodState{
		Host: "machine1",
	}
	machine2State := api.PodState{
		Host: "machine2",
	}
	tests := []struct {
		pod          api.Pod
		pods         []api.Pod
		nodes        []string
		expectErr    bool
		expectedList HostPriorityList
		test         string
	}{
		{api.Pod{}, []api.Pod{}, []string{"machine1", "machine2"}, false, []HostPriority{{"machine1", 0}, {"machine2", 0}}, "nothing scheduled"},
		{api.Pod{Labels: labels1}, []api.Pod{{CurrentState: machine1State}}, []string{"machine1", "machine2"}, false, []HostPriority{{"machine1", 0}, {"machine2", 0}}, "no labels"},
		{api.Pod{Labels: labels1}, []api.Pod{{CurrentState: machine1State, Labels: labels2}}, []string{"machine1", "machine2"}, false, []HostPriority{{"machine1", 0}, {"machine2", 0}}, "different labels"},
		{api.Pod{Labels: labels1}, []api.Pod{{CurrentState: machine1State, Labels: labels2}, {CurrentState: machine2State, Labels: labels1}}, []string{"machine1", "machine2"}, false, []HostPriority{{"machine1", 0}, {"machine2", 1}}, "one label match"},
		{api.Pod{Labels: labels1}, []api.Pod{{CurrentState: machine1State, Labels: labels2}, {CurrentState: machine1State, Labels: labels1}, {CurrentState: machine2State, Labels: labels1}}, []string{"machine1", "machine2"}, false, []HostPriority{{"machine1", 1}, {"machine2", 1}}, "two label matches on different machines"},
		{api.Pod{Labels: labels1}, []api.Pod{{CurrentState: machine1State, Labels: labels2}, {CurrentState: machine1State, Labels: labels1}, {CurrentState: machine2State, Labels: labels1}, {CurrentState: machine2State, Labels: labels1}}, []string{"machine1", "machine2"}, false, []HostPriority{{"machine1", 1}, {"machine2", 2}}, "three label matches"},
	}

	for _, test := range tests {
		list, err := CalculateSpreadPriority(test.pod, FakePodLister(test.pods), &listMinionLister{test.nodes})
		if test.expectErr {
			if err == nil {
				t.Errorf("%s: unexpected non-error", test.test)
			}
		} else {
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
			}
		}
	}
}
