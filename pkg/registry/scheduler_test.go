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
package registry

import (
	"math/rand"
	"testing"

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func expectSchedule(scheduler Scheduler, task Pod, expected string, t *testing.T) {
	actual, err := scheduler.Schedule(task)
	expectNoError(t, err)
	if actual != expected {
		t.Errorf("Unexpected scheduling value: %d, expected %d", actual, expected)
	}
}

func TestRoundRobinScheduler(t *testing.T) {
	scheduler := MakeRoundRobinScheduler([]string{"m1", "m2", "m3", "m4"})
	expectSchedule(scheduler, Pod{}, "m1", t)
	expectSchedule(scheduler, Pod{}, "m2", t)
	expectSchedule(scheduler, Pod{}, "m3", t)
	expectSchedule(scheduler, Pod{}, "m4", t)
}

func TestRandomScheduler(t *testing.T) {
	random := rand.New(rand.NewSource(0))
	scheduler := MakeRandomScheduler([]string{"m1", "m2", "m3", "m4"}, *random)
	_, err := scheduler.Schedule(Pod{})
	expectNoError(t, err)
}

func TestFirstFitSchedulerNothingScheduled(t *testing.T) {
	mockRegistry := MockTaskRegistry{}
	scheduler := MakeFirstFitScheduler([]string{"m1", "m2", "m3"}, &mockRegistry)
	expectSchedule(scheduler, Pod{}, "m1", t)
}

func makeTask(host string, hostPorts ...int) Pod {
	networkPorts := []Port{}
	for _, port := range hostPorts {
		networkPorts = append(networkPorts, Port{HostPort: port})
	}
	return Pod{
		CurrentState: TaskState{
			Host: host,
		},
		DesiredState: TaskState{
			Manifest: ContainerManifest{
				Containers: []Container{
					Container{
						Ports: networkPorts,
					},
				},
			},
		},
	}
}

func TestFirstFitSchedulerFirstScheduled(t *testing.T) {
	mockRegistry := MockTaskRegistry{
		tasks: []Pod{
			makeTask("m1", 8080),
		},
	}
	scheduler := MakeFirstFitScheduler([]string{"m1", "m2", "m3"}, &mockRegistry)
	expectSchedule(scheduler, makeTask("", 8080), "m2", t)
}

func TestFirstFitSchedulerFirstScheduledComplicated(t *testing.T) {
	mockRegistry := MockTaskRegistry{
		tasks: []Pod{
			makeTask("m1", 80, 8080),
			makeTask("m2", 8081, 8082, 8083),
			makeTask("m3", 80, 443, 8085),
		},
	}
	scheduler := MakeFirstFitScheduler([]string{"m1", "m2", "m3"}, &mockRegistry)
	expectSchedule(scheduler, makeTask("", 8080, 8081), "m3", t)
}

func TestFirstFitSchedulerFirstScheduledImpossible(t *testing.T) {
	mockRegistry := MockTaskRegistry{
		tasks: []Pod{
			makeTask("m1", 8080),
			makeTask("m2", 8081),
			makeTask("m3", 8080),
		},
	}
	scheduler := MakeFirstFitScheduler([]string{"m1", "m2", "m3"}, &mockRegistry)
	_, err := scheduler.Schedule(makeTask("", 8080, 8081))
	if err == nil {
		t.Error("Unexpected non-error.")
	}
}
