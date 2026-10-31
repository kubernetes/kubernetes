/*
Copyright The Kubernetes Authors.

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

package state

import (
	"reflect"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/cpuset"
)

// testPendingResize grows c1 of pod from {1,2} to {1,2,3} and the pod-level
// assignment from {1,2,3,4} to {1,2,3,4,5}, taking 5 from the default CPU set,
// and shrinks c2 from {4,6} to {4}, releasing 6.
// This is an artificial scenario as pod-level exclusive CPUs are not scalable.
func testPendingResize() PendingResize {
	return PendingResize{
		PodUID: "pod",
		ContainerCPUSets: map[string]cpuset.CPUSet{
			"c1": cpuset.New(1, 2, 3),
			"c2": cpuset.New(4),
		},
		PodCPUSet: new(cpuset.New(1, 2, 3, 4, 5)),
	}
}

// newTestState returns the state testPendingResize applies to.
func newTestState(s State) State {
	s.SetCPUSet("pod", "c1", cpuset.New(1, 2))
	s.SetCPUSet("pod", "c2", cpuset.New(4, 6))
	s.SetPodCPUSet("pod", cpuset.New(1, 2, 3, 4))
	s.SetDefaultCPUSet(cpuset.New(5, 7))
	return s
}

func TestMemoryStateCommitPendingResize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	logger, _ := ktesting.NewTestContext(t)

	s := newTestState(NewMemoryState(logger))
	s.SetPendingResize(testPendingResize())
	pending, ok := s.GetPendingResize()
	if !ok {
		t.Fatalf("Expected a pending resize, got none")
	}
	if !reflect.DeepEqual(testPendingResize(), pending) {
		t.Errorf("Expected pending resize %+v, got %+v", testPendingResize(), pending)
	}

	s.CommitPendingResize()
	for containerName, expected := range map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 3), "c2": cpuset.New(4)} {
		if actual, _ := s.GetCPUSet("pod", containerName); !actual.Equals(expected) {
			t.Errorf("Expected %s to have CPUs %s, got %s", containerName, expected, actual)
		}
	}
	if actual, _ := s.GetPodCPUSet("pod"); !actual.Equals(cpuset.New(1, 2, 3, 4, 5)) {
		t.Errorf("Expected the pod to have CPUs 1-5, got %s", actual)
	}
	if actual := s.GetDefaultCPUSet(); !actual.Equals(cpuset.New(6, 7)) {
		t.Errorf("Expected default CPU set 6-7, got %s", actual)
	}
	if _, ok := s.GetPendingResize(); ok {
		t.Errorf("Expected the pending resize to be deleted by the commit")
	}
	if baseline, _ := s.GetBaselineCPUSet("pod", "c1"); !baseline.Equals(cpuset.New(1, 2)) {
		t.Errorf("Expected the commit to keep the baseline 1-2, got %s", baseline)
	}
}

func TestMemoryStateSetPendingResizeReplaces(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	s := newTestState(NewMemoryState(logger))
	s.SetPendingResize(PendingResize{PodUID: "other", ContainerCPUSets: map[string]cpuset.CPUSet{"c": cpuset.New(7)}})
	s.SetPendingResize(testPendingResize())
	pending, ok := s.GetPendingResize()
	if !ok || !reflect.DeepEqual(testPendingResize(), pending) {
		t.Errorf("Expected the second pending resize %+v to replace the first, got %+v (found %t)", testPendingResize(), pending, ok)
	}
}

func TestMemoryStateGetPendingResizeReturnsCopy(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	s := newTestState(NewMemoryState(logger))
	s.SetPendingResize(testPendingResize())
	pending, _ := s.GetPendingResize()
	pending.ContainerCPUSets["c1"] = cpuset.New(7)
	*pending.PodCPUSet = cpuset.New(7)

	if actual, _ := s.GetPendingResize(); !reflect.DeepEqual(testPendingResize(), actual) {
		t.Errorf("Expected changes to a returned pending resize not to change the state, got %+v", actual)
	}
}

func TestMemoryStateDeletePendingResize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	logger, _ := ktesting.NewTestContext(t)

	s := newTestState(NewMemoryState(logger))
	s.SetPendingResize(testPendingResize())
	s.DeletePendingResize()
	if _, ok := s.GetPendingResize(); ok {
		t.Errorf("Expected the pending resize to be deleted")
	}
	if actual, _ := s.GetCPUSet("pod", "c1"); !actual.Equals(cpuset.New(1, 2)) {
		t.Errorf("Expected the delete not to apply the resize, c1 has %s", actual)
	}
}

func TestMemoryStateClearStateDeletesPendingResize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	logger, _ := ktesting.NewTestContext(t)

	s := newTestState(NewMemoryState(logger))
	s.SetPendingResize(testPendingResize())
	s.ClearState()
	if pending, ok := s.GetPendingResize(); ok {
		t.Errorf("Expected ClearState to delete the pending resize, got %+v", pending)
	}
}
