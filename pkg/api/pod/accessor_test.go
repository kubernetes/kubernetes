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

package pod

import (
	reflect "reflect"
	testing "testing"

	core "k8s.io/kubernetes/pkg/apis/core"
)

func TestPodAccessor(t *testing.T) {
	t.Run("Accessor", func(t *testing.T) {
		nilAcc, err := Accessor((*core.Pod)(nil))
		if err != nil || nilAcc != nil {
			t.Errorf("expected (nil, nil) for typed nil *core.Pod, got (%v, %v)", nilAcc, err)
		}
		if _, err := Accessor("not-a-pod"); err == nil {
			t.Errorf("expected error for invalid object type")
		}
		if got := MustAccessor((*core.Pod)(nil)); got != nil {
			t.Errorf("expected nil for typed nil *core.Pod, got %v", got)
		}
		func() {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected MustAccessor to panic on invalid object type")
				}
			}()
			_ = MustAccessor("not-a-pod")
		}()
	})

	t.Run("GetSpecInitContainers", func(t *testing.T) {
		want := []core.Container{{}}
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				InitContainers: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecInitContainers(); !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecInitContainers(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecInitContainers(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecContainers", func(t *testing.T) {
		want := []core.Container{{}}
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				Containers: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecContainers(); !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecContainers(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecContainers(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecEphemeralContainers", func(t *testing.T) {
		want := []core.EphemeralContainer{{}}
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				EphemeralContainers: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecEphemeralContainers(); !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecEphemeralContainers(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecEphemeralContainers(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecRestartPolicy", func(t *testing.T) {
		want := core.RestartPolicy("test-value")
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				RestartPolicy: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecRestartPolicy(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecRestartPolicy(); got != "" {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecRestartPolicy(); got != "" {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecTerminationGracePeriodSeconds", func(t *testing.T) {
		want := new(int64(42))
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				TerminationGracePeriodSeconds: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecTerminationGracePeriodSeconds(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecTerminationGracePeriodSeconds(); got != nil {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecTerminationGracePeriodSeconds(); got != nil {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecServiceAccountName", func(t *testing.T) {
		want := "test-value"
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				ServiceAccountName: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecServiceAccountName(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecServiceAccountName(); got != "" {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecServiceAccountName(); got != "" {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecNodeName", func(t *testing.T) {
		want := "test-value"
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				NodeName: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecNodeName(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecNodeName(); got != "" {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecNodeName(); got != "" {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecHostNetwork", func(t *testing.T) {
		want := true
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				HostNetwork: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecHostNetwork(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecHostNetwork(); got != false {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecHostNetwork(); got != false {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecAffinity", func(t *testing.T) {
		want := &core.Affinity{}
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				Affinity: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecAffinity(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecAffinity(); got != nil {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecAffinity(); got != nil {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetSpecSchedulerName", func(t *testing.T) {
		want := "test-value"
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				SchedulerName: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetSpecSchedulerName(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetSpecSchedulerName(); got != "" {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetSpecSchedulerName(); got != "" {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusPhase", func(t *testing.T) {
		want := core.PodPhase("test-value")
		populated, err := Accessor(&core.Pod{
			Status: core.PodStatus{
				Phase: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetStatusPhase(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusPhase(); got != "" {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusPhase(); got != "" {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusNominatedNodeName", func(t *testing.T) {
		want := "test-value"
		populated, err := Accessor(&core.Pod{
			Status: core.PodStatus{
				NominatedNodeName: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetStatusNominatedNodeName(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusNominatedNodeName(); got != "" {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusNominatedNodeName(); got != "" {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusHostIPs", func(t *testing.T) {
		want := []core.HostIP{{}}
		populated, err := Accessor(&core.Pod{
			Status: core.PodStatus{
				HostIPs: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetStatusHostIPs(); !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusHostIPs(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusHostIPs(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusPodIPs", func(t *testing.T) {
		want := []core.PodIP{{}}
		populated, err := Accessor(&core.Pod{
			Status: core.PodStatus{
				PodIPs: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetStatusPodIPs(); !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusPodIPs(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusPodIPs(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusResourceClaimStatuses", func(t *testing.T) {
		want := []core.PodResourceClaimStatus{{}}
		populated, err := Accessor(&core.Pod{
			Status: core.PodStatus{
				ResourceClaimStatuses: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetStatusResourceClaimStatuses(); !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusResourceClaimStatuses(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusResourceClaimStatuses(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusExtendedResourceClaimStatus", func(t *testing.T) {
		want := &core.PodExtendedResourceClaimStatus{}
		populated, err := Accessor(&core.Pod{
			Status: core.PodStatus{
				ExtendedResourceClaimStatus: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetStatusExtendedResourceClaimStatus(); got != want {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusExtendedResourceClaimStatus(); got != nil {
			t.Errorf("expected zero value, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusExtendedResourceClaimStatus(); got != nil {
			t.Errorf("expected zero value on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusNodeAllocatableResourceClaimStatuses", func(t *testing.T) {
		want := []core.NodeAllocatableResourceClaimStatus{{}}
		populated, err := Accessor(&core.Pod{
			Status: core.PodStatus{
				NodeAllocatableResourceClaimStatuses: want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := populated.GetStatusNodeAllocatableResourceClaimStatuses(); !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusNodeAllocatableResourceClaimStatuses(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusNodeAllocatableResourceClaimStatuses(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
}
