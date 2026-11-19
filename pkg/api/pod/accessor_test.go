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
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
		pod := &core.Pod{
			Spec: core.PodSpec{
				InitContainers: []core.Container{{Name: "init", Args: []string{"a"}}},
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetSpecInitContainers()
		if !reflect.DeepEqual(got, pod.Spec.InitContainers) {
			t.Errorf("expected %v, got %v", pod.Spec.InitContainers, got)
		}
		got[0].Name = "mutated"
		got[0].Args[0] = "mutated"
		if pod.Spec.InitContainers[0].Name != "init" || pod.Spec.InitContainers[0].Args[0] != "a" {
			t.Errorf("mutating returned slice modified underlying pod: %+v", pod.Spec.InitContainers)
		}

		emptySlice, err := Accessor(&core.Pod{Spec: core.PodSpec{InitContainers: []core.Container{}}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := emptySlice.GetSpecInitContainers(); got == nil || len(got) != 0 {
			t.Errorf("expected non-nil empty slice, got %#v", got)
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
		pod := &core.Pod{
			Spec: core.PodSpec{
				Containers: []core.Container{{Name: "app", Args: []string{"a"}}},
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetSpecContainers()
		if !reflect.DeepEqual(got, pod.Spec.Containers) {
			t.Errorf("expected %v, got %v", pod.Spec.Containers, got)
		}
		got[0].Name = "mutated"
		got[0].Args[0] = "mutated"
		if pod.Spec.Containers[0].Name != "app" || pod.Spec.Containers[0].Args[0] != "a" {
			t.Errorf("mutating returned slice modified underlying pod: %+v", pod.Spec.Containers)
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
		pod := &core.Pod{
			Spec: core.PodSpec{
				EphemeralContainers: []core.EphemeralContainer{{
					EphemeralContainerCommon: core.EphemeralContainerCommon{
						Name: "debug",
						Args: []string{"a"},
					},
				}},
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetSpecEphemeralContainers()
		if !reflect.DeepEqual(got, pod.Spec.EphemeralContainers) {
			t.Errorf("expected %v, got %v", pod.Spec.EphemeralContainers, got)
		}
		got[0].Name = "mutated"
		got[0].Args[0] = "mutated"
		if pod.Spec.EphemeralContainers[0].Name != "debug" || pod.Spec.EphemeralContainers[0].Args[0] != "a" {
			t.Errorf("mutating returned slice modified underlying pod: %+v", pod.Spec.EphemeralContainers)
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
		want := int64(42)
		populated, err := Accessor(&core.Pod{
			Spec: core.PodSpec{
				TerminationGracePeriodSeconds: &want,
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got, ok := populated.GetSpecTerminationGracePeriodSeconds(); !ok || got != want {
			t.Errorf("expected (%v, true), got (%v, %v)", want, got, ok)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got, ok := empty.GetSpecTerminationGracePeriodSeconds(); ok || got != 0 {
			t.Errorf("expected (0, false), got (%v, %v)", got, ok)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got, ok := nilReceiver.GetSpecTerminationGracePeriodSeconds(); ok || got != 0 {
			t.Errorf("expected (0, false) on nil core.Pod, got (%v, %v)", got, ok)
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
		want := &core.Affinity{
			PodAntiAffinity: &core.PodAntiAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{{TopologyKey: "zone"}},
			},
		}
		pod := &core.Pod{
			Spec: core.PodSpec{
				Affinity: want,
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetSpecAffinity()
		if !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}
		if got == want {
			t.Errorf("expected copied *core.Affinity, got same pointer")
		}
		got.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].TopologyKey = "mutated"
		if pod.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].TopologyKey != "zone" {
			t.Errorf("mutating returned Affinity modified underlying pod")
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
		pod := &core.Pod{
			Status: core.PodStatus{
				HostIPs: []core.HostIP{{IP: "10.0.0.1"}},
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetStatusHostIPs()
		if !reflect.DeepEqual(got, pod.Status.HostIPs) {
			t.Errorf("expected %v, got %v", pod.Status.HostIPs, got)
		}
		got[0].IP = "mutated"
		if pod.Status.HostIPs[0].IP != "10.0.0.1" {
			t.Errorf("mutating returned slice modified underlying pod: %+v", pod.Status.HostIPs)
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
		pod := &core.Pod{
			Status: core.PodStatus{
				PodIPs: []core.PodIP{{IP: "10.0.0.2"}},
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetStatusPodIPs()
		if !reflect.DeepEqual(got, pod.Status.PodIPs) {
			t.Errorf("expected %v, got %v", pod.Status.PodIPs, got)
		}
		got[0].IP = "mutated"
		if pod.Status.PodIPs[0].IP != "10.0.0.2" {
			t.Errorf("mutating returned slice modified underlying pod: %+v", pod.Status.PodIPs)
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
	t.Run("GetStatusContainerStatuses", func(t *testing.T) {
		started := true
		pod := &core.Pod{
			Status: core.PodStatus{
				ContainerStatuses: []core.ContainerStatus{{Name: "app", Started: &started}},
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetStatusContainerStatuses()
		if !reflect.DeepEqual(got, pod.Status.ContainerStatuses) {
			t.Errorf("expected %v, got %v", pod.Status.ContainerStatuses, got)
		}
		got[0].Name = "mutated"
		*got[0].Started = false
		if pod.Status.ContainerStatuses[0].Name != "app" || !*pod.Status.ContainerStatuses[0].Started {
			t.Errorf("mutating returned slice modified underlying pod: %+v", pod.Status.ContainerStatuses)
		}

		empty, err := Accessor(&core.Pod{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := empty.GetStatusContainerStatuses(); got != nil {
			t.Errorf("expected nil, got %v", got)
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if got := nilReceiver.GetStatusContainerStatuses(); got != nil {
			t.Errorf("expected nil on nil core.Pod, got %v", got)
		}
	})
	t.Run("GetStatusResourceClaimStatuses", func(t *testing.T) {
		claimName := "claim-a"
		pod := &core.Pod{
			Status: core.PodStatus{
				ResourceClaimStatuses: []core.PodResourceClaimStatus{{
					Name:              "req",
					ResourceClaimName: &claimName,
				}},
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetStatusResourceClaimStatuses()
		if !reflect.DeepEqual(got, pod.Status.ResourceClaimStatuses) {
			t.Errorf("expected %v, got %v", pod.Status.ResourceClaimStatuses, got)
		}
		got[0].Name = "mutated"
		*got[0].ResourceClaimName = "mutated-claim"
		if pod.Status.ResourceClaimStatuses[0].Name != "req" || *pod.Status.ResourceClaimStatuses[0].ResourceClaimName != "claim-a" {
			t.Errorf("mutating returned slice modified underlying pod: %+v", pod.Status.ResourceClaimStatuses)
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
		want := &core.PodExtendedResourceClaimStatus{
			ResourceClaimName: "ext-claim",
			RequestMappings: []core.ContainerExtendedResourceRequest{{
				ContainerName: "app",
				ResourceName:  "example.com/gpu",
				RequestName:   "req-0",
			}},
		}
		pod := &core.Pod{
			Status: core.PodStatus{
				ExtendedResourceClaimStatus: want,
			},
		}
		populated, err := Accessor(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got := populated.GetStatusExtendedResourceClaimStatus()
		if !reflect.DeepEqual(got, want) {
			t.Errorf("expected %v, got %v", want, got)
		}
		if got == want {
			t.Errorf("expected copied *core.PodExtendedResourceClaimStatus, got same pointer")
		}
		got.ResourceClaimName = "mutated"
		got.RequestMappings[0].RequestName = "mutated"
		if pod.Status.ExtendedResourceClaimStatus.ResourceClaimName != "ext-claim" ||
			pod.Status.ExtendedResourceClaimStatus.RequestMappings[0].RequestName != "req-0" {
			t.Errorf("mutating returned ExtendedResourceClaimStatus modified underlying pod")
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
	t.Run("ObjectMetaMutationIsolation", func(t *testing.T) {
		now := metav1.NewTime(time.Unix(1700000000, 0).UTC())
		grace := int64(30)
		isController := true
		fieldsV1 := metav1.FieldsV1{}
		fieldsV1.SetRawBytes([]byte(`{"f:metadata":{}}`))
		pod := &core.Pod{
			ObjectMeta: metav1.ObjectMeta{
				DeletionTimestamp:          &now,
				DeletionGracePeriodSeconds: &grace,
				Labels:                     map[string]string{"app": "v1"},
				Annotations:                map[string]string{"note": "v1"},
				Finalizers:                 []string{"f1"},
				OwnerReferences:            []metav1.OwnerReference{{Name: "rs", Controller: &isController}},
				ManagedFields:              []metav1.ManagedFieldsEntry{{Manager: "mgr", FieldsV1: &fieldsV1}},
			},
		}
		acc := MustAccessor(pod)

		gotTS := acc.GetDeletionTimestamp()
		gotTS.Time = time.Unix(0, 0)
		if !pod.DeletionTimestamp.Equal(&now) {
			t.Errorf("mutating GetDeletionTimestamp modified underlying pod")
		}

		gotGrace := acc.GetDeletionGracePeriodSeconds()
		*gotGrace = 99
		if *pod.DeletionGracePeriodSeconds != 30 {
			t.Errorf("mutating GetDeletionGracePeriodSeconds modified underlying pod")
		}

		gotLabels := acc.GetLabels()
		gotLabels["app"] = "mutated"
		if pod.Labels["app"] != "v1" {
			t.Errorf("mutating GetLabels modified underlying pod")
		}
		inputLabels := map[string]string{"app": "v2"}
		acc.SetLabels(inputLabels)
		inputLabels["app"] = "after-set"
		if pod.Labels["app"] != "v2" {
			t.Errorf("mutating input map after SetLabels modified underlying pod: %v", pod.Labels)
		}

		gotAnnotations := acc.GetAnnotations()
		gotAnnotations["note"] = "mutated"
		if pod.Annotations["note"] != "v1" {
			t.Errorf("mutating GetAnnotations modified underlying pod")
		}
		inputAnnotations := map[string]string{"note": "v2"}
		acc.SetAnnotations(inputAnnotations)
		inputAnnotations["note"] = "after-set"
		if pod.Annotations["note"] != "v2" {
			t.Errorf("mutating input map after SetAnnotations modified underlying pod: %v", pod.Annotations)
		}

		gotFinalizers := acc.GetFinalizers()
		gotFinalizers[0] = "mutated"
		if pod.Finalizers[0] != "f1" {
			t.Errorf("mutating GetFinalizers modified underlying pod")
		}
		inputFinalizers := []string{"f2"}
		acc.SetFinalizers(inputFinalizers)
		inputFinalizers[0] = "after-set"
		if pod.Finalizers[0] != "f2" {
			t.Errorf("mutating input slice after SetFinalizers modified underlying pod: %v", pod.Finalizers)
		}

		gotOwners := acc.GetOwnerReferences()
		*gotOwners[0].Controller = false
		if !*pod.OwnerReferences[0].Controller {
			t.Errorf("mutating GetOwnerReferences modified underlying pod")
		}
		inputCtrl := true
		inputOwners := []metav1.OwnerReference{{Name: "rs2", Controller: &inputCtrl}}
		acc.SetOwnerReferences(inputOwners)
		inputCtrl = false
		if !*pod.OwnerReferences[0].Controller {
			t.Errorf("mutating input pointer after SetOwnerReferences modified underlying pod")
		}

		gotMF := acc.GetManagedFields()
		gotMF[0].Manager = "mutated"
		if pod.ManagedFields[0].Manager != "mgr" {
			t.Errorf("mutating GetManagedFields modified underlying pod")
		}
		inputMF := []metav1.ManagedFieldsEntry{{Manager: "mgr2"}}
		acc.SetManagedFields(inputMF)
		inputMF[0].Manager = "after-set"
		if pod.ManagedFields[0].Manager != "mgr2" {
			t.Errorf("mutating input slice after SetManagedFields modified underlying pod")
		}

		newTS := metav1.NewTime(time.Unix(1800000000, 0).UTC())
		acc.SetDeletionTimestamp(&newTS)
		newTS.Time = time.Unix(0, 0)
		if pod.DeletionTimestamp.Unix() != 1800000000 {
			t.Errorf("mutating input after SetDeletionTimestamp modified underlying pod")
		}

		newGrace := int64(60)
		acc.SetDeletionGracePeriodSeconds(&newGrace)
		newGrace = 1
		if *pod.DeletionGracePeriodSeconds != 60 {
			t.Errorf("mutating input after SetDeletionGracePeriodSeconds modified underlying pod")
		}

		var nilReceiver PodAccessor = (*podAccessor)(nil)
		if nilReceiver.DeepCopyObject() != nil ||
			nilReceiver.GetDeletionTimestamp() != nil ||
			nilReceiver.GetDeletionGracePeriodSeconds() != nil ||
			nilReceiver.GetLabels() != nil ||
			nilReceiver.GetAnnotations() != nil ||
			nilReceiver.GetFinalizers() != nil ||
			nilReceiver.GetOwnerReferences() != nil ||
			nilReceiver.GetManagedFields() != nil {
			t.Errorf("expected nil returns on nil *podAccessor receiver")
		}
		nilReceiver.SetDeletionTimestamp(&now)
		nilReceiver.SetDeletionGracePeriodSeconds(&grace)
		nilReceiver.SetLabels(map[string]string{"k": "v"})
		nilReceiver.SetAnnotations(map[string]string{"k": "v"})
		nilReceiver.SetFinalizers([]string{"f"})
		nilReceiver.SetOwnerReferences([]metav1.OwnerReference{{Name: "o"}})
		nilReceiver.SetManagedFields([]metav1.ManagedFieldsEntry{{Manager: "m"}})
	})
}
