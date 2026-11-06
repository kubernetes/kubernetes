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

package memorymanager

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	kubelettypes "k8s.io/kubelet/pkg/types"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type checkpointRecoveryRuntime struct {
	list  func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error)
	calls int
}

func (r *checkpointRecoveryRuntime) UpdateContainerResources(context.Context, string, *runtimeapi.ContainerResources) error {
	return errors.New("recovery must not change runtime resources")
}

func (r *checkpointRecoveryRuntime) ListContainers(ctx context.Context, filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	r.calls++
	return r.list(ctx, filter)
}

func startCheckpointRecoveryManager(t *testing.T, dir string, pods []*v1.Pod, rt runtimeService, initial containermap.ContainerMap) (Manager, error) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("Memory Manager static policy is not available on Windows")
	}
	tCtx := ktesting.Init(t)
	info := returnMachineInfo()
	mgr, err := NewManager(tCtx.Logger(), string(PolicyTypeStatic), &info,
		v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(2*gb, resource.BinarySI)},
		[]kubeletconfig.MemoryReservation{
			{NumaNode: 0, Limits: v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI)}},
			{NumaNode: 1, Limits: v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI)}},
		}, dir, topologymanager.NewFakeManager(tCtx.Logger()))
	if err != nil {
		t.Fatal(err)
	}
	return mgr, mgr.Start(tCtx, func() []*v1.Pod { return pods }, &sourcesReadyStub{}, mockPodStatusProvider{}, rt, initial)
}

func recoveryCheckpoint(t *testing.T) (state.State, string) {
	t.Helper()
	dir := t.TempDir()
	mgr, err := startCheckpointRecoveryManager(t, dir, nil, mockRuntimeService{}, containermap.NewContainerMap())
	if err != nil {
		t.Fatal(err)
	}
	return mgr.(*manager).state, dir
}

func checkpointBytes(t *testing.T, dir string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, memoryManagerStateFileName))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func runtimeRecoveryContainer(state runtimeapi.ContainerState, uid string) *runtimeapi.Container {
	return &runtimeapi.Container{Id: "runtime-id", State: state, Labels: map[string]string{kubelettypes.KubernetesPodUIDLabel: uid}}
}

func TestMemoryManagerRecoverPodOnlyCheckpointRuntime(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	unavailable := errors.New("runtime unavailable")
	for _, tc := range []struct {
		name             string
		containers       []*runtimeapi.Container
		listError        error
		wantError        string
		initialReference bool
	}{
		{name: "absent"},
		{name: "exited", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_EXITED, "orphan")}, initialReference: true},
		{name: "created", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_CREATED, "orphan")}, wantError: `[memorymanager] cannot recover pod "orphan": container "runtime-id" is in state CONTAINER_CREATED`},
		{name: "running despite empty initial map", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_RUNNING, "orphan")}, wantError: `[memorymanager] cannot recover pod "orphan": container "runtime-id" is in state CONTAINER_RUNNING`},
		{name: "running with initial reference", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_RUNNING, "orphan")}, initialReference: true, wantError: `[memorymanager] cannot recover pod "orphan": container "runtime-id" is in state CONTAINER_RUNNING`},
		{name: "unknown", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_UNKNOWN, "orphan")}, wantError: `[memorymanager] cannot recover pod "orphan": container "runtime-id" is in state CONTAINER_UNKNOWN`},
		{name: "unrecognized state", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState(99), "orphan")}, wantError: `[memorymanager] cannot recover pod "orphan": container "runtime-id" is in state 99`},
		{name: "exited followed by running", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_EXITED, "orphan"), runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_RUNNING, "orphan")}, wantError: `[memorymanager] cannot recover pod "orphan": container "runtime-id" is in state CONTAINER_RUNNING`},
		{name: "query error", listError: unavailable, wantError: `[memorymanager] cannot check runtime containers for pod "orphan": runtime unavailable`},
		{name: "partial result with query error", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_EXITED, "orphan")}, listError: unavailable, wantError: `[memorymanager] cannot check runtime containers for pod "orphan": runtime unavailable`},
		{name: "mismatched pod label", containers: []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_EXITED, "other")}, wantError: `[memorymanager] cannot recover pod "orphan": runtime returned a container without the requested pod UID`},
		{name: "missing pod label", containers: []*runtimeapi.Container{{Id: "runtime-id", State: runtimeapi.ContainerState_CONTAINER_EXITED}}, wantError: `[memorymanager] cannot recover pod "orphan": runtime returned a container without the requested pod UID`},
		{name: "missing container ID", containers: []*runtimeapi.Container{{State: runtimeapi.ContainerState_CONTAINER_EXITED, Labels: map[string]string{kubelettypes.KubernetesPodUIDLabel: "orphan"}}}, wantError: `[memorymanager] cannot recover pod "orphan": runtime returned a container without an ID`},
		{name: "nil container", containers: []*runtimeapi.Container{nil}, wantError: `[memorymanager] cannot recover pod "orphan": runtime returned a container without the requested pod UID`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, dir := recoveryCheckpoint(t)
			// Keep a committed reservation on the orphan's NUMA node so a
			// repeated removal cannot silently release another pod's memory.
			s.SetMemoryBlocks("background", "container", []state.Block{{NUMAAffinity: []int{1}, Type: v1.ResourceMemory, Size: 128 * mb}})
			machine := s.GetMachineState()
			machine[1].NumberOfAssignments++
			machine[1].MemoryMap[v1.ResourceMemory].Reserved += 128 * mb
			machine[1].MemoryMap[v1.ResourceMemory].Free -= 128 * mb
			s.SetMachineState(machine)
			s.SetPodMemoryBlocks("orphan", []state.Block{{NUMAAffinity: []int{1}, Type: v1.ResourceMemory, Size: 128 * mb}})
			beforePods, beforeMachine, beforeContainers := s.GetPodMemoryAssignments(), s.GetMachineState(), s.GetMemoryAssignments()
			before := checkpointBytes(t, dir)
			rt := &checkpointRecoveryRuntime{list: func(_ context.Context, filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
				if filter == nil || filter.State != nil || filter.Id != "" || filter.PodSandboxId != "" {
					t.Fatalf("runtime query must include all container states: %v", filter)
				}
				want := map[string]string{kubelettypes.KubernetesPodUIDLabel: "orphan"}
				if diff := cmp.Diff(want, filter.LabelSelector); diff != "" {
					t.Errorf("unexpected runtime filter (-want +got): %s", diff)
				}
				return tc.containers, tc.listError
			}}
			initial := containermap.NewContainerMap()
			if tc.initialReference {
				initial.Add("orphan", "container", "runtime-id")
			}
			mgr, err := startCheckpointRecoveryManager(t, dir, nil, rt, initial)
			if tc.wantError != "" {
				if err == nil || err.Error() != tc.wantError {
					t.Fatalf("expected %q, got %v", tc.wantError, err)
				}
				if tc.listError != nil && !errors.Is(err, tc.listError) {
					t.Errorf("runtime error was not preserved: %v", err)
				}
				if !bytes.Equal(before, checkpointBytes(t, dir)) {
					t.Error("failed recovery rewrote checkpoint")
				}
				if diff := cmp.Diff(beforePods, mgr.State().GetPodMemoryAssignments()); diff != "" {
					t.Errorf("failed recovery changed pod state: %s", diff)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if len(mgr.State().GetPodMemoryAssignments()) != 0 {
					t.Error("orphan was not removed")
				}
				persisted, err := state.NewCheckpointState(ktesting.Init(t).Logger(), dir, memoryManagerStateFileName, string(PolicyTypeStatic))
				if err != nil {
					t.Fatal(err)
				}
				if len(persisted.GetPodMemoryAssignments()) != 0 {
					t.Error("cleanup was not persisted")
				}
				if tc.initialReference {
					if err := mgr.RemoveContainer(ktesting.Init(t).Logger(), "runtime-id"); err != nil {
						t.Fatal(err)
					}
				}
				if _, err := startCheckpointRecoveryManager(t, dir, nil, rt, initial); err != nil {
					t.Fatalf("second restart failed: %v", err)
				}
			}
			if rt.calls != 1 {
				t.Errorf("expected one runtime query, got %d", rt.calls)
			}
			if diff := cmp.Diff(beforeMachine, mgr.State().GetMachineState()); diff != "" {
				t.Errorf("changed machine state: %s", diff)
			}
			if diff := cmp.Diff(beforeContainers, mgr.State().GetMemoryAssignments()); diff != "" {
				t.Errorf("changed container assignments: %s", diff)
			}
		})
	}
}

func TestMemoryManagerRejectMalformedPodOnlyCheckpoint(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	block := func(affinity []int, kind v1.ResourceName, size uint64) state.Block {
		return state.Block{NUMAAffinity: affinity, Type: kind, Size: size}
	}
	for _, tc := range []struct {
		name   string
		blocks []state.Block
		reason string
	}{
		{"unknown NUMA node", []state.Block{block([]int{99}, v1.ResourceMemory, mb)}, "NUMA node 99 does not exist"},
		{"unknown resource", []state.Block{block([]int{0}, "hugepages-512Mi", mb)}, `memory resource "hugepages-512Mi" does not exist on NUMA node 0`},
		{"empty affinity", []state.Block{block(nil, v1.ResourceMemory, mb)}, "empty NUMA affinity"},
		{"capacity exceeded", []state.Block{block([]int{0}, v1.ResourceMemory, 20*gb)}, `memory block exceeds allocatable "memory" in its NUMA affinity`},
		{"empty blocks", nil, "no memory blocks"},
		{"zero size", []state.Block{block([]int{0}, v1.ResourceMemory, 0)}, "zero-sized memory block"},
		{"duplicate node", []state.Block{block([]int{0, 0}, v1.ResourceMemory, mb)}, "NUMA affinity is not strictly increasing"},
		{"unsorted affinity", []state.Block{block([]int{1, 0}, v1.ResourceMemory, mb)}, "NUMA affinity is not strictly increasing"},
		{"duplicate resource", []state.Block{block([]int{0}, v1.ResourceMemory, mb), block([]int{0}, v1.ResourceMemory, mb)}, `duplicate memory resource "memory"`},
		{"different affinities", []state.Block{block([]int{0}, v1.ResourceMemory, mb), block([]int{1}, hugepages1Gi, gb)}, "memory blocks use different NUMA affinities"},
		{"missing memory", []state.Block{block([]int{0}, hugepages1Gi, gb)}, "no memory resource"},
		{"unaligned hugepages", []state.Block{block([]int{0}, v1.ResourceMemory, mb), block([]int{0}, hugepages1Gi, 1)}, `memory block is not aligned to "hugepages-1Gi"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, dir := recoveryCheckpoint(t)
			s.SetPodMemoryAssignments(state.PodMemoryAssignments{
				"a-valid":   {MemoryBlocks: []state.Block{block([]int{0}, v1.ResourceMemory, mb)}},
				"z-invalid": {MemoryBlocks: tc.blocks},
			})
			before := checkpointBytes(t, dir)
			rt := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
				t.Error("invalid state reached runtime lookup")
				return nil, nil
			}}
			mgr, err := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap())
			want := fmt.Sprintf("[memorymanager] invalid pod memory assignment for pod %q: %s", "z-invalid", tc.reason)
			if err == nil || err.Error() != want {
				t.Fatalf("expected %q, got %v", want, err)
			}
			if rt.calls != 0 {
				t.Errorf("unexpected runtime queries: %d", rt.calls)
			}
			if !bytes.Equal(before, checkpointBytes(t, dir)) {
				t.Error("invalid checkpoint was rewritten")
			}
			if diff := cmp.Diff(s.GetPodMemoryAssignments(), mgr.State().GetPodMemoryAssignments()); diff != "" {
				t.Errorf("invalid state was changed: %s", diff)
			}
		})
	}
}

func TestMemoryManagerRecoverPodOnlyCheckpointAllOrNothing(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	for _, queryFails := range []bool{false, true} {
		t.Run(fmt.Sprintf("query-failure=%t", queryFails), func(t *testing.T) {
			s, dir := recoveryCheckpoint(t)
			for _, uid := range []string{"a-valid", "z-blocked"} {
				s.SetPodMemoryBlocks(uid, []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: mb}})
			}
			before := checkpointBytes(t, dir)
			unavailable := errors.New("runtime unavailable")
			rt := &checkpointRecoveryRuntime{list: func(_ context.Context, f *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
				if f.LabelSelector[kubelettypes.KubernetesPodUIDLabel] == "z-blocked" {
					if queryFails {
						return nil, unavailable
					}
					return []*runtimeapi.Container{runtimeRecoveryContainer(runtimeapi.ContainerState_CONTAINER_RUNNING, "z-blocked")}, nil
				}
				return nil, nil
			}}
			mgr, err := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap())
			want := `[memorymanager] cannot recover pod "z-blocked": container "runtime-id" is in state CONTAINER_RUNNING`
			if queryFails {
				want = `[memorymanager] cannot check runtime containers for pod "z-blocked": runtime unavailable`
			}
			if err == nil || err.Error() != want {
				t.Fatalf("expected %q, got %v", want, err)
			}
			if queryFails && !errors.Is(err, unavailable) {
				t.Errorf("lost runtime error: %v", err)
			}
			if rt.calls != 2 {
				t.Errorf("expected both candidates to be checked, got %d queries", rt.calls)
			}
			if !bytes.Equal(before, checkpointBytes(t, dir)) {
				t.Error("earlier candidate was removed before all runtime checks completed")
			}
			if diff := cmp.Diff(s.GetPodMemoryAssignments(), mgr.State().GetPodMemoryAssignments()); diff != "" {
				t.Errorf("partially changed pod assignments: %s", diff)
			}
		})
	}
}

func TestMemoryManagerCompletePodCheckpointSkipsRuntime(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	dir := t.TempDir()
	pod := getPodWithContainersAndPodLevelResources("complete", "128Mi", "128Mi", nil, []containerSpec{{name: "container", memRequest: "128Mi", memLimit: "128Mi"}})
	rt := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
		t.Error("normal lifecycle must not query runtime")
		return nil, errors.New("runtime unavailable")
	}}
	mgr, err := startCheckpointRecoveryManager(t, dir, []*v1.Pod{pod}, rt, containermap.NewContainerMap())
	if err != nil {
		t.Fatal(err)
	}
	baseline := mgr.State().GetMachineState()
	logger := ktesting.Init(t).Logger()
	if err := mgr.AllocatePod(logger, pod, lifecycle.AddOperation); err != nil {
		t.Fatal(err)
	}
	before := checkpointBytes(t, dir)
	mgr2, err := startCheckpointRecoveryManager(t, dir, []*v1.Pod{pod}, rt, containermap.NewContainerMap())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, checkpointBytes(t, dir)) {
		t.Error("complete checkpoint changed during restore")
	}
	if err := mgr2.AllocatePod(logger, pod, lifecycle.AddOperation); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, checkpointBytes(t, dir)) {
		t.Error("complete allocation was not idempotent")
	}
	mgr2.AddContainer(logger, pod, &pod.Spec.Containers[0], "runtime-id")
	if err := mgr2.RemoveContainer(logger, "runtime-id"); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(baseline, mgr2.State().GetMachineState()); diff != "" {
		t.Errorf("release did not restore machine state: %s", diff)
	}
	if len(mgr2.State().GetPodMemoryAssignments()) != 0 || len(mgr2.State().GetMemoryAssignments()) != 0 {
		t.Error("release retained assignments")
	}
	if _, err := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap()); err != nil {
		t.Fatal(err)
	}
	if rt.calls != 0 {
		t.Errorf("normal lifecycle queried runtime %d times", rt.calls)
	}
}

func TestMemoryManagerPodCheckpointRecoveryFeatureDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	s, dir := recoveryCheckpoint(t)
	s.SetPodMemoryBlocks("orphan", []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: mb}})
	before := checkpointBytes(t, dir)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, false)
	rt := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
		t.Error("disabled feature must not query runtime")
		return nil, errors.New("runtime unavailable")
	}}
	if _, err := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap()); err != nil {
		t.Fatal(err)
	}
	if rt.calls != 0 || !bytes.Equal(before, checkpointBytes(t, dir)) {
		t.Error("disabled feature changed recovery behavior")
	}
}

func TestMemoryManagerPodCheckpointRecoveryUnsupportedPolicy(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	s := state.NewMemoryState(ktesting.Init(t).Logger())
	s.SetPodMemoryBlocks("orphan", []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: mb}})
	mgr := &manager{policy: &bestEffortPolicy{}, state: s}
	want := `[memorymanager] pod "orphan" has a pod memory assignment but no container memory assignments`
	if err := mgr.recoverPodMemoryAssignments(ktesting.Init(t)); err == nil || err.Error() != want {
		t.Fatalf("expected %q, got %v", want, err)
	}
	if len(s.GetPodMemoryAssignments()) != 1 {
		t.Error("unsupported policy removed pod assignment")
	}
}

func TestMemoryManagerMultiNUMAHugepagesCheckpointRecovery(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	for _, withOrphan := range []bool{false, true} {
		t.Run(fmt.Sprintf("orphan=%t", withOrphan), func(t *testing.T) {
			s, dir := recoveryCheckpoint(t)
			machine := s.GetMachineState()
			// A valid distribution that differs from replay order within the group.
			for id, reserved := range map[int]uint64{0: 3 * gb, 1: 2 * gb} {
				machine[id].NumberOfAssignments = 3
				machine[id].Cells = []int{0, 1}
				machine[id].MemoryMap[v1.ResourceMemory].Reserved = reserved
				machine[id].MemoryMap[v1.ResourceMemory].Free -= reserved
			}
			machine[0].MemoryMap[hugepages1Gi].Reserved = gb
			machine[0].MemoryMap[hugepages1Gi].Free -= gb
			// One pod-level memory block and one hugepage block count once each;
			// the container-level pod accounts for the third assignment.
			s.SetPodMemoryBlocks("complete", []state.Block{
				{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 4 * gb},
				{NUMAAffinity: []int{0, 1}, Type: hugepages1Gi, Size: gb},
			})
			s.SetMemoryBlocks("complete", "sidecar", []state.Block{{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: gb}})
			s.SetMemoryBlocks("complete", "app", []state.Block{{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 2 * gb}})
			s.SetMemoryBlocks("container-pod", "container", []state.Block{{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: gb}})
			s.SetMachineState(machine)
			wantPods, wantContainers := s.GetPodMemoryAssignments(), s.GetMemoryAssignments()
			if withOrphan {
				s.SetPodMemoryBlocks("orphan", []state.Block{{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 128 * mb}, {NUMAAffinity: []int{0, 1}, Type: hugepages1Gi, Size: gb}})
			}
			before := checkpointBytes(t, dir)
			rt := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) { return nil, nil }}
			mgr, err := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap())
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(wantPods, mgr.State().GetPodMemoryAssignments()); diff != "" {
				t.Errorf("changed completed pods: %s", diff)
			}
			if diff := cmp.Diff(wantContainers, mgr.State().GetMemoryAssignments()); diff != "" {
				t.Errorf("changed completed containers: %s", diff)
			}
			if diff := cmp.Diff(machine, mgr.State().GetMachineState()); diff != "" {
				t.Errorf("changed group distribution: %s", diff)
			}
			if !withOrphan && !bytes.Equal(before, checkpointBytes(t, dir)) {
				t.Error("normal restore rewrote checkpoint")
			}
			wantCalls := 0
			if withOrphan {
				wantCalls = 1
			}
			if rt.calls != wantCalls {
				t.Errorf("runtime queries: got %d, want %d", rt.calls, wantCalls)
			}
		})
	}
}

func TestMemoryManagerPodRecoveryRejectsInvalidNodeAccounting(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	s, dir := recoveryCheckpoint(t)
	machine := s.GetMachineState()
	for id, reserved := range map[int]uint64{0: gb, 1: gb} {
		machine[id].NumberOfAssignments = 1
		machine[id].Cells = []int{0, 1}
		machine[id].MemoryMap[v1.ResourceMemory].Reserved = reserved
		machine[id].MemoryMap[v1.ResourceMemory].Free -= reserved
	}
	// Keep group totals unchanged while breaking per-node conservation.
	machine[0].MemoryMap[v1.ResourceMemory].Free++
	machine[1].MemoryMap[v1.ResourceMemory].Free--
	s.SetMemoryBlocks("complete", "container", []state.Block{{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 2 * gb}})
	s.SetMachineState(machine)
	s.SetPodMemoryBlocks("orphan", []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: mb}})
	before := checkpointBytes(t, dir)
	rt := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
		t.Error("invalid accounting reached runtime lookup")
		return nil, nil
	}}
	_, err := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap())
	want := `[memorymanager] cannot recover pod memory assignments: invalid accounting for NUMA node 0 resource "memory"`
	if err == nil || err.Error() != want {
		t.Fatalf("expected %q, got %v", want, err)
	}
	if !bytes.Equal(before, checkpointBytes(t, dir)) {
		t.Error("invalid accounting was rewritten")
	}
}

func TestMemoryManagerPodCheckpointRecoveryWriteFailureRetry(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	s, dir := recoveryCheckpoint(t)
	s.SetPodMemoryBlocks("orphan", []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: mb}})
	before := checkpointBytes(t, dir)
	saved := dir + ".saved"
	rt := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
		// Fail only the cleanup write, after restoration and validation have
		// read the real checkpoint. ENOTDIR is deterministic even as root.
		if err := os.Rename(dir, saved); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dir, []byte("force ENOTDIR"), 0600); err != nil {
			t.Fatal(err)
		}
		return nil, nil
	}}
	mgr, startErr := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap())
	if err := os.Remove(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(saved, dir); err != nil {
		t.Fatal(err)
	}
	if startErr != nil {
		t.Fatalf("existing State API does not return cleanup write failures: %v", startErr)
	}
	if len(mgr.State().GetPodMemoryAssignments()) != 0 {
		t.Error("cache still contains orphan")
	}
	if !bytes.Equal(before, checkpointBytes(t, dir)) {
		t.Error("failed write changed durable checkpoint")
	}
	retryRuntime := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) { return nil, nil }}
	if _, err := startCheckpointRecoveryManager(t, dir, nil, retryRuntime, containermap.NewContainerMap()); err != nil {
		t.Fatal(err)
	}
	persisted, err := state.NewCheckpointState(ktesting.Init(t).Logger(), dir, memoryManagerStateFileName, string(PolicyTypeStatic))
	if err != nil {
		t.Fatal(err)
	}
	if len(persisted.GetPodMemoryAssignments()) != 0 {
		t.Error("retry did not persist cleanup")
	}
}

func TestMemoryManagerPodRecoveryRejectsEmptyUID(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	s, dir := recoveryCheckpoint(t)
	s.SetPodMemoryBlocks("", []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: mb}})
	before := checkpointBytes(t, dir)
	rt := &checkpointRecoveryRuntime{list: func(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
		t.Error("empty UID reached runtime lookup")
		return nil, nil
	}}
	_, err := startCheckpointRecoveryManager(t, dir, nil, rt, containermap.NewContainerMap())
	want := `[memorymanager] invalid pod memory assignment for pod "": empty pod UID`
	if err == nil || err.Error() != want {
		t.Fatalf("expected %q, got %v", want, err)
	}
	if rt.calls != 0 || !bytes.Equal(before, checkpointBytes(t, dir)) {
		t.Error("invalid UID changed state or reached runtime lookup")
	}
}
