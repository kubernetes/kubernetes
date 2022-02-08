//go:build linux
// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package cm

import (
	"errors"
	"io/ioutil"
	"os"
	"path"
	"testing"
	"testing/fstest"

	"github.com/golang/mock/gomock"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	cpumanagertesting "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/testing"
	devicemanagertesting "k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/testing"
	memorymanagertesting "k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/mount-utils"
)

func fakeContainerMgrMountInt() mount.Interface {
	return mount.NewFakeMounter(
		[]mount.MountPoint{
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuset"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpu"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuacct"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "memory"},
			},
		})
}

func TestCgroupMountValidationSuccess(t *testing.T) {
	f, err := validateSystemRequirements(fakeContainerMgrMountInt())
	assert.NoError(t, err)
	if cgroups.IsCgroup2UnifiedMode() {
		assert.True(t, f.cpuHardcapping, "cpu hardcapping is expected to be enabled")
	} else {
		assert.False(t, f.cpuHardcapping, "cpu hardcapping is expected to be disabled")
	}
}

func TestCgroupMountValidationMemoryMissing(t *testing.T) {
	if cgroups.IsCgroup2UnifiedMode() {
		t.Skip("skipping cgroup v1 test on a cgroup v2 system")
	}
	mountInt := mount.NewFakeMounter(
		[]mount.MountPoint{
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuset"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpu"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuacct"},
			},
		})
	_, err := validateSystemRequirements(mountInt)
	assert.Error(t, err)
}

func TestCgroupMountValidationMultipleSubsystem(t *testing.T) {
	if cgroups.IsCgroup2UnifiedMode() {
		t.Skip("skipping cgroup v1 test on a cgroup v2 system")
	}
	mountInt := mount.NewFakeMounter(
		[]mount.MountPoint{
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuset", "memory"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpu"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuacct"},
			},
		})
	_, err := validateSystemRequirements(mountInt)
	assert.NoError(t, err)
}

func TestGetCpuWeight(t *testing.T) {
	assert.Equal(t, uint64(0), getCpuWeight(nil))

	v := uint64(2)
	assert.Equal(t, uint64(1), getCpuWeight(&v))

	v = uint64(262144)
	assert.Equal(t, uint64(10000), getCpuWeight(&v))

	v = uint64(1000000000)
	assert.Equal(t, uint64(10000), getCpuWeight(&v))
}

func TestSoftRequirementsValidationSuccess(t *testing.T) {
	if cgroups.IsCgroup2UnifiedMode() {
		t.Skip("skipping cgroup v1 test on a cgroup v2 system")
	}
	req := require.New(t)
	tempDir, err := ioutil.TempDir("", "")
	req.NoError(err)
	defer os.RemoveAll(tempDir)
	req.NoError(ioutil.WriteFile(path.Join(tempDir, "cpu.cfs_period_us"), []byte("0"), os.ModePerm))
	req.NoError(ioutil.WriteFile(path.Join(tempDir, "cpu.cfs_quota_us"), []byte("0"), os.ModePerm))
	mountInt := mount.NewFakeMounter(
		[]mount.MountPoint{
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuset"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpu"},
				Path:   tempDir,
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuacct", "memory"},
			},
		})
	f, err := validateSystemRequirements(mountInt)
	assert.NoError(t, err)
	assert.True(t, f.cpuHardcapping, "cpu hardcapping is expected to be enabled")
}

func TestValidateSwapConfiguration(t *testing.T) {
	t.Parallel()

	tc := []struct {
		name             string
		swapFileContents string
		failSwapOn       bool
		expectedErr      string
	}{
		{
			name:       "swapfile_does_not_exist_fail_on",
			failSwapOn: true,
		},
		{
			name:       "swapfile_does_not_exist_fail_off",
			failSwapOn: false,
		},
		{
			name:       "swapfile_is_only_a_header_fail_off",
			failSwapOn: false,
			swapFileContents: `
Filename                                Type            Size            Used            Priority
			`,
		},
		{
			name:       "swapfile_is_only_a_header_fail_on",
			failSwapOn: true,
			swapFileContents: `
Filename                                Type            Size            Used            Priority
			`,
		},
		{
			name:       "swap_is_enabled_fail_off",
			failSwapOn: false,
			swapFileContents: `
Filename                                Type            Size            Used            Priority
/dev/dm-1                               partition       16760828        0               -2
			`,
		},
		{
			name:       "swap_is_enabled_fail_on",
			failSwapOn: true,
			swapFileContents: `
Filename                                Type            Size            Used            Priority
/dev/dm-1                               partition       16760828        0               -2
			`,
			expectedErr: "running with swap on is not supported, please disable swap! or set --fail-swap-on flag to false. /proc/swaps contained:",
		},
	}

	for _, c := range tc {
		t.Run(c.name, func(t *testing.T) {
			tfs := fstest.MapFS{}

			if c.swapFileContents != "" {
				tfs = fstest.MapFS{
					"swaps": {
						Data: []byte(c.swapFileContents),
					},
				}
			}

			result := validateSwapConfiguration(tfs, c.failSwapOn)
			if c.expectedErr == "" {
				require.NoError(t, result)
			} else {
				require.Error(t, result)
				require.Contains(t, result.Error(), c.expectedErr)
			}
		})
	}
}

type fakeRuntimeService struct {
	podSandboxesResponse []*runtimeapi.PodSandbox
	podSandboxesErr      error

	listContainersResponse []*runtimeapi.Container
	listContainersErr      error
}

func (f *fakeRuntimeService) ListPodSandbox(filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
	if filter != nil {
		return nil, errors.New("filters unsupported")
	}

	return f.podSandboxesResponse, f.podSandboxesErr
}

func (f *fakeRuntimeService) ListContainers(filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	if filter != nil {
		return nil, errors.New("filters unsupported")
	}

	return f.listContainersResponse, f.listContainersErr
}

func TestBuildContainerMapFromRuntime(t *testing.T) {
	t.Parallel()

	tc := []struct {
		name string

		sandboxes  []*runtimeapi.PodSandbox
		sandboxErr error

		containers   []*runtimeapi.Container
		containerErr error

		findableContainers []string
	}{
		{
			name:      "no_containers_to_be_found_when_the_sandbox_is_missing",
			sandboxes: []*runtimeapi.PodSandbox{},
			containers: []*runtimeapi.Container{
				{
					Id:           "54df6a69-2a1e-4e9b-a387-abc759232e4b",
					PodSandboxId: "aa04d9ab-e593-433c-9ed0-40ae6933ec90",
					Metadata: &runtimeapi.ContainerMetadata{
						Name: "container",
					},
				},
			},

			findableContainers: []string{},
		},
		{
			name:       "ignores_errors_from_the_runtimeservice_for_sandboxes",
			sandboxErr: errors.New("service unavailable"),
			containers: []*runtimeapi.Container{
				{
					Id:           "54df6a69-2a1e-4e9b-a387-abc759232e4b",
					PodSandboxId: "aa04d9ab-e593-433c-9ed0-40ae6933ec90",
					Metadata: &runtimeapi.ContainerMetadata{
						Name: "container",
					},
				},
			},

			findableContainers: []string{},
		},
		{
			name: "ignores_errors_from_the_runtimeservice_for_containers",
			sandboxes: []*runtimeapi.PodSandbox{
				{
					Id: "aa04d9ab-e593-433c-9ed0-40ae6933ec90",
					Metadata: &runtimeapi.PodSandboxMetadata{
						Uid: "aa04d9ab-e593-433c-9ed0-40ae6933ec90",
					},
				},
			},
			containerErr: errors.New("service unavailable"),

			findableContainers: []string{},
		},
		{
			name: "happy_case_when_all_containers_have_sandboxes",
			sandboxes: []*runtimeapi.PodSandbox{
				{
					Id: "aa04d9ab-e593-433c-9ed0-40ae6933ec90",
					Metadata: &runtimeapi.PodSandboxMetadata{
						Uid: "aa04d9ab-e593-433c-9ed0-40ae6933ec90",
					},
				},
			},
			containers: []*runtimeapi.Container{
				{
					Id:           "54df6a69-2a1e-4e9b-a387-abc759232e4b",
					PodSandboxId: "aa04d9ab-e593-433c-9ed0-40ae6933ec90",
					Metadata: &runtimeapi.ContainerMetadata{
						Name: "container",
					},
				},
			},

			findableContainers: []string{
				"54df6a69-2a1e-4e9b-a387-abc759232e4b",
			},
		},
	}

	for _, c := range tc {
		t.Run(c.name, func(t *testing.T) {
			rs := &fakeRuntimeService{
				podSandboxesResponse:   c.sandboxes,
				podSandboxesErr:        c.sandboxErr,
				listContainersResponse: c.containers,
				listContainersErr:      c.containerErr,
			}

			cm := buildContainerMapFromRuntime(rs)

			findableIDs := make(map[string]struct{}, len(c.findableContainers))

			// Validate that all expected containers can be found by id
			for _, ctrID := range c.findableContainers {
				findableIDs[ctrID] = struct{}{}
				_, _, err := cm.GetContainerRef(ctrID)
				require.NoError(t, err)
			}

			// Validate pods that should not be found cannot be.
			for _, ctr := range c.containers {
				if _, ok := findableIDs[ctr.Id]; ok {
					break
				}

				_, _, err := cm.GetContainerRef(ctr.Id)
				require.Error(t, err)
			}
		})
	}
}

func TestResourceAllocator_Admit(t *testing.T) {
	t.Parallel()

	tc := []struct {
		name string

		deviceManagerShouldSucceed bool
		cpuManagerShouldSucceed    bool
		memoryManagerShouldSucceed bool

		shouldAdmit bool
	}{
		{
			name:                       "when_all_managers_admit",
			deviceManagerShouldSucceed: true,
			cpuManagerShouldSucceed:    true,
			memoryManagerShouldSucceed: true,

			shouldAdmit: true,
		},
		{
			name:                       "when_device_manager_rejects",
			deviceManagerShouldSucceed: false,
			cpuManagerShouldSucceed:    true,
			memoryManagerShouldSucceed: true,

			shouldAdmit: false,
		},
		{
			name:                       "when_cpu_manager_rejects",
			deviceManagerShouldSucceed: true,
			cpuManagerShouldSucceed:    false,
			memoryManagerShouldSucceed: true,

			shouldAdmit: false,
		},
		{
			name:                       "when_memory_manager_rejects",
			deviceManagerShouldSucceed: true,
			cpuManagerShouldSucceed:    true,
			memoryManagerShouldSucceed: false,

			shouldAdmit: false,
		},
	}

	for _, c := range tc {
		t.Run(c.name, func(t *testing.T) {
			mockCtrl := gomock.NewController(t)
			cpuManager := cpumanagertesting.NewMockManager(mockCtrl)
			memoryManager := memorymanagertesting.NewMockManager(mockCtrl)
			deviceManager := devicemanagertesting.NewMockManager(mockCtrl)

			func() {
				// This test is regrettably tied to ordering of validation in the allocator.
				// We stop checking on first failure, so many of these won't be called if
				// an earlier manager fails. We setup expectations in a closure to simplify
				// representing that.
				if c.deviceManagerShouldSucceed {
					deviceManager.EXPECT().Allocate(gomock.Any(), gomock.Any()).Return(nil)
				} else {
					deviceManager.EXPECT().Allocate(gomock.Any(), gomock.Any()).Return(errors.New("unexpected error"))
					return
				}

				if c.cpuManagerShouldSucceed {
					cpuManager.EXPECT().Allocate(gomock.Any(), gomock.Any()).Return(nil)
				} else {
					cpuManager.EXPECT().Allocate(gomock.Any(), gomock.Any()).Return(errors.New("unexpected error"))
					return
				}

				if c.memoryManagerShouldSucceed {
					memoryManager.EXPECT().Allocate(gomock.Any(), gomock.Any()).Return(nil)
				} else {
					memoryManager.EXPECT().Allocate(gomock.Any(), gomock.Any()).Return(errors.New("unexpected error"))
					return
				}
			}()

			ra := &resourceAllocator{
				cpuManager:    cpuManager,
				memoryManager: memoryManager,
				deviceManager: deviceManager,
			}

			attrs := &lifecycle.PodAdmitAttributes{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "some-container",
							},
						},
					},
				},
			}

			admissionResult := ra.Admit(attrs)
			require.Equal(t, c.shouldAdmit, admissionResult.Admit)
		})
	}
}
