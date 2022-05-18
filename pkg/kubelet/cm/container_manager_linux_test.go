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

	gomock "github.com/golang/mock/gomock"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubefeatures "k8s.io/kubernetes/pkg/features"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"

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

func TestGetCapacity(t *testing.T) {
	ephemeralStorageFromCapacity := int64(2000)
	ephemeralStorageFromCadvisor := int64(8000)
	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()
	mockCtrlError := gomock.NewController(t)
	defer mockCtrlError.Finish()

	mockCadvisor := cadvisortest.NewMockInterface(mockCtrl)
	rootfs := cadvisorapiv2.FsInfo{
		Capacity: 8000,
	}
	mockCadvisor.EXPECT().RootFsInfo().Return(rootfs, nil)
	mockCadvisorError := cadvisortest.NewMockInterface(mockCtrlError)
	mockCadvisorError.EXPECT().RootFsInfo().Return(cadvisorapiv2.FsInfo{}, errors.New("Unable to get rootfs data from cAdvisor interface"))
	cases := []struct {
		name                                string
		cm                                  *containerManagerImpl
		expectedResourceQuantity            *resource.Quantity
		expectedNoEphemeralStorage          bool
		enableLocalStorageCapacityIsolation bool
	}{
		{
			name: "capacity property has ephemeral-storage",
			cm: &containerManagerImpl{
				cadvisorInterface: mockCadvisor,
				capacity: v1.ResourceList{
					v1.ResourceEphemeralStorage: *resource.NewQuantity(ephemeralStorageFromCapacity, resource.BinarySI),
				},
			},
			expectedResourceQuantity:            resource.NewQuantity(ephemeralStorageFromCapacity, resource.BinarySI),
			expectedNoEphemeralStorage:          false,
			enableLocalStorageCapacityIsolation: true,
		},
		{
			name: "capacity property does not have ephemeral-storage",
			cm: &containerManagerImpl{
				cadvisorInterface: mockCadvisor,
				capacity:          v1.ResourceList{},
			},
			expectedResourceQuantity:            resource.NewQuantity(ephemeralStorageFromCadvisor, resource.BinarySI),
			expectedNoEphemeralStorage:          false,
			enableLocalStorageCapacityIsolation: true,
		},
		{
			name: "capacity property does not have ephemeral-storage, error from rootfs",
			cm: &containerManagerImpl{
				cadvisorInterface: mockCadvisorError,
				capacity:          v1.ResourceList{},
			},
			expectedNoEphemeralStorage:          true,
			enableLocalStorageCapacityIsolation: true,
		},
		{
			name: "capacity property does not have ephemeral-storage, cadvisor interface is nil",
			cm: &containerManagerImpl{
				cadvisorInterface: nil,
				capacity:          v1.ResourceList{},
			},
			expectedNoEphemeralStorage:          true,
			enableLocalStorageCapacityIsolation: true,
		},
		{
			name: "LocalStorageCapacityIsolation feature flag is disabled",
			cm: &containerManagerImpl{
				cadvisorInterface: mockCadvisor,
				capacity: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("4"),
					v1.ResourceMemory: resource.MustParse("16G"),
				},
			},
			expectedNoEphemeralStorage:          true,
			enableLocalStorageCapacityIsolation: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.LocalStorageCapacityIsolation, c.enableLocalStorageCapacityIsolation)()
			ret := c.cm.GetCapacity()
			if v, exists := ret[v1.ResourceEphemeralStorage]; !exists {
				if !c.expectedNoEphemeralStorage {
					t.Errorf("did not get any ephemeral storage data")
				}
			} else {
				if v.Value() != c.expectedResourceQuantity.Value() {
					t.Errorf("got unexpected %s value, expected %d, got %d", v1.ResourceEphemeralStorage, c.expectedResourceQuantity.Value(), v.Value())
				}
			}
		})
	}
}
