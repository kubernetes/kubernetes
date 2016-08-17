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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/kubernetes/pkg/util/mount"
)

type fakeMountInterface struct {
	mountPoints []mount.MountPoint
}

func (mi *fakeMountInterface) Mount(source string, target string, fstype string, options []string) error {
	return fmt.Errorf("unsupported")
}

func (mi *fakeMountInterface) Unmount(target string) error {
	return fmt.Errorf("unsupported")
}

func (mi *fakeMountInterface) List() ([]mount.MountPoint, error) {
	return mi.mountPoints, nil
}

func (mi *fakeMountInterface) IsLikelyNotMountPoint(file string) (bool, error) {
	return false, fmt.Errorf("unsupported")
}
func (mi *fakeMountInterface) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return "", nil
}

func (mi *fakeMountInterface) DeviceOpened(pathname string) (bool, error) {
	for _, mp := range mi.mountPoints {
		if mp.Device == pathname {
			return true, nil
		}
	}
	return false, nil
}

func (mi *fakeMountInterface) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

func fakeContainerMgrMountInt() mount.Interface {
	return &fakeMountInterface{
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
		},
	}
}

func TestCgroupMountValidationSuccess(t *testing.T) {
	f, err := validateSystemRequirements(fakeContainerMgrMountInt())
	assert.Nil(t, err)
	assert.False(t, f.cpuHardcapping, "cpu hardcapping is expected to be disabled")
}

func TestCgroupMountValidationMemoryMissing(t *testing.T) {
	mountInt := &fakeMountInterface{
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
		},
	}
	_, err := validateSystemRequirements(mountInt)
	assert.Error(t, err)
}

func TestCgroupMountValidationMultipleSubsytem(t *testing.T) {
	mountInt := &fakeMountInterface{
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
		},
	}
	_, err := validateSystemRequirements(mountInt)
	assert.Nil(t, err)
}

func TestSoftRequirementsValidationSuccess(t *testing.T) {
	req := require.New(t)
	tempDir, err := ioutil.TempDir("", "")
	req.NoError(err)
	req.NoError(ioutil.WriteFile(path.Join(tempDir, "cpu.cfs_period_us"), []byte("0"), os.ModePerm))
	req.NoError(ioutil.WriteFile(path.Join(tempDir, "cpu.cfs_quota_us"), []byte("0"), os.ModePerm))
	mountInt := &fakeMountInterface{
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
		},
	}
	f, err := validateSystemRequirements(mountInt)
	assert.NoError(t, err)
	assert.True(t, f.cpuHardcapping, "cpu hardcapping is expected to be enabled")
}
