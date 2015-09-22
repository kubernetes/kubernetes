// +build linux

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

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
	assert.Nil(t, validateSystemRequirements(fakeContainerMgrMountInt()))
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
	assert.Error(t, validateSystemRequirements(mountInt))
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
	assert.Nil(t, validateSystemRequirements(mountInt))
}
