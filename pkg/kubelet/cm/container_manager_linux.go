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

package cm

import (
	"fmt"

	systemdUtil "github.com/coreos/go-systemd/util"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm/bootstrap"
	"k8s.io/kubernetes/pkg/kubelet/cm/bootstrap/cgroupfs"
	"k8s.io/kubernetes/pkg/kubelet/cm/bootstrap/systemd"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/sets"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
)

type containerManagerImpl struct {
	cadvisorInterface cadvisor.Interface
	mountUtil         mount.Interface
	bootstrapManager  bootstrap.BootstrapManager
}

var _ ContainerManager = &containerManagerImpl{}

// checks if the required cgroups subsystems are mounted.
// As of now, only 'cpu' and 'memory' are required.
func validateSystemRequirements(mountUtil mount.Interface) error {
	const (
		cgroupMountType = "cgroup"
		localErr        = "system validation failed"
	)
	mountPoints, err := mountUtil.List()
	if err != nil {
		return fmt.Errorf("%s - %v", localErr, err)
	}
	expectedCgroups := sets.NewString("cpu", "cpuacct", "cpuset", "memory")
	for _, mountPoint := range mountPoints {
		if mountPoint.Type == cgroupMountType {
			for _, opt := range mountPoint.Opts {
				if expectedCgroups.Has(opt) {
					expectedCgroups.Delete(opt)
				}
			}
		}
	}

	if expectedCgroups.Len() > 0 {
		return fmt.Errorf("%s - Following Cgroup subsystem not mounted: %v", localErr, expectedCgroups.List())
	}
	return nil
}

// TODO(vmarmol): Add limits to the system containers.
// Takes the absolute name of the specified containers.
// Empty container name disables use of the specified container.
func NewContainerManager(mountUtil mount.Interface, cadvisorInterface cadvisor.Interface) (ContainerManager, error) {
	return &containerManagerImpl{
		cadvisorInterface: cadvisorInterface,
		mountUtil:         mountUtil,
	}, nil
}

// TODO: plumb this up as a flag to Kubelet in a future PR
type KernelTunableBehavior string

const (
	KernelTunableWarn   KernelTunableBehavior = "warn"
	KernelTunableError  KernelTunableBehavior = "error"
	KernelTunableModify KernelTunableBehavior = "modify"
)

// setupKernelTunables validates kernel tunable flags are set as expected
// depending upon the specified option, it will either warn, error, or modify the kernel tunable flags
func setupKernelTunables(option KernelTunableBehavior) error {
	desiredState := map[string]int{
		utilsysctl.VmOvercommitMemory: utilsysctl.VmOvercommitMemoryAlways,
		utilsysctl.VmPanicOnOOM:       utilsysctl.VmPanicOnOOMInvokeOOMKiller,
		utilsysctl.KernelPanic:        utilsysctl.KernelPanicRebootTimeout,
		utilsysctl.KernelPanicOnOops:  utilsysctl.KernelPanicOnOopsAlways,
	}

	errList := []error{}
	for flag, expectedValue := range desiredState {
		val, err := utilsysctl.GetSysctl(flag)
		if err != nil {
			errList = append(errList, err)
			continue
		}
		if val == expectedValue {
			continue
		}

		switch option {
		case KernelTunableError:
			errList = append(errList, fmt.Errorf("Invalid kernel flag: %v, expected value: %v, actual value: %v", flag, expectedValue, val))
		case KernelTunableWarn:
			glog.V(2).Infof("Invalid kernel flag: %v, expected value: %v, actual value: %v", flag, expectedValue, val)
		case KernelTunableModify:
			glog.V(2).Infof("Updating kernel flag: %v, expected value: %v, actual value: %v", flag, expectedValue, val)
			err = utilsysctl.SetSysctl(flag, expectedValue)
			if err != nil {
				errList = append(errList, err)
			}
		}
	}
	return utilerrors.NewAggregate(errList)
}

func (cm *containerManagerImpl) Start(nodeConfig bootstrap.NodeConfig) error {
	if err := validateSystemRequirements(cm.mountUtil); err != nil {
		return err
	}
	// TODO: plumb kernel tunable options into container manager, right now, we modify by default
	if err := setupKernelTunables(KernelTunableModify); err != nil {
		return err
	}

	var err error
	if systemdUtil.IsRunningSystemd() {
		cm.bootstrapManager, err = systemd.NewBootstrapManager(cm.mountUtil, cm.cadvisorInterface)
	} else {
		cm.bootstrapManager, err = cgroupfs.NewBootstrapManager(cm.mountUtil, cm.cadvisorInterface)
	}
	if err != nil {
		return err
	}

	if err := cm.bootstrapManager.Start(nodeConfig); err != nil {
		return err
	}

	return nil
}

// SystemContainersLimit returns resources allocated to system containers in the machine.
func (cm *containerManagerImpl) SystemContainersLimit() api.ResourceList {
	result := api.ResourceList{}
	for _, container := range cm.bootstrapManager.SystemContainers() {
		result = add(result, container.Limits())
	}
	return result
}

// add returns the result of a + b for each named resource
// TODO: this is a utility cribbed from https://github.com/kubernetes/kubernetes/pull/20446
func add(a api.ResourceList, b api.ResourceList) api.ResourceList {
	result := api.ResourceList{}
	for key, value := range a {
		quantity := *value.Copy()
		if other, found := b[key]; found {
			quantity.Add(other)
		}
		result[key] = quantity
	}
	for key, value := range b {
		if _, found := result[key]; !found {
			quantity := *value.Copy()
			result[key] = quantity
		}
	}
	return result
}
