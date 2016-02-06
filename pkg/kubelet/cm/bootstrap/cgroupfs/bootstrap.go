// +build linux

/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package cgroupfs

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm/bootstrap"
	"k8s.io/kubernetes/pkg/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/oom"
)

// ensure we implement the required interface
var _ bootstrap.BootstrapManager = &bootstrapManagerImpl{}

// bootstrapManagerImpl implements bootstrap.BootstrapManager for systemd environments
type bootstrapManagerImpl struct {
	cadvisorInterface cadvisor.Interface
	bootstrap.NodeConfig
	mountUtil        mount.Interface
	systemContainers []bootstrap.SystemContainer
}

// NewBootstrapManager creates a manager for cgroupfs systems
func NewBootstrapManager(mountUtil mount.Interface, cadvisorInterface cadvisor.Interface) (bootstrap.BootstrapManager, error) {
	return &bootstrapManagerImpl{
		cadvisorInterface: cadvisorInterface,
		NodeConfig:        bootstrap.NodeConfig{},
		mountUtil:         mountUtil,
	}, nil
}

const (
	// The percent of the machine memory capacity. The value is used to calculate
	// docker memory resource container's hardlimit to workaround docker memory
	// leakage issue. Please see kubernetes/issues/9881 for more detail.
	DockerMemoryLimitThresholdPercent = 70
	// The minimum memory limit allocated to docker container: 150Mi
	MinDockerMemoryLimit = 150 * 1024 * 1024
)

// A non-user container tracked by the Kubelet.
type systemContainer struct {
	// Absolute name of the container.
	name string

	// CPU limit in millicores.
	cpuMillicores int64

	// Function that ensures the state of the container.
	// m is the cgroup manager for the specified container.
	ensureStateFunc func(m *fs.Manager) error

	// Manager for the cgroups of the external container.
	manager *fs.Manager
}

// Name is the absolute name of the container
func (s *systemContainer) Name() string {
	return s.name
}

// Limits is the limits for the container
func (s *systemContainer) Limits() api.ResourceList {
	return api.ResourceList{
		api.ResourceCPU: *resource.NewMilliQuantity(
			s.cpuMillicores,
			resource.DecimalSI),
	}
}

func newSystemContainer(containerName string) *systemContainer {
	return &systemContainer{
		name:    containerName,
		manager: createManager(containerName),
	}
}

// Create a cgroup container manager.
func createManager(containerName string) *fs.Manager {
	return &fs.Manager{
		Cgroups: &configs.Cgroup{
			Parent: "/",
			Name:   containerName,
			Resources: &configs.Resources{
				AllowAllDevices: true,
			},
		},
	}
}

// Start performs initial cgroup bootstrapping of the node
// - Move kubelet into a container
// - Ensures that the Docker daemon is in a container.
// - Creates the system container where all non-containerized processes run.
func (bm *bootstrapManagerImpl) Start(nodeConfig bootstrap.NodeConfig) error {
	bm.NodeConfig = nodeConfig
	systemContainers := []*systemContainer{}

	glog.Info("Bootstrapping container manager for cgroupfs.")

	/// Move kubelet to a container, if required.
	if bm.KubeletContainerName != "" {
		err := util.RunInResourceContainer(bm.KubeletContainerName)
		if err != nil {
			glog.Warningf("Failed to move Kubelet to container %q: %v", bm.KubeletContainerName, err)
		}
		systemContainers = append(systemContainers, newSystemContainer(bm.KubeletContainerName))
		glog.Infof("Running in container %q", bm.KubeletContainerName)
	}

	// Setup the container that manages the docker daemon
	if bm.DockerDaemonContainerName != "" {
		cont := newSystemContainer(bm.DockerDaemonContainerName)
		info, err := bm.cadvisorInterface.MachineInfo()
		var capacity = api.ResourceList{}
		if err != nil {
		} else {
			capacity = cadvisor.CapacityFromMachineInfo(info)
		}
		memoryLimit := (int64(capacity.Memory().Value() * DockerMemoryLimitThresholdPercent / 100))
		if memoryLimit < MinDockerMemoryLimit {
			glog.Warningf("Memory limit %d for container %s is too small, reset it to %d", memoryLimit, bm.DockerDaemonContainerName, MinDockerMemoryLimit)
			memoryLimit = MinDockerMemoryLimit
		}
		glog.V(2).Infof("Configure resource-only container %s with memory limit: %d", bm.DockerDaemonContainerName, memoryLimit)
		dockerContainer := &fs.Manager{
			Cgroups: &configs.Cgroup{
				Parent: "/",
				Name:   bm.DockerDaemonContainerName,
				Resources: &configs.Resources{
					Memory:          memoryLimit,
					MemorySwap:      -1,
					AllowAllDevices: true,
				},
			},
		}
		cont.ensureStateFunc = func(manager *fs.Manager) error {
			return ensureDockerInContainer(bm.cadvisorInterface, -900, dockerContainer)
		}
		systemContainers = append(systemContainers, cont)
	}

	// Move non-kernel processes into the system container specified
	if bm.SystemContainerName != "" {
		if bm.SystemContainerName == "/" {
			return fmt.Errorf("system container cannot be root (\"/\")")
		}
		rootContainer := &fs.Manager{
			Cgroups: &configs.Cgroup{
				Parent: "/",
				Name:   "/",
			},
		}
		manager := createManager(bm.SystemContainerName)
		err := ensureSystemContainer(rootContainer, manager)
		if err != nil {
			return err
		}
		systemContainers = append(systemContainers, newSystemContainer(bm.SystemContainerName))
		glog.Info("Container manager using system-container: %v", bm.SystemContainerName)
	}

	// Don't run a background thread if there are no ensureStateFuncs.
	numEnsureStateFuncs := 0
	for _, cont := range systemContainers {
		if cont.ensureStateFunc != nil {
			numEnsureStateFuncs++
		}
	}
	if numEnsureStateFuncs == 0 {
		return nil
	}
	// Run ensure state functions every minute.
	go util.Until(func() {
		for _, cont := range systemContainers {
			if cont.ensureStateFunc != nil {
				if err := cont.ensureStateFunc(cont.manager); err != nil {
					glog.Warningf("[ContainerManager] Failed to ensure state of %q: %v", cont.name, err)
				}
			}
		}
	}, time.Minute, util.NeverStop)

	// setup the external containers list
	containers := []bootstrap.SystemContainer{}
	for _, container := range systemContainers {
		containers = append(containers, container)
	}
	bm.systemContainers = containers

	return nil
}

// SystemContainers is the list of non-user containers managed during bootstrapping.
func (bm *bootstrapManagerImpl) SystemContainers() []bootstrap.SystemContainer {
	return bm.systemContainers
}

// Ensures that the Docker daemon is in the desired container.
func ensureDockerInContainer(cadvisor cadvisor.Interface, oomScoreAdj int, manager *fs.Manager) error {
	// What container is Docker in?
	out, err := exec.Command("pidof", "docker").Output()
	if err != nil {
		return fmt.Errorf("failed to find pid of Docker container: %v", err)
	}

	// The output of pidof is a list of pids.
	// Docker may be forking and thus there would be more than one result.
	pids := []int{}
	for _, pidStr := range strings.Split(strings.TrimSpace(string(out)), " ") {
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			continue
		}
		pids = append(pids, pid)
	}

	// Move if the pid is not already in the desired container.
	errs := []error{}
	for _, pid := range pids {
		cont, err := getContainer(pid)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to find container of PID %d: %v", pid, err))
		}

		if cont != manager.Cgroups.Name {
			err = manager.Apply(pid)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to move PID %d (in %q) to %q", pid, cont, manager.Cgroups.Name))
			}
		}

		// Also apply oom-score-adj to processes
		oomAdjuster := oom.NewOOMAdjuster()
		if err := oomAdjuster.ApplyOOMScoreAdj(pid, oomScoreAdj); err != nil {
			errs = append(errs, fmt.Errorf("failed to apply oom score %d to PID %d", oomScoreAdj, pid))
		}
	}

	return utilerrors.NewAggregate(errs)
}

// Gets the (CPU) container the specified pid is in.
func getContainer(pid int) (string, error) {
	cgs, err := cgroups.ParseCgroupFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}

	cg, ok := cgs["cpu"]
	if ok {
		return cg, nil
	}

	return "", cgroups.NewNotFoundError("cpu")
}

// Ensures the system container is created and all non-kernel threads and process 1
// without a container are moved to it.
//
// The reason of leaving kernel threads at root cgroup is that we don't want to tie the
// execution of these threads with to-be defined /system quota and create priority inversions.
//
// The reason of leaving process 1 at root cgroup is that libcontainer hardcoded on
// the base cgroup path based on process 1. Please see:
// https://github.com/kubernetes/kubernetes/issues/12789#issuecomment-132384126
// for detail explanation.
func ensureSystemContainer(rootContainer *fs.Manager, manager *fs.Manager) error {
	// Move non-kernel PIDs to the system container.
	attemptsRemaining := 10
	var errs []error
	for attemptsRemaining >= 0 {
		// Only keep errors on latest attempt.
		errs = []error{}
		attemptsRemaining--

		allPids, err := rootContainer.GetPids()
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to list PIDs for root: %v", err))
			continue
		}

		// Remove kernel pids and other protected PIDs (pid 1, PIDs already in system & kubelet containers)
		pids := make([]int, 0, len(allPids))
		for _, pid := range allPids {
			if isKernelPid(pid) {
				continue
			}

			pids = append(pids, pid)
		}
		glog.Infof("Found %d PIDs in root, %d of them are not to be moved", len(allPids), len(allPids)-len(pids))

		// Check if we have moved all the non-kernel PIDs.
		if len(pids) == 0 {
			break
		}

		glog.Infof("Moving non-kernel processes: %v", pids)
		for _, pid := range pids {
			err := manager.Apply(pid)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to move PID %d into the system container %q: %v", pid, manager.Cgroups.Name, err))
			}
		}

	}
	if attemptsRemaining < 0 {
		errs = append(errs, fmt.Errorf("ran out of attempts to create system containers %q", manager.Cgroups.Name))
	}

	return utilerrors.NewAggregate(errs)
}

// Determines whether the specified PID is a kernel PID.
func isKernelPid(pid int) bool {
	// Kernel threads have no associated executable.
	_, err := os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
	return err != nil
}
