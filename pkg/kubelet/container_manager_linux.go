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
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/cgroups/fs"
	"github.com/docker/libcontainer/configs"
	"github.com/golang/glog"
)

type containerManagerImpl struct {
	// Whether to create and use the specified containers.
	useDockerContainer bool
	useSystemContainer bool

	// OOM score for the Docker container.
	dockerOomScoreAdj int

	// Managers for containers.
	dockerContainer fs.Manager
	systemContainer fs.Manager
	rootContainer   fs.Manager
}

var _ containerManager = &containerManagerImpl{}

// Takes the absolute name of the specified containers.
// Empty container name disables use of the specified container.
func newContainerManager(dockerDaemonContainer, systemContainer string) (containerManager, error) {
	if systemContainer == "/" {
		return nil, fmt.Errorf("system container cannot be root (\"/\")")
	}

	return &containerManagerImpl{
		useDockerContainer: dockerDaemonContainer != "",
		useSystemContainer: systemContainer != "",
		dockerOomScoreAdj:  -900,
		dockerContainer: fs.Manager{
			Cgroups: &configs.Cgroup{
				Name:            dockerDaemonContainer,
				AllowAllDevices: true,
			},
		},
		systemContainer: fs.Manager{
			Cgroups: &configs.Cgroup{
				Name:            systemContainer,
				AllowAllDevices: true,
			},
		},
		rootContainer: fs.Manager{
			Cgroups: &configs.Cgroup{
				Name: "/",
			},
		},
	}, nil
}

func (cm *containerManagerImpl) Start() error {
	if cm.useSystemContainer {
		err := cm.ensureSystemContainer()
		if err != nil {
			return err
		}
	}
	if cm.useDockerContainer {
		go util.Until(func() {
			err := cm.ensureDockerInContainer()
			if err != nil {
				glog.Warningf("[ContainerManager] Failed to ensure Docker is in a container: %v", err)
			}
		}, time.Minute, util.NeverStop)
	}
	return nil
}

// Ensures that the Docker daemon is in the desired container.
func (cm *containerManagerImpl) ensureDockerInContainer() error {
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
			errs = append(errs, fmt.Errorf("failed to find container of PID %q: %v", pid, err))
		}

		if cont != cm.dockerContainer.Cgroups.Name {
			err = cm.dockerContainer.Apply(pid)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to move PID %q (in %q) to %q", pid, cont, cm.dockerContainer.Cgroups.Name))
			}
		}

		// Also apply oom_score_adj to processes
		if err := util.ApplyOomScoreAdj(pid, cm.dockerOomScoreAdj); err != nil {
			errs = append(errs, fmt.Errorf("failed to apply oom score %q to PID %q", cm.dockerOomScoreAdj, pid))
		}
	}

	return errors.NewAggregate(errs)
}

// Gets the (CPU) container the specified pid is in.
func getContainer(pid int) (string, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}
	defer f.Close()

	return cgroups.ParseCgroupFile("cpu", f)
}

// Ensures the system container is created and all non-kernel processes without
// a container are moved to it.
func (cm *containerManagerImpl) ensureSystemContainer() error {
	// Move non-kernel PIDs to the system container.
	attemptsRemaining := 10
	var errs []error
	for attemptsRemaining >= 0 {
		// Only keep errors on latest attempt.
		errs = []error{}
		attemptsRemaining--

		allPids, err := cm.rootContainer.GetPids()
		if err != nil {
			errs = append(errs, fmt.Errorf("Failed to list PIDs for root: %v", err))
			continue
		}

		// Remove kernel pids
		pids := make([]int, 0, len(allPids))
		for _, pid := range allPids {
			if isKernelPid(pid) {
				continue
			}

			pids = append(pids, pid)
		}
		glog.Infof("Found %d PIDs in root, %d of them are kernel related", len(allPids), len(allPids)-len(pids))

		// Check if we moved all the non-kernel PIDs.
		if len(pids) == 0 {
			break
		}

		glog.Infof("Moving non-kernel threads: %v", pids)
		for _, pid := range pids {
			err := cm.systemContainer.Apply(pid)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to move PID %d into the system container %q: %v", pid, cm.systemContainer.Cgroups.Name, err))
				continue
			}
		}

	}
	if attemptsRemaining < 0 {
		errs = append(errs, fmt.Errorf("ran out of attempts to create system containers %q", cm.systemContainer.Cgroups.Name))
	}

	return errors.NewAggregate(errs)
}

// Determines whether the specified PID is a kernel PID.
func isKernelPid(pid int) bool {
	// Kernel threads have no associated executable.
	_, err := os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
	return err != nil
}
