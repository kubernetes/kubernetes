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
	// Absolute name of the desired container that Docker should be in.
	dockerContainerName string

	// The manager of the resource-only container Docker should be in.
	manager fs.Manager
}

var _ containerManager = &containerManagerImpl{}

// Takes the absolute name that the Docker daemon should be in.
// Empty container name disables moving the Docker daemon.
func newContainerManager(dockerDaemonContainer string) (containerManager, error) {
	return &containerManagerImpl{
		dockerContainerName: dockerDaemonContainer,
		manager: fs.Manager{
			Cgroups: &configs.Cgroup{
				Name:            dockerDaemonContainer,
				AllowAllDevices: true,
			},
		},
	}, nil
}

func (cm *containerManagerImpl) Start() error {
	if cm.dockerContainerName != "" {
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

		if cont != cm.dockerContainerName {
			err = cm.manager.Apply(pid)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to move PID %q (in %q) to %q", pid, cont, cm.dockerContainerName))
			}
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
