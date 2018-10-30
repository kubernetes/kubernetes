/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"strings"

	"github.com/pkg/errors"

	utilsexec "k8s.io/utils/exec"
)

// TODO: add support for detecting the cgroup driver for CRI other than
// Docker. Currently only Docker driver detection is supported:
// Discussion:
//     https://github.com/kubernetes/kubeadm/issues/844

// GetCgroupDriverDocker runs 'docker info' to obtain the docker cgroup driver
func GetCgroupDriverDocker(execer utilsexec.Interface) (string, error) {
	info, err := callDockerInfo(execer)
	if err != nil {
		return "", err
	}
	return getCgroupDriverFromDockerInfo(info)
}

func validateCgroupDriver(driver string) error {
	if driver != "cgroupfs" && driver != "systemd" {
		return errors.Errorf("unknown cgroup driver %q", driver)
	}
	return nil
}

// TODO: Docker 1.13 has a new way to obatain the cgroup driver:
//     docker info -f "{{.CgroupDriver}}
// If the minimum supported Docker version in K8s becomes 1.13, move to
// this syntax.
func callDockerInfo(execer utilsexec.Interface) (string, error) {
	out, err := execer.Command("docker", "info").Output()
	if err != nil {
		return "", errors.Wrap(err, "cannot execute 'docker info'")
	}
	return string(out), nil
}

func getCgroupDriverFromDockerInfo(info string) (string, error) {
	lineSeparator := ": "
	prefix := "Cgroup Driver"
	for _, line := range strings.Split(info, "\n") {
		if !strings.Contains(line, prefix+lineSeparator) {
			continue
		}
		lineSplit := strings.Split(line, lineSeparator)
		// At this point len(lineSplit) is always >= 2
		driver := lineSplit[1]
		if err := validateCgroupDriver(driver); err != nil {
			return "", err
		}
		return driver, nil
	}
	return "", errors.New("cgroup driver is not defined in 'docker info'")
}
