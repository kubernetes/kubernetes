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

const (
	// CgroupDriverSystemd holds the systemd driver type
	CgroupDriverSystemd = "systemd"
	// CgroupDriverCgroupfs holds the cgroupfs driver type
	CgroupDriverCgroupfs = "cgroupfs"
)

// TODO: add support for detecting the cgroup driver for CRI other than
// Docker. Currently only Docker driver detection is supported:
// Discussion:
//     https://github.com/kubernetes/kubeadm/issues/844

// GetCgroupDriverDocker runs 'docker info -f "{{.CgroupDriver}}"' to obtain the docker cgroup driver
func GetCgroupDriverDocker(execer utilsexec.Interface) (string, error) {
	driver, err := callDockerInfo(execer)
	if err != nil {
		return "", err
	}
	return strings.TrimSuffix(driver, "\n"), nil
}

func callDockerInfo(execer utilsexec.Interface) (string, error) {
	out, err := execer.Command("docker", "info", "-f", "{{.CgroupDriver}}").Output()
	if err != nil {
		return "", errors.Wrap(err, "cannot execute 'docker info -f {{.CgroupDriver}}'")
	}
	return string(out), nil
}
