/*
Copyright 2014 Google Inc. All rights reserved.

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

package health

import (
	"fmt"
	"os/exec"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

const defaultHealthyRegex = "^OK$"

type CommandRunner interface {
	RunInContainer(podFullName, containerName string, cmd []string) ([]byte, error)
}

type ExecHealthChecker struct {
	runner CommandRunner
}

func NewExecHealthChecker(runner CommandRunner) HealthChecker {
	return &ExecHealthChecker{runner}
}

func IsExitError(err error) bool {
	_, ok := err.(*exec.ExitError)
	return ok
}

func (e *ExecHealthChecker) HealthCheck(podFullName string, currentState api.PodState, container api.Container) (Status, error) {
	if container.LivenessProbe.Exec == nil {
		return Unknown, fmt.Errorf("Missing exec parameters")
	}
	data, err := e.runner.RunInContainer(podFullName, container.Name, container.LivenessProbe.Exec.Command)
	glog.V(1).Infof("container %s failed health check: %s", podFullName, string(data))
	if err != nil {
		if IsExitError(err) {
			return Unhealthy, nil
		}
		return Unknown, err
	}
	return Healthy, nil
}
