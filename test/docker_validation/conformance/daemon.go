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

package docker_validation

import (
	"errors"
	"os/exec"
	"path"
	"strings"
)

var (
	ErrUnsupportedRuntime        = errors.New("Only 'docker' supported")
	ErrUnsupportedServiceManager = errors.New("Only 'init' or 'systemd' supported")
)

type ConformanceContainerD struct {
	Name string

	serviceType string
}

func NewConformanceContainerD(name string) (ccd ConformanceContainerD, err error) {
	if name != "docker" {
		err = ErrUnsupportedRuntime
		return
	}

	var out []byte
	out, err = exec.Command("readlink", "/proc/1/exe").CombinedOutput()
	if err != nil {
		return
	}

	ccd.Name = name
	ccd.serviceType = path.Base(strings.TrimSpace(string(out)))
	if ccd.serviceType != "init" && ccd.serviceType != "systemd" {
		err = ErrUnsupportedServiceManager
	}
	return
}

func (ccd *ConformanceContainerD) IsAlive() bool {
	if _, err := exec.Command("docker", "ps").CombinedOutput(); err != nil {
		return false
	}

	return true
}

func (ccd *ConformanceContainerD) Run(image string, args []string, sync bool) error {
	argsNew := make([]string, len(args)+2)
	argsNew[0] = "run"
	argsNew[1] = image
	for i, arg := range args {
		argsNew[i+2] = arg
	}

	cmd := exec.Command("docker", argsNew...)
	err := cmd.Start()
	if sync {
		err = cmd.Wait()
	}

	return err
}

func (ccd *ConformanceContainerD) Start() (err error) {
	return execService(ccd.serviceType, ccd.Name, "start")
}

func (ccd *ConformanceContainerD) Restart() error {
	return execService(ccd.serviceType, ccd.Name, "restart")
}

func (ccd *ConformanceContainerD) Stop() error {
	return execService(ccd.serviceType, ccd.Name, "stop")
}

func execService(serviceType string, serviceName string, serviceOper string) (err error) {
	switch serviceType {
	case "init":
		_, err = exec.Command("service", serviceName, serviceOper).CombinedOutput()
	case "systemd":
		_, err = exec.Command("systemctl", serviceOper, serviceName).CombinedOutput()
	default:
		err = ErrUnsupportedServiceManager
	}
	return
}
