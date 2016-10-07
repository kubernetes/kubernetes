/*
Copyright 2016 The Kubernetes Authors.

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

package checks

import (
	"fmt"
	"io"
	"net"
	"os"
)

// PreFlightCheckers validate the state of the system to ensure kubeadm will be
// successful as often as possilble.
type PreFlightChecker interface {
	Check() (warnings []string, errors []string)
}

// ServiceChecker verifies that if this is a systemd system, the
// kubelet service exists, is enabled, and running.  We do not want to directly
// assume systemd is being used, so if this does not appear to be the case we
// just silently skip these checks and return no warnings or errors.
type ServiceChecker struct {
	initSystem InitSystem
	service    string
}

func (sc ServiceChecker) Check() (warnings []string, errors []string) {

	warnings = []string{}

	if !sc.initSystem.ServiceExists(sc.service) {
		warnings = append(warnings, fmt.Sprintf("%s service does not exist", sc.service))
		return warnings, nil
	}

	if !sc.initSystem.ServiceIsEnabled(sc.service) {
		errors = append(errors,
			fmt.Sprintf("%s service is not enabled, please run 'systemctl enable %s.service'",
				sc.service, sc.service))
	}

	if !sc.initSystem.ServiceIsActive(sc.service) {
		errors = append(errors,
			fmt.Sprintf("%s service is not active, please run 'systemctl start %s.service'",
				sc.service, sc.service))
	}

	return warnings, nil
}

// PortOpenChecker ensures the given port is available for use.
type PortOpenChecker struct {
	port int
}

func (poc PortOpenChecker) Check() (warnings []string, errors []string) {
	errors = []string{}
	// TODO: Get IP from KubeadmConfig
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", poc.port))
	if err != nil {
		errors = append(errors, fmt.Sprintf("Port %d is in use.", poc.port))
	}
	if ln != nil {
		ln.Close()
	}

	return nil, errors
}

// DirAvailableChecker checks if the given directory either does not exist, or
// is empty.
type DirAvailableChecker struct {
	path string
}

func (dac DirAvailableChecker) Check() (warnings []string, errors []string) {
	errors = []string{}
	// If it doesn't exist we are good:
	if _, err := os.Stat(dac.path); os.IsNotExist(err) {
		return nil, nil
	}

	f, err := os.Open(dac.path)
	if err != nil {
		errors = append(errors, fmt.Sprintf("Unable to check if %s is empty: %s", dac.path, err))
		return nil, errors
	}
	defer f.Close()

	_, err = f.Readdirnames(1) // Or f.Readdir(1)
	if err != io.EOF {
		errors = append(errors, fmt.Sprintf("%s is not empty", dac.path))
	}

	return nil, errors

}

func RunMasterChecks() {

	checks := []PreFlightChecker{}

	initSystem := getInitSystem()
	// Warn if we weren't able to detect a supported init system, and skip service checks:
	if initSystem == nil {
		fmt.Println("no kubeadm supported init system detected, skipping service checks")
	}
	checks = append(checks,
		ServiceChecker{
			service:    "kubelet",
			initSystem: getInitSystem(),
		},
		ServiceChecker{
			service:    "docker",
			initSystem: getInitSystem(),
		},
	)

	// TODO: Some of these ports should come from kubeadm config eventually:
	checks = append(checks,
		PortOpenChecker{443},
		PortOpenChecker{2379},
		PortOpenChecker{8080},
		PortOpenChecker{10250},
		PortOpenChecker{10251},
		PortOpenChecker{10252},
		DirAvailableChecker{"/etc/kubernetes"},
		DirAvailableChecker{"/var/lib/etcd"},
		DirAvailableChecker{"/var/lib/kubelet"},
	)

	foundErrors := false
	for _, check := range checks {
		warnings, errors := check.Check()
		for _, warnMsg := range warnings {
			fmt.Printf("WARNING: %s\n", warnMsg)
		}
		for _, errMsg := range errors {
			foundErrors = true
			fmt.Printf("ERROR: %s\n", errMsg)
		}
	}
	if foundErrors {
		os.Exit(1)
	}
}
