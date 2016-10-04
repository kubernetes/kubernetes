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

package preflight

import (
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"

	"k8s.io/kubernetes/pkg/util/initsystem"
)

type PreFlightError struct {
	Msg   string
	Count int
}

func (e *PreFlightError) Error() string {
	return fmt.Sprintf("preflight check error\n  count: %d \n  msg: %s", e.Count, e.Msg)
}

// PreFlightCheck validates the state of the system to ensure kubeadm will be
// successful as often as possilble.
type PreFlightCheck interface {
	Check() (warnings, errors []error)
}

// ServiceCheck verifies that the given service is enabled and active. If we do not
// detect a supported init system however, all checks are skipped and a warning is
// returned.
type ServiceCheck struct {
	service string
}

func (sc ServiceCheck) Check() (warnings, errors []error) {

	initSystem := initsystem.GetInitSystem()
	if initSystem == nil {
		return []error{fmt.Errorf("no supported init system detected, skipping service checks for %s", sc.service)}, nil
	}

	warnings = []error{}

	if !initSystem.ServiceExists(sc.service) {
		warnings = append(warnings, fmt.Errorf("%s service does not exist", sc.service))
		return warnings, nil
	}

	if !initSystem.ServiceIsEnabled(sc.service) {
		warnings = append(warnings,
			fmt.Errorf("%s service is not enabled, please run 'systemctl enable %s.service'",
				sc.service, sc.service))
	}

	if !initSystem.ServiceIsActive(sc.service) {
		errors = append(errors,
			fmt.Errorf("%s service is not active, please run 'systemctl start %s.service'",
				sc.service, sc.service))
	}

	return warnings, nil
}

// PortOpenCheck ensures the given port is available for use.
type PortOpenCheck struct {
	port int
}

func (poc PortOpenCheck) Check() (warnings, errors []error) {
	errors = []error{}
	// TODO: Get IP from KubeadmConfig
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", poc.port))
	if err != nil {
		errors = append(errors, fmt.Errorf("Port %d is in use", poc.port))
	}
	if ln != nil {
		ln.Close()
	}

	return nil, errors
}

// IsRootCheck verifies user is root
type IsRootCheck struct {
	root bool
}

func (irc IsRootCheck) Check() (warnings, errors []error) {
	errors = []error{}
	if os.Getuid() != 0 {
		errors = append(errors, fmt.Errorf("user is not running as root"))
	}

	return nil, errors
}

// DirAvailableCheck checks if the given directory either does not exist, or
// is empty.
type DirAvailableCheck struct {
	path string
}

func (dac DirAvailableCheck) Check() (warnings, errors []error) {
	errors = []error{}
	// If it doesn't exist we are good:
	if _, err := os.Stat(dac.path); os.IsNotExist(err) {
		return nil, nil
	}

	f, err := os.Open(dac.path)
	if err != nil {
		errors = append(errors, fmt.Errorf("unable to check if %s is empty: %s", dac.path, err))
		return nil, errors
	}
	defer f.Close()

	_, err = f.Readdirnames(1)
	if err != io.EOF {
		errors = append(errors, fmt.Errorf("%s is not empty", dac.path))
	}

	return nil, errors
}

// InPathChecks checks if the given executable is present in the path.
type InPathCheck struct {
	executable string
	mandatory  bool
}

func (ipc InPathCheck) Check() (warnings, errors []error) {
	_, err := exec.LookPath(ipc.executable)
	if err != nil {
		if ipc.mandatory {
			// Return as an error:
			return nil, []error{fmt.Errorf("%s not found in system path", ipc.executable)}
		}
		// Return as a warning:
		return []error{fmt.Errorf("%s not found in system path", ipc.executable)}, nil
	}
	return nil, nil
}

func RunInitMasterChecks() error {
	// TODO: Some of these ports should come from kubeadm config eventually:
	checks := []PreFlightCheck{
		IsRootCheck{root: true},
		ServiceCheck{service: "kubelet"},
		ServiceCheck{service: "docker"},
		PortOpenCheck{port: 443},
		PortOpenCheck{port: 2379},
		PortOpenCheck{port: 8080},
		PortOpenCheck{port: 10250},
		PortOpenCheck{port: 10251},
		PortOpenCheck{port: 10252},
		DirAvailableCheck{path: "/etc/kubernetes"},
		DirAvailableCheck{path: "/var/lib/etcd"},
		DirAvailableCheck{path: "/var/lib/kubelet"},
		InPathCheck{executable: "ebtables", mandatory: true},
		InPathCheck{executable: "ethtool", mandatory: true},
		InPathCheck{executable: "ip", mandatory: true},
		InPathCheck{executable: "iptables", mandatory: true},
		InPathCheck{executable: "mount", mandatory: true},
		InPathCheck{executable: "nsenter", mandatory: true},
		InPathCheck{executable: "socat", mandatory: true},
		InPathCheck{executable: "tc", mandatory: false},
		InPathCheck{executable: "touch", mandatory: false},
	}

	return runChecks(checks)
}

func RunJoinNodeChecks() error {
	// TODO: Some of these ports should come from kubeadm config eventually:
	checks := []PreFlightCheck{
		IsRootCheck{root: true},
		ServiceCheck{service: "docker"},
		ServiceCheck{service: "kubelet"},
		PortOpenCheck{port: 8080},
		PortOpenCheck{port: 10250},
		PortOpenCheck{port: 10251},
		PortOpenCheck{port: 10252},
		DirAvailableCheck{path: "/etc/kubernetes"},
		DirAvailableCheck{path: "/var/lib/kubelet"},
		InPathCheck{executable: "ebtables", mandatory: true},
		InPathCheck{executable: "ethtool", mandatory: true},
		InPathCheck{executable: "ip", mandatory: true},
		InPathCheck{executable: "iptables", mandatory: true},
		InPathCheck{executable: "mount", mandatory: true},
		InPathCheck{executable: "nsenter", mandatory: true},
		InPathCheck{executable: "socat", mandatory: true},
		InPathCheck{executable: "tc", mandatory: false},
		InPathCheck{executable: "touch", mandatory: false},
	}

	return runChecks(checks)
}

// runChecks runs each check, displays it's warnings/errors, and once all
// are processed will exit if any errors occurred.
func runChecks(checks []PreFlightCheck) error {
	found := []error{}
	for _, c := range checks {
		warnings, errors := c.Check()
		for _, w := range warnings {
			fmt.Printf("WARNING: %s\n", w)
		}
		for _, e := range errors {
			found = append(found, e)
		}
	}
	if len(found) > 0 {
		errors := "\n"
		for _, i := range found {
			errors += "\t" + i.Error() + "\n"
		}
		return &PreFlightError{
			Msg:   errors,
			Count: len(found),
		}
	}
	return nil
}
