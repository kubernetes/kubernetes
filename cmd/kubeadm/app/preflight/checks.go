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
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/util/initsystem"
	"k8s.io/kubernetes/pkg/util/node"
)

type PreFlightError struct {
	Msg string
}

func (e *PreFlightError) Error() string {
	return fmt.Sprintf("preflight check errors:\n%s", e.Msg)
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
	Service string
}

func (sc ServiceCheck) Check() (warnings, errors []error) {
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		return []error{err}, nil
	}

	warnings = []error{}

	if !initSystem.ServiceExists(sc.Service) {
		warnings = append(warnings, fmt.Errorf("%s service does not exist", sc.Service))
		return warnings, nil
	}

	if !initSystem.ServiceIsEnabled(sc.Service) {
		warnings = append(warnings,
			fmt.Errorf("%s service is not enabled, please run 'systemctl enable %s.service'",
				sc.Service, sc.Service))
	}

	if !initSystem.ServiceIsActive(sc.Service) {
		errors = append(errors,
			fmt.Errorf("%s service is not active, please run 'systemctl start %s.service'",
				sc.Service, sc.Service))
	}

	return warnings, errors
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
	Path string
}

func (dac DirAvailableCheck) Check() (warnings, errors []error) {
	errors = []error{}
	// If it doesn't exist we are good:
	if _, err := os.Stat(dac.Path); os.IsNotExist(err) {
		return nil, nil
	}

	f, err := os.Open(dac.Path)
	if err != nil {
		errors = append(errors, fmt.Errorf("unable to check if %s is empty: %s", dac.Path, err))
		return nil, errors
	}
	defer f.Close()

	_, err = f.Readdirnames(1)
	if err != io.EOF {
		errors = append(errors, fmt.Errorf("%s is not empty", dac.Path))
	}

	return nil, errors
}

// FileAvailableCheck checks that the given file does not already exist.
type FileAvailableCheck struct {
	Path string
}

func (fac FileAvailableCheck) Check() (warnings, errors []error) {
	errors = []error{}
	if _, err := os.Stat(fac.Path); err == nil {
		errors = append(errors, fmt.Errorf("%s already exists", fac.Path))
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

// HostnameCheck checks if hostname match dns sub domain regex.
// If hostname doesn't match this regex, kubelet will not launch static pods like kube-apiserver/kube-controller-manager and so on.
type HostnameCheck struct{}

func (hc HostnameCheck) Check() (warnings, errors []error) {
	errors = []error{}
	hostname := node.GetHostname("")
	for _, msg := range validation.ValidateNodeName(hostname, false) {
		errors = append(errors, fmt.Errorf("hostname \"%s\" %s", hostname, msg))
	}
	return nil, errors
}

// HttpProxyCheck checks if https connection to specific host is going
// to be done directly or over proxy. If proxy detected, it will return warning.
type HttpProxyCheck struct {
	Proto string
	Host  string
	Port  int
}

func (hst HttpProxyCheck) Check() (warnings, errors []error) {

	url := fmt.Sprintf("%s://%s:%d", hst.Proto, hst.Host, hst.Port)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, []error{err}
	}

	proxy, err := http.DefaultTransport.(*http.Transport).Proxy(req)
	if err != nil {
		return nil, []error{err}
	}
	if proxy != nil {
		return []error{fmt.Errorf("Connection to %q uses proxy %q. If that is not intended, adjust your proxy settings", url, proxy)}, nil
	}
	return nil, nil
}

func RunInitMasterChecks(cfg *kubeadmapi.MasterConfiguration) error {
	// TODO: Some of these ports should come from kubeadm config eventually:
	checks := []PreFlightCheck{
		IsRootCheck{root: true},
		HostnameCheck{},
		ServiceCheck{Service: "kubelet"},
		ServiceCheck{Service: "docker"},
		PortOpenCheck{port: int(cfg.API.BindPort)},
		PortOpenCheck{port: 2379},
		PortOpenCheck{port: 8080},
		PortOpenCheck{port: int(cfg.Discovery.BindPort)},
		PortOpenCheck{port: 10250},
		PortOpenCheck{port: 10251},
		PortOpenCheck{port: 10252},
		HttpProxyCheck{Proto: "https", Host: cfg.API.AdvertiseAddresses[0], Port: int(cfg.API.BindPort)},
		DirAvailableCheck{Path: "/etc/kubernetes/manifests"},
		DirAvailableCheck{Path: "/etc/kubernetes/pki"},
		DirAvailableCheck{Path: "/var/lib/etcd"},
		DirAvailableCheck{Path: "/var/lib/kubelet"},
		FileAvailableCheck{Path: "/etc/kubernetes/admin.conf"},
		FileAvailableCheck{Path: "/etc/kubernetes/kubelet.conf"},
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

	return runChecks(checks, os.Stderr)
}

func RunJoinNodeChecks(cfg *kubeadmapi.NodeConfiguration) error {
	// TODO: Some of these ports should come from kubeadm config eventually:
	checks := []PreFlightCheck{
		IsRootCheck{root: true},
		HostnameCheck{},
		ServiceCheck{Service: "docker"},
		ServiceCheck{Service: "kubelet"},
		PortOpenCheck{port: 10250},
		HttpProxyCheck{Proto: "https", Host: cfg.MasterAddresses[0], Port: int(cfg.APIPort)},
		HttpProxyCheck{Proto: "http", Host: cfg.MasterAddresses[0], Port: int(cfg.DiscoveryPort)},
		DirAvailableCheck{Path: "/etc/kubernetes/manifests"},
		DirAvailableCheck{Path: "/var/lib/kubelet"},
		FileAvailableCheck{Path: "/etc/kubernetes/kubelet.conf"},
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

	return runChecks(checks, os.Stderr)
}

func RunResetCheck() error {
	checks := []PreFlightCheck{
		IsRootCheck{root: true},
	}

	return runChecks(checks, os.Stderr)
}

// runChecks runs each check, displays it's warnings/errors, and once all
// are processed will exit if any errors occurred.
func runChecks(checks []PreFlightCheck, ww io.Writer) error {
	found := []error{}
	for _, c := range checks {
		warnings, errs := c.Check()
		for _, w := range warnings {
			io.WriteString(ww, fmt.Sprintf("WARNING: %s\n", w))
		}
		for _, e := range errs {
			found = append(found, e)
		}
	}
	if len(found) > 0 {
		errs := ""
		for _, i := range found {
			errs += "\t" + i.Error() + "\n"
		}
		return errors.New(errs)
	}
	return nil
}
