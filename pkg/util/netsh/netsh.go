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

package netsh

import (
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

// Interface is an injectable interface for running netsh commands.  Implementations must be goroutine-safe.
type Interface interface {
	// EnsurePortProxyRule checks if the specified redirect exists, if not creates it
	EnsurePortProxyRule(args []string) (bool, error)
	// DeletePortProxyRule deletes the specified portproxy rule.  If the rule did not exist, return error.
	DeletePortProxyRule(args []string) error
	// EnsureIPAddress checks if the specified IP Address is added to vEthernet (HNSTransparent) interface, if not, add it.  If the address existed, return true.
	EnsureIPAddress(args []string, ip net.IP) (bool, error)
	// DeleteIPAddress checks if the specified IP address is present and, if so, deletes it.
	DeleteIPAddress(args []string) error
	// Restore runs `netsh exec` to restore portproxy or addresses using a file.
	// TODO Check if this is required, most likely not
	Restore(args []string) error

	// GetInterfaceToAddIP returns the interface name where Service IP needs to be added
	// IP Address needs to be added for netsh portproxy to redirect traffic
	// Reads Environment variable INTERFACE_TO_ADD_SERVICE_IP, if it is not defined then "vEthernet (HNSTransparent)" is returned
	GetInterfaceToAddIP() string
}

const (
	cmdNetsh string = "netsh"
)

// runner implements Interface in terms of exec("netsh").
type runner struct {
	mu   sync.Mutex
	exec utilexec.Interface
}

// New returns a new Interface which will exec netsh.
func New(exec utilexec.Interface) Interface {
	runner := &runner{
		exec: exec,
	}
	return runner
}

// EnsurePortProxyRule checks if the specified redirect exists, if not creates it.
func (runner *runner) EnsurePortProxyRule(args []string) (bool, error) {
	glog.V(4).Infof("running netsh interface portproxy add v4tov4 %v", args)
	out, err := runner.exec.Command(cmdNetsh, args...).CombinedOutput()

	if err == nil {
		return true, nil
	}
	if ee, ok := err.(utilexec.ExitError); ok {
		// netsh uses exit(0) to indicate a success of the operation,
		// as compared to a malformed commandline, for example.
		if ee.Exited() && ee.ExitStatus() != 0 {
			return false, nil
		}
	}
	return false, fmt.Errorf("error checking portproxy rule: %v: %s", err, out)

}

// DeletePortProxyRule deletes the specified portproxy rule.  If the rule did not exist, return error.
func (runner *runner) DeletePortProxyRule(args []string) error {
	glog.V(4).Infof("running netsh interface portproxy delete v4tov4 %v", args)
	out, err := runner.exec.Command(cmdNetsh, args...).CombinedOutput()

	if err == nil {
		return nil
	}
	if ee, ok := err.(utilexec.ExitError); ok {
		// netsh uses exit(0) to indicate a success of the operation,
		// as compared to a malformed commandline, for example.
		if ee.Exited() && ee.ExitStatus() == 0 {
			return nil
		}
	}
	return fmt.Errorf("error deleting portproxy rule: %v: %s", err, out)
}

// EnsureIPAddress checks if the specified IP Address is added to interface identified by Environment variable INTERFACE_TO_ADD_SERVICE_IP, if not, add it.  If the address existed, return true.
func (runner *runner) EnsureIPAddress(args []string, ip net.IP) (bool, error) {
	// Check if the ip address exists
	intName := runner.GetInterfaceToAddIP()
	argsShowAddress := []string{
		"interface", "ipv4", "show", "address",
		"name=" + intName,
	}

	ipToCheck := ip.String()

	exists, _ := checkIPExists(ipToCheck, argsShowAddress, runner)
	if exists == true {
		glog.V(4).Infof("not adding IP address %q as it already exists", ipToCheck)
		return true, nil
	}

	// IP Address is not already added, add it now
	glog.V(4).Infof("running netsh interface ipv4 add address %v", args)
	out, err := runner.exec.Command(cmdNetsh, args...).CombinedOutput()

	if err == nil {
		// Once the IP Address is added, it takes a bit to initialize and show up when querying for it
		// Query all the IP addresses and see if the one we added is present
		// PS: We are using netsh interface ipv4 show address here to query all the IP addresses, instead of
		// querying net.InterfaceAddrs() as it returns the IP address as soon as it is added even though it is uninitialized
		glog.V(3).Infof("Waiting until IP: %v is added to the network adapter", ipToCheck)
		for {
			if exists, _ := checkIPExists(ipToCheck, argsShowAddress, runner); exists {
				return true, nil
			}
			time.Sleep(500 * time.Millisecond)
		}
	}
	if ee, ok := err.(utilexec.ExitError); ok {
		// netsh uses exit(0) to indicate a success of the operation,
		// as compared to a malformed commandline, for example.
		if ee.Exited() && ee.ExitStatus() != 0 {
			return false, nil
		}
	}
	return false, fmt.Errorf("error adding ipv4 address: %v: %s", err, out)
}

// DeleteIPAddress checks if the specified IP address is present and, if so, deletes it.
func (runner *runner) DeleteIPAddress(args []string) error {
	glog.V(4).Infof("running netsh interface ipv4 delete address %v", args)
	out, err := runner.exec.Command(cmdNetsh, args...).CombinedOutput()

	if err == nil {
		return nil
	}
	if ee, ok := err.(utilexec.ExitError); ok {
		// netsh uses exit(0) to indicate a success of the operation,
		// as compared to a malformed commandline, for example.
		if ee.Exited() && ee.ExitStatus() == 0 {
			return nil
		}
	}
	return fmt.Errorf("error deleting ipv4 address: %v: %s", err, out)
}

// GetInterfaceToAddIP returns the interface name where Service IP needs to be added
// IP Address needs to be added for netsh portproxy to redirect traffic
// Reads Environment variable INTERFACE_TO_ADD_SERVICE_IP, if it is not defined then "vEthernet (HNS Internal NIC)" is returned
func (runner *runner) GetInterfaceToAddIP() string {
	if iface := os.Getenv("INTERFACE_TO_ADD_SERVICE_IP"); len(iface) > 0 {
		return iface
	}
	return "vEthernet (HNS Internal NIC)"
}

// Restore is part of Interface.
func (runner *runner) Restore(args []string) error {
	return nil
}

// checkIPExists checks if an IP address exists in 'netsh interface ipv4 show address' output
func checkIPExists(ipToCheck string, args []string, runner *runner) (bool, error) {
	ipAddress, err := runner.exec.Command(cmdNetsh, args...).CombinedOutput()
	if err != nil {
		return false, err
	}
	ipAddressString := string(ipAddress[:])
	glog.V(3).Infof("Searching for IP: %v in IP dump: %v", ipToCheck, ipAddressString)
	showAddressArray := strings.Split(ipAddressString, "\n")
	for _, showAddress := range showAddressArray {
		if strings.Contains(showAddress, "IP Address:") {
			ipFromNetsh := strings.TrimLeft(showAddress, "IP Address:")
			ipFromNetsh = strings.TrimSpace(ipFromNetsh)
			if ipFromNetsh == ipToCheck {
				return true, nil
			}
		}
	}

	return false, nil
}
