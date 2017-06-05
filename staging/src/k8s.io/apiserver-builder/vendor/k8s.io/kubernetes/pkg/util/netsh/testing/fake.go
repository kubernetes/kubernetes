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

package testing

import (
	"net"

	"k8s.io/kubernetes/pkg/util/netsh"
)

// no-op implementation of netsh Interface
type FakeNetsh struct {
}

func NewFake() *FakeNetsh {
	return &FakeNetsh{}
}

func (*FakeNetsh) EnsurePortProxyRule(args []string) (bool, error) {
	return true, nil
}

// DeletePortProxyRule deletes the specified portproxy rule.  If the rule did not exist, return error.
func (*FakeNetsh) DeletePortProxyRule(args []string) error {
	// Do Nothing
	return nil
}

// EnsureIPAddress checks if the specified IP Address is added to vEthernet (HNSTransparent) interface, if not, add it.  If the address existed, return true.
func (*FakeNetsh) EnsureIPAddress(args []string, ip net.IP) (bool, error) {
	return true, nil
}

// DeleteIPAddress checks if the specified IP address is present and, if so, deletes it.
func (*FakeNetsh) DeleteIPAddress(args []string) error {
	// Do Nothing
	return nil
}

// Restore runs `netsh exec` to restore portproxy or addresses using a file.
// TODO Check if this is required, most likely not
func (*FakeNetsh) Restore(args []string) error {
	// Do Nothing
	return nil
}

// GetInterfaceToAddIP returns the interface name where Service IP needs to be added
// IP Address needs to be added for netsh portproxy to redirect traffic
// Reads Environment variable INTERFACE_TO_ADD_SERVICE_IP, if it is not defined then "vEthernet (HNSTransparent)" is returned
func (*FakeNetsh) GetInterfaceToAddIP() string {
	return "Interface 1"
}

var _ = netsh.Interface(&FakeNetsh{})
