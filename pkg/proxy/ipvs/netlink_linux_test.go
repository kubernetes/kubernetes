// +build linux

/*
Copyright 2017 The Kubernetes Authors.

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

package ipvs

import (
	"testing"

	utilexec "k8s.io/utils/exec"
)

const testDummyDevice = "test-dummy"

func TestAddressBindAndUnbind(t *testing.T) {
	execer := utilexec.New()
	// Delete test dummy device and ignore errors
	deleteDummyDevice(execer, testDummyDevice)
	// Create a new dummy device
	exist, err := ensureDummyDevice(execer, testDummyDevice)
	if err != nil {
		t.Errorf("Unexpected error when try to create dummy device, error: %v", err)
	}
	if exist {
		t.Errorf("Unexpected dummy device: %s exist", testDummyDevice)
	}
	// Bind IP addresses to dummy device
	nl := NewNetLinkHandle()
	ips := []string{"1.2.3.4", "2001:db8::1:1"}
	for _, ip := range ips {
		exist, err = nl.EnsureAddressBind(ip, testDummyDevice)
		if err != nil {
			t.Errorf("Unexpected error when try to bind IP address: %s to dummy device, error: %v", ip, err)
		}
		if exist {
			t.Errorf("Unexpected IP address: %s is already bound to dummy device: %s", ip, testDummyDevice)
		}
	}
	// Bind again and expect to see no error
	for _, ip := range ips {
		exist, err = nl.EnsureAddressBind(ip, testDummyDevice)
		if err != nil {
			t.Errorf("Unexpected error when try to bind an existing IP address: %s to dummy device, error: %v", ip, err)
		}
		if !exist {
			t.Errorf("Unexpected IP address: %s is not bound to dummy device: %s", ip, testDummyDevice)
		}
	}
	// Unbind IP addresses from dummy device
	for _, ip := range ips {
		err = nl.UnbindAddress(ip, testDummyDevice)
		if err != nil {
			t.Errorf("Unexpected error when try to unbind IP address: %s from dummy device, error: %v", ip, err)
		}
	}
	// UnBind again and expect to see errors
	for _, ip := range ips {
		err = nl.UnbindAddress(ip, testDummyDevice)
		if err == nil {
			t.Errorf("Expected error when try to unbind IP address: %s from dummy device, got nil", ip)
		}
	}
}
