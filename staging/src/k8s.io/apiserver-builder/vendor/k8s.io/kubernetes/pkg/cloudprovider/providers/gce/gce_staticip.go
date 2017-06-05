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

package gce

import compute "google.golang.org/api/compute/v1"

// Global static IP management

// ReserveGlobalStaticIP creates a global static IP.
// Caller is allocated a random IP if they do not specify an ipAddress. If an
// ipAddress is specified, it must belong to the current project, eg: an
// ephemeral IP associated with a global forwarding rule.
func (gce *GCECloud) ReserveGlobalStaticIP(name, ipAddress string) (address *compute.Address, err error) {
	op, err := gce.service.GlobalAddresses.Insert(gce.projectID, &compute.Address{Name: name, Address: ipAddress}).Do()
	if err != nil {
		return nil, err
	}
	if err := gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	// We have to get the address to know which IP was allocated for us.
	return gce.service.GlobalAddresses.Get(gce.projectID, name).Do()
}

// DeleteGlobalStaticIP deletes a global static IP by name.
func (gce *GCECloud) DeleteGlobalStaticIP(name string) error {
	op, err := gce.service.GlobalAddresses.Delete(gce.projectID, name).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// GetGlobalStaticIP returns the global static IP by name.
func (gce *GCECloud) GetGlobalStaticIP(name string) (address *compute.Address, err error) {
	return gce.service.GlobalAddresses.Get(gce.projectID, name).Do()
}
