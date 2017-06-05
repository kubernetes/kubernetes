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

import (
	"net/http"

	compute "google.golang.org/api/compute/v1"
)

// BackendService Management

// GetBackendService retrieves a backend by name.
func (gce *GCECloud) GetBackendService(name string) (*compute.BackendService, error) {
	return gce.service.BackendServices.Get(gce.projectID, name).Do()
}

// UpdateBackendService applies the given BackendService as an update to an existing service.
func (gce *GCECloud) UpdateBackendService(bg *compute.BackendService) error {
	op, err := gce.service.BackendServices.Update(gce.projectID, bg.Name, bg).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// DeleteBackendService deletes the given BackendService by name.
func (gce *GCECloud) DeleteBackendService(name string) error {
	op, err := gce.service.BackendServices.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// CreateBackendService creates the given BackendService.
func (gce *GCECloud) CreateBackendService(bg *compute.BackendService) error {
	op, err := gce.service.BackendServices.Insert(gce.projectID, bg).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListBackendServices lists all backend services in the project.
func (gce *GCECloud) ListBackendServices() (*compute.BackendServiceList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.BackendServices.List(gce.projectID).Do()
}

// GetHealth returns the health of the BackendService identified by the given
// name, in the given instanceGroup. The instanceGroupLink is the fully
// qualified self link of an instance group.
func (gce *GCECloud) GetHealth(name string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	groupRef := &compute.ResourceGroupReference{Group: instanceGroupLink}
	return gce.service.BackendServices.GetHealth(gce.projectID, name, groupRef).Do()
}
