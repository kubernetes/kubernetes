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

// Health Checks

// GetHttpHealthCheck returns the given HttpHealthCheck by name.
func (gce *GCECloud) GetHttpHealthCheck(name string) (*compute.HttpHealthCheck, error) {
	return gce.service.HttpHealthChecks.Get(gce.projectID, name).Do()
}

// UpdateHttpHealthCheck applies the given HttpHealthCheck as an update.
func (gce *GCECloud) UpdateHttpHealthCheck(hc *compute.HttpHealthCheck) error {
	op, err := gce.service.HttpHealthChecks.Update(gce.projectID, hc.Name, hc).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// DeleteHttpHealthCheck deletes the given HttpHealthCheck by name.
func (gce *GCECloud) DeleteHttpHealthCheck(name string) error {
	op, err := gce.service.HttpHealthChecks.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// CreateHttpHealthCheck creates the given HttpHealthCheck.
func (gce *GCECloud) CreateHttpHealthCheck(hc *compute.HttpHealthCheck) error {
	op, err := gce.service.HttpHealthChecks.Insert(gce.projectID, hc).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListHttpHealthCheck lists all HttpHealthChecks in the project.
func (gce *GCECloud) ListHttpHealthChecks() (*compute.HttpHealthCheckList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.HttpHealthChecks.List(gce.projectID).Do()
}
