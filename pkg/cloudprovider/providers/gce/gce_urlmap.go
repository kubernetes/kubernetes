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

// UrlMap management

// GetUrlMap returns the UrlMap by name.
func (gce *GCECloud) GetUrlMap(name string) (*compute.UrlMap, error) {
	return gce.service.UrlMaps.Get(gce.projectID, name).Do()
}

// CreateUrlMap creates an url map, using the given backend service as the default service.
func (gce *GCECloud) CreateUrlMap(backend *compute.BackendService, name string) (*compute.UrlMap, error) {
	urlMap := &compute.UrlMap{
		Name:           name,
		DefaultService: backend.SelfLink,
	}
	op, err := gce.service.UrlMaps.Insert(gce.projectID, urlMap).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.GetUrlMap(name)
}

// UpdateUrlMap applies the given UrlMap as an update, and returns the new UrlMap.
func (gce *GCECloud) UpdateUrlMap(urlMap *compute.UrlMap) (*compute.UrlMap, error) {
	op, err := gce.service.UrlMaps.Update(gce.projectID, urlMap.Name, urlMap).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.service.UrlMaps.Get(gce.projectID, urlMap.Name).Do()
}

// DeleteUrlMap deletes a url map by name.
func (gce *GCECloud) DeleteUrlMap(name string) error {
	op, err := gce.service.UrlMaps.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListUrlMaps lists all UrlMaps in the project.
func (gce *GCECloud) ListUrlMaps() (*compute.UrlMapList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.UrlMaps.List(gce.projectID).Do()
}
