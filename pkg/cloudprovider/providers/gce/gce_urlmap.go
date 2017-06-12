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

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
)

func newUrlMapMetricContext(request string) *metricContext {
	return newGenericMetricContext("urlmap", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetUrlMap returns the UrlMap by name.
func (gce *GCECloud) GetUrlMap(name string) (*compute.UrlMap, error) {
	mc := newUrlMapMetricContext("get")
	glog.V(4).Infof("UrlMaps.Get(%s, %s): start", gce.projectID, name)
	v, err := gce.service.UrlMaps.Get(gce.projectID, name).Do()
	glog.V(4).Infof("UrlMaps.Get(%s, %s): end", gce.projectID, name)
	return v, mc.Observe(err)
}

// CreateUrlMap creates a url map
func (gce *GCECloud) CreateUrlMap(urlMap *compute.UrlMap) error {
	mc := newUrlMapMetricContext("create")
	glog.V(4).Infof("UrlMaps.Insert(%s, %v): start", gce.projectID, urlMap)
	op, err := gce.service.UrlMaps.Insert(gce.projectID, urlMap).Do()
	glog.V(4).Infof("UrlMaps.Insert(%s, %v): end", gce.projectID, urlMap)
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// UpdateUrlMap applies the given UrlMap as an update
func (gce *GCECloud) UpdateUrlMap(urlMap *compute.UrlMap) error {
	mc := newUrlMapMetricContext("update")
	glog.V(4).Infof("UrlMaps.Update(%s, %s, %v): start", gce.projectID, urlMap.Name, urlMap)
	op, err := gce.service.UrlMaps.Update(gce.projectID, urlMap.Name, urlMap).Do()
	glog.V(4).Infof("UrlMaps.Update(%s, %s, %v): end", gce.projectID, urlMap.Name, urlMap)
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// DeleteUrlMap deletes a url map by name.
func (gce *GCECloud) DeleteUrlMap(name string) error {
	mc := newUrlMapMetricContext("delete")
	glog.V(4).Infof("UrlMaps.Delete(%s, %s): start", gce.projectID, name)
	op, err := gce.service.UrlMaps.Delete(gce.projectID, name).Do()
	glog.V(4).Infof("UrlMaps.Delete(%s, %s): end", gce.projectID, name)
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// ListUrlMaps lists all UrlMaps in the project.
func (gce *GCECloud) ListUrlMaps() (*compute.UrlMapList, error) {
	mc := newUrlMapMetricContext("list")
	// TODO: use PageToken to list all not just the first 500
	glog.V(4).Infof("UrlMaps.List(%s): start", gce.projectID)
	v, err := gce.service.UrlMaps.List(gce.projectID).Do()
	glog.V(4).Infof("UrlMaps.List(%s): end", gce.projectID)
	return v, mc.Observe(err)
}
