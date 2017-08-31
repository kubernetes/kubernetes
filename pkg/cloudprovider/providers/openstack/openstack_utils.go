/*
Copyright 2014 The Kubernetes Authors.

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

package openstack

import (
	"errors"
	"net/http"
	"sort"
	"strings"

	"github.com/gophercloud/gophercloud"
	apiversions_v1 "github.com/gophercloud/gophercloud/openstack/blockstorage/v1/apiversions"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions"
	"github.com/gophercloud/gophercloud/pagination"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

var ErrNotFound = errors.New("Failed to find object")
var ErrMultipleResults = errors.New("Multiple results where only one expected")
var ErrNoAddressFound = errors.New("No address found for host")

// mapNodeNameToServerName maps a k8s NodeName to an OpenStack Server Name
// This is a simple string cast.
func mapNodeNameToServerName(nodeName types.NodeName) string {
	return string(nodeName)
}

// mapServerToNodeName maps an OpenStack Server to a k8s NodeName
func mapServerToNodeName(server *servers.Server) types.NodeName {
	// Node names are always lowercase, and (at least)
	// routecontroller does case-sensitive string comparisons
	// assuming this
	return types.NodeName(strings.ToLower(server.Name))
}

// Tiny helper for conditional unwind logic
type Caller bool

func NewCaller() Caller   { return Caller(true) }
func (c *Caller) Disarm() { *c = false }

func (c *Caller) Call(f func()) {
	if *c {
		f()
	}
}

func isNotFound(err error) bool {
	e, ok := err.(*gophercloud.ErrUnexpectedResponseCode)
	return ok && e.Actual == http.StatusNotFound
}

// Implementation of sort interface for blockstorage version probing
type APIVersionsByID []apiversions_v1.APIVersion

func (apiVersions APIVersionsByID) Len() int {
	return len(apiVersions)
}

func (apiVersions APIVersionsByID) Swap(i, j int) {
	apiVersions[i], apiVersions[j] = apiVersions[j], apiVersions[i]
}

func (apiVersions APIVersionsByID) Less(i, j int) bool {
	return apiVersions[i].ID > apiVersions[j].ID
}

func autoVersionSelector(apiVersion *apiversions_v1.APIVersion) string {
	switch strings.ToLower(apiVersion.ID) {
	case "v2.0":
		return "v2"
	case "v1.0":
		return "v1"
	default:
		return ""
	}
}

func doBsApiVersionAutodetect(availableApiVersions []apiversions_v1.APIVersion) string {
	sort.Sort(APIVersionsByID(availableApiVersions))
	for _, status := range []string{"CURRENT", "SUPPORTED"} {
		for _, version := range availableApiVersions {
			if strings.ToUpper(version.Status) == status {
				if detectedApiVersion := autoVersionSelector(&version); detectedApiVersion != "" {
					glog.V(3).Infof("Blockstorage API version probing has found a suitable %s api version: %s", status, detectedApiVersion)
					return detectedApiVersion
				}
			}
		}
	}

	return ""

}

func networkExtensions(client *gophercloud.ServiceClient) (map[string]bool, error) {
	seen := make(map[string]bool)

	pager := extensions.List(client)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		exts, err := extensions.ExtractExtensions(page)
		if err != nil {
			return false, err
		}
		for _, ext := range exts {
			seen[ext.Alias] = true
		}
		return true, nil
	})

	return seen, err
}
