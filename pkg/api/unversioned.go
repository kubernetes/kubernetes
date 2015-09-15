/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api

import (
	"strings"
)

// This file contains API types that are unversioned.

// APIVersions lists the versions that are available, to allow clients to
// discover the API at /api, which is the root path of the legacy v1 API.
type APIVersions struct {
	// versions are the api versions that are available.
	Versions []string `json:"versions"`
}

// APIGroupList is a list of APIGroup, to allow clients to discover the API at
// /apis.
type APIGroupList struct {
	// groups is a list of APIGroup.
	Groups []APIGroup `json:"groups"`
}

// APIGroup contains the name, the supported versions, and the preferred version
// of a group.
type APIGroup struct {
	// name is the name of the group.
	Name string `json:"name"`
	// versions are the versions supported in this group.
	Versions []GroupVersion `json:"versions"`
	// preferredVersion is the version preferred by the API server, which
	// probably is the storage version.
	PreferredVersion GroupVersion `json:"preferredVersion,omitempty"`
}

// GroupVersion contains the "group/version" and "version" string of a version.
// It is made a struct to keep extensiblity.
type GroupVersion struct {
	// groupVersion specifies the API group and version in the form "group/version"
	GroupVersion string `json:"groupVersion"`
	// version specifies the version in the form of "version". This is to save
	// the clients the trouble of splitting the GroupVersion.
	Version string `json:"version"`
}

// APIResource specifies the name of a resource and whether it is namespaced.
type APIResource struct {
	// name is the name of the resource.
	Name string `json:"name"`
	// namespaced indicates if a resource is namespaced or not.
	Namespaced bool `json:"namespaced"`
}

// APIResourceList is a list of APIResource, it is used to expose the name of the
// resources supported in a specific group and version, and if the resource
// is namespaced.
type APIResourceList struct {
	// groupVersion is the group and version this APIResourceList is for.
	GroupVersion string `json:"groupVersion"`
	// resources contains the name of the resources and if they are namespaced.
	APIResources []APIResource `json:"resources"`
}

// RootPaths lists the paths available at root.
// For example: "/healthz", "/apis".
type RootPaths struct {
	// paths are the paths available at root.
	Paths []string `json:"paths"`
}

// TODO: remove me when watch is refactored
func LabelSelectorQueryParam(version string) string {
	return "labelSelector"
}

// TODO: remove me when watch is refactored
func FieldSelectorQueryParam(version string) string {
	return "fieldSelector"
}

// String returns available api versions as a human-friendly version string.
func (apiVersions APIVersions) String() string {
	return strings.Join(apiVersions.Versions, ",")
}

func (apiVersions APIVersions) GoString() string {
	return apiVersions.String()
}

// Patch is provided to give a concrete name and type to the Kubernetes PATCH request body.
type Patch struct{}
