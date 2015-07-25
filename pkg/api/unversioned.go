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

// APIVersions lists the api versions that are available, to allow
// version negotiation. APIVersions isn't just an unnamed array of
// strings in order to allow for future evolution, though unversioned
type APIVersions struct {
	Versions []string `json:"versions"`
}

// RootPaths lists the paths available at root.
// For example: "/healthz", "/api".
type RootPaths struct {
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
