// Copyright 2019 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package discovery_v1

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/googleapis/gnostic/compiler"
)

// APIsListServiceURL is the URL for the Google APIs Discovery Service
const APIsListServiceURL = "https://www.googleapis.com/discovery/v1/apis"

// A List represents the results of a call to the apis/list API.
// https://developers.google.com/discovery/v1/reference/apis/list
type List struct {
	Kind             string `json:"kind"`
	DiscoveryVersion string `json:"discoveryVersion"`
	APIs             []*API `json:"items"`
}

func FetchListBytes() ([]byte, error) {
	return compiler.FetchFile(APIsListServiceURL)
}

// Read the list of APIs from the apis/list service.
func FetchList() (*List, error) {
	bytes, err := FetchListBytes()
	if err != nil {
		return nil, err
	}
	return ParseList(bytes)
}

// ParseList unmarshals the bytes into a Document.
func ParseList(bytes []byte) (*List, error) {
	var listResponse List
	err := json.Unmarshal(bytes, &listResponse)
	return &listResponse, err
}

// An API represents an API description returned by the apis/list API.
type API struct {
	Kind              string            `json:"kind"`
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Version           string            `json:"version"`
	Title             string            `json:"title"`
	Description       string            `json:"description"`
	DiscoveryRestURL  string            `json:"discoveryRestUrl"`
	DiscoveryLink     string            `json:"discoveryLink"`
	Icons             map[string]string `json:"icons"`
	DocumentationLink string            `json:"documentationLink"`
	Labels            []string          `json:"labels"`
	Preferred         bool              `json:"preferred"`
}

// APIWithNameAndVersion returns the API with a specified name and version.
// If version is the empty string, the API name must be unique.
func (a *List) APIWithNameAndVersion(name string, version string) (*API, error) {
	var api *API                  // the API to return
	versions := make([]string, 0) // the matching version names
	// Scan the list for matching APIs and versions.
	for _, item := range a.APIs {
		if item.Name == name {
			if version == "" || version == item.Version {
				api = item
				versions = append(versions, item.Version)
			}
		}
	}
	switch {
	case len(versions) == 0:
		return nil, errors.New(name + " was not found.")
	case len(versions) > 1:
		return nil, errors.New(name + " has multiple versions: " + strings.Join(versions, ", "))
	default:
		return api, nil
	}
}
