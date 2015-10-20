/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"encoding/json"
	"fmt"
	"net/url"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// DiscoveryInterface holds the methods that discover server-supported API groups,
// versions and resources.
type DiscoveryInterface interface {
	ServerGroupsInterface
	ServerResourcesInterface
}

// ServerGroupsInterface has methods for obtaining supported groups on the API server
type ServerGroupsInterface interface {
	// ServerGroups returns the supported groups, with information like supported versions and the
	// preferred version.
	ServerGroups() (*unversioned.APIGroupList, error)
}

// ServerResourcesInterface has methods for obtaining supported resources on the API server
type ServerResourcesInterface interface {
	// ServerResourcesForGroupVersion returns the supported resources for a group and version.
	ServerResourcesForGroupVersion(groupVersion string) (*unversioned.APIResourceList, error)
	// ServerResources returns the supported resources for all groups and versions.
	ServerResources() (map[string]*unversioned.APIResourceList, error)
}

// DiscoveryClient implements the functions that dicovery server-supported API groups,
// versions and resources.
type DiscoveryClient struct {
	*RESTClient
}

// Convert unversioned.APIVersions to unversioned.APIGroup. APIVersions is used by legacy v1, so
// group would be "".
func apiVersionsToAPIGroup(apiVersions *unversioned.APIVersions) (apiGroup unversioned.APIGroup) {
	groupVersions := []unversioned.GroupVersion{}
	for _, version := range apiVersions.Versions {
		groupVersion := unversioned.GroupVersion{
			GroupVersion: version,
			Version:      version,
		}
		groupVersions = append(groupVersions, groupVersion)
	}
	apiGroup.Versions = groupVersions
	// There should be only one groupVersion returned at /api
	apiGroup.PreferredVersion = groupVersions[0]
	return
}

// ServerGroups returns the supported groups, with information like supported versions and the
// preferred version.
func (d *DiscoveryClient) ServerGroups() (apiGroupList *unversioned.APIGroupList, err error) {
	// Get the groupVersions exposed at /api
	url := url.URL{}
	url.Path = "/api"
	resp, err := d.Get().AbsPath(url.String()).Do().Raw()
	if err != nil {
		return nil, err
	}
	var v unversioned.APIVersions
	err = json.Unmarshal(resp, &v)
	if err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}
	apiGroup := apiVersionsToAPIGroup(&v)

	// Get the groupVersions exposed at /apis
	url.Path = "/apis"
	resp2, err := d.Get().AbsPath(url.String()).Do().Raw()
	if err != nil {
		return nil, err
	}
	apiGroupList = &unversioned.APIGroupList{}
	if err = json.Unmarshal(resp2, &apiGroupList); err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}

	// append the group retrieved from /api to the list
	apiGroupList.Groups = append(apiGroupList.Groups, apiGroup)
	return apiGroupList, nil
}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *DiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (resources *unversioned.APIResourceList, err error) {
	url := url.URL{}
	if groupVersion == "v1" {
		url.Path = "/api/" + groupVersion
	} else {
		url.Path = "/apis/" + groupVersion
	}
	resp, err := d.Get().AbsPath(url.String()).Do().Raw()
	if err != nil {
		return nil, err
	}
	resources = &unversioned.APIResourceList{}
	if err = json.Unmarshal(resp, resources); err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}
	return resources, nil
}

// ServerResources returns the supported resources for all groups and versions.
func (d *DiscoveryClient) ServerResources() (map[string]*unversioned.APIResourceList, error) {
	apiGroups, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}
	groupVersions := extractGroupVersions(apiGroups)
	result := map[string]*unversioned.APIResourceList{}
	for _, groupVersion := range groupVersions {
		resources, err := d.ServerResourcesForGroupVersion(groupVersion)
		if err != nil {
			return nil, err
		}
		result[groupVersion] = resources
	}
	return result, nil
}

func setDiscoveryDefaults(config *Config) error {
	config.Prefix = ""
	config.Version = ""
	// The discoveryClient shouldn't need a codec for now.
	config.Codec = nil
	return nil
}

// NewDiscoveryClient creates a new DiscoveryClient for the given config. This client
// can be used to discover supported resources in the API server.
func NewDiscoveryClient(c *Config) (*DiscoveryClient, error) {
	config := *c
	if err := setDiscoveryDefaults(&config); err != nil {
		return nil, err
	}
	client, err := RESTClientFor(&config)
	return &DiscoveryClient{client}, err
}
