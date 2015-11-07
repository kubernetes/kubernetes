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
	"net/url"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
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
	v := &unversioned.APIVersions{}
	err = d.Get().AbsPath("/api").Do().Into(v)
	apiGroup := unversioned.APIGroup{}
	if err == nil {
		apiGroup = apiVersionsToAPIGroup(v)
	}
	if err != nil && !errors.IsNotFound(err) && !errors.IsForbidden(err) {
		return nil, err
	}

	// Get the groupVersions exposed at /apis
	apiGroupList = &unversioned.APIGroupList{}
	err = d.Get().AbsPath("/apis").Do().Into(apiGroupList)
	if err != nil && !errors.IsNotFound(err) && !errors.IsForbidden(err) {
		return nil, err
	}
	// to be compatible with a v1.0 server, if it's a 403 or 404, ignore and return whatever we got from /api
	if err != nil && (errors.IsNotFound(err) || errors.IsForbidden(err)) {
		apiGroupList = &unversioned.APIGroupList{}
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
	resources = &unversioned.APIResourceList{}
	err = d.Get().AbsPath(url.String()).Do().Into(resources)
	if err != nil {
		// ignore 403 or 404 error to be compatible with an v1.0 server.
		if groupVersion == "v1" && (errors.IsNotFound(err) || errors.IsForbidden(err)) {
			return resources, nil
		} else {
			return nil, err
		}
	}
	return resources, nil
}

// ServerResources returns the supported resources for all groups and versions.
func (d *DiscoveryClient) ServerResources() (map[string]*unversioned.APIResourceList, error) {
	apiGroups, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}
	groupVersions := ExtractGroupVersions(apiGroups)
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
	// Discovery client deals with unversioned objects, so we use api.Codec.
	config.Codec = api.Codec
	return nil
}

// NewDiscoveryClient creates a new DiscoveryClient for the given config. This client
// can be used to discover supported resources in the API server.
func NewDiscoveryClient(c *Config) (*DiscoveryClient, error) {
	config := *c
	if err := setDiscoveryDefaults(&config); err != nil {
		return nil, err
	}
	client, err := UnversionedRESTClientFor(&config)
	return &DiscoveryClient{client}, err
}
