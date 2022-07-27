/*
Copyright 2022 The Kubernetes Authors.

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

package aggregated

import (
	"context"
	"fmt"
	"net/http"

	//nolint:staticcheck // SA1019 Keep using module since it's still being maintained and the api of google.golang.org/protobuf/proto differs
	openapi_v2 "github.com/google/gnostic/openapiv2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/disk"
	"k8s.io/client-go/openapi"
	restclient "k8s.io/client-go/rest"
)

const AggregatedDiscoveryEndpoint = "discovery/v1"

// AggregatedDiscoveryClient implements the functions that discovery server-supported API groups,
// versions and resources.
type AggregatedDiscoveryClient struct {
	delegate discovery.DiscoveryInterface
}

var _ discovery.CachedDiscoveryInterface = &AggregatedDiscoveryClient{}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (d *AggregatedDiscoveryClient) RESTClient() restclient.Interface {
	return d.delegate.RESTClient()
}

// ServerGroups returns the supported groups, with information like supported versions and the
// preferred version.
func (d *AggregatedDiscoveryClient) ServerGroups() (apiGroupList *metav1.APIGroupList, err error) {
	apiGroupList = &metav1.APIGroupList{}
	groups, _, err := d.ServerGroupsAndResources()
	for _, group := range groups {
		apiGroupList.Groups = append(apiGroupList.Groups, *group)
	}
	return apiGroupList, err
}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *AggregatedDiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	_, resources, err := d.ServerGroupsAndResources()
	if err != nil {
		return &metav1.APIResourceList{}, err
	}
	for _, resourceList := range resources {
		if resourceList.GroupVersion == groupVersion {
			return resourceList, nil
		}
	}
	return &metav1.APIResourceList{}, fmt.Errorf("Resources for GroupVersion %s not found", groupVersion)
}

// ServerGroupsAndResources returns the supported resources for all groups and versions.
func (d *AggregatedDiscoveryClient) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	// Retrieve the aggregated discovery document as a DiscoveryAPIGroupList.
	gl := &metav1.DiscoveryAPIGroupList{}
	err := d.RESTClient().Get().AbsPath(AggregatedDiscoveryEndpoint).Do(context.TODO()).Into(gl)
	if err != nil {
		return []*metav1.APIGroup{}, []*metav1.APIResourceList{}, err
	}
	// Convert the DiscoveryAPI* structs into APIGroups and APIResourceLists.
	groups := []*metav1.APIGroup{}
	resources := []*metav1.APIResourceList{}
	for _, g := range gl.Groups {
		group, gvResources := convertAPIGroup(g)
		groups = append(groups, group)
		resources = append(resources, gvResources...)
	}
	return groups, resources, nil
}

// ServerPreferredResources returns the supported resources with the version preferred by the
// server.
func (d *AggregatedDiscoveryClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	groups, resources, err := d.ServerGroupsAndResources()
	if err != nil {
		return []*metav1.APIResourceList{}, err
	}
	// Create a map of preferred GroupVersions.
	preferredVersions := map[string]bool{}
	for _, group := range groups {
		for _, version := range group.Versions {
			if group.PreferredVersion.GroupVersion == version.GroupVersion {
				preferredVersions[version.GroupVersion] = true
			}
		}
	}
	// Filter the ResourceLists accepting only ResourceLists with a preferred GroupVersion.
	preferredResources := []*metav1.APIResourceList{}
	for _, resource := range resources {
		if _, ok := preferredVersions[resource.GroupVersion]; ok {
			preferredResources = append(preferredResources, resource)
		}
	}
	return preferredResources, nil
}

// ServerPreferredNamespacedResources returns the supported namespaced resources with the
// version preferred by the server.
func (d *AggregatedDiscoveryClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	resources, err := d.ServerPreferredResources()
	return discovery.FilteredBy(discovery.ResourcePredicateFunc(func(groupVersion string, r *metav1.APIResource) bool {
		return r.Namespaced
	}), resources), err
}

// ServerVersion retrieves and parses the server's version (git version).
func (d *AggregatedDiscoveryClient) ServerVersion() (*version.Info, error) {
	return d.delegate.ServerVersion()
}

// OpenAPISchema fetches the open api v2 schema using a rest client and parses the proto.
func (d *AggregatedDiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	return d.delegate.OpenAPISchema()
}

func (d *AggregatedDiscoveryClient) OpenAPIV3() openapi.Client {
	return openapi.NewClient(d.RESTClient())
}

// Fresh always return true, signaling no need to retry.
func (d *AggregatedDiscoveryClient) Fresh() bool {
	return true
}

// No need to invalidate the discovery client.
func (d *AggregatedDiscoveryClient) Invalidate() {}

// NewAggregatedDiscoveryClientForConfig creates and returns an AggregatedDiscoveryClient
// (or an error) that implements the CachedDiscoveryClient interface. This client contains
// a RoundTripper that caches the single aggregated discovery document, using etags to
// determine the freshness/staleness of the document.
func NewAggregatedDiscoveryClientForConfig(config *restclient.Config, httpCacheDir string) (*AggregatedDiscoveryClient, error) {
	if len(httpCacheDir) > 0 {
		// update the given restconfig with a custom roundtripper that
		// understands how to handle cache responses. Uses etags from
		// response to determine freshness/staleness.
		config = restclient.CopyConfig(config)
		config.Wrap(func(rt http.RoundTripper) http.RoundTripper {
			return disk.NewCacheRoundTripper(httpCacheDir, rt)
		})
	}
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}
	return &AggregatedDiscoveryClient{delegate: discoveryClient}, nil
}

// convertAPIGroup tranforms a DiscoveryAPIGroup to an APIGroup, also returning
// the list of APIResourceList for resources within GroupVersions.
func convertAPIGroup(g metav1.DiscoveryAPIGroup) (*metav1.APIGroup, []*metav1.APIResourceList) {
	group := &metav1.APIGroup{}
	gvResources := []*metav1.APIResourceList{}
	group.Name = g.Name
	for i, v := range g.Versions {
		version := metav1.GroupVersionForDiscovery{}
		version.GroupVersion = fmt.Sprintf("%s/%s", g.Name, v.Version)
		version.Version = v.Version
		group.Versions = append(group.Versions, version)
		if i == 0 {
			group.PreferredVersion = version
		}
		resourceList := &metav1.APIResourceList{}
		resourceList.GroupVersion = version.GroupVersion
		for _, r := range v.APIResources {
			resource := convertAPIResource(r)
			resourceList.APIResources = append(resourceList.APIResources, resource)
		}
		gvResources = append(gvResources, resourceList)
	}
	return group, gvResources
}

// convertAPIResource tranforms a DiscoveryAPIResource to an APIResource.
func convertAPIResource(in metav1.DiscoveryAPIResource) metav1.APIResource {
	resource := metav1.APIResource{}
	resource.Name = in.Name
	resource.SingularName = in.SingularName
	resource.Namespaced = in.Namespaced
	resource.Group = in.Group
	resource.Version = in.Version
	resource.Kind = in.Kind
	resource.Verbs = in.Verbs
	resource.ShortNames = in.ShortNames
	resource.Categories = in.Categories
	//resource.StorageVersionHash = in.StorageVersionHash ?? Shouldn't this be stored

	return resource
}
