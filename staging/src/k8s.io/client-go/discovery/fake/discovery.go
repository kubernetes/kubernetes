/*
Copyright 2016 The Kubernetes Authors.

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

package fake

import (
	"fmt"
	"net/http"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi"
	kubeversion "k8s.io/client-go/pkg/version"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/testing"
)

// FakeDiscovery implements discovery.DiscoveryInterface and sometimes calls testing.Fake.Invoke with an action,
// but doesn't respect the return value if any. There is a way to fake static values like ServerVersion by using the Faked... fields on the struct.
type FakeDiscovery struct {
	*testing.Fake
	FakedServerVersion *version.Info
}

// ServerResourcesForGroupVersion returns the supported resources for a group
// and version.
func (c *FakeDiscovery) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	action := testing.ActionImpl{
		Verb:     "get",
		Resource: schema.GroupVersionResource{Resource: "resource"},
	}
	c.Invokes(action, nil)
	for _, resourceList := range c.Resources {
		if resourceList.GroupVersion == groupVersion {
			return resourceList, nil
		}
	}
	return nil, &errors.StatusError{
		ErrStatus: metav1.Status{
			Status:  metav1.StatusFailure,
			Code:    http.StatusNotFound,
			Reason:  metav1.StatusReasonNotFound,
			Message: fmt.Sprintf("the server could not find the requested resource, GroupVersion %q not found", groupVersion),
		}}
}

// ServerGroupsAndResources returns the supported groups and resources for all groups and versions.
func (c *FakeDiscovery) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	sgs, err := c.ServerGroups()
	if err != nil {
		return nil, nil, err
	}
	resultGroups := []*metav1.APIGroup{}
	for i := range sgs.Groups {
		resultGroups = append(resultGroups, &sgs.Groups[i])
	}

	action := testing.ActionImpl{
		Verb:     "get",
		Resource: schema.GroupVersionResource{Resource: "resource"},
	}
	c.Invokes(action, nil)
	return resultGroups, c.Resources, nil
}

// ServerPreferredResources returns the supported resources with the version
// preferred by the server.
func (c *FakeDiscovery) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

// ServerPreferredNamespacedResources returns the supported namespaced resources
// with the version preferred by the server.
func (c *FakeDiscovery) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

// ServerGroups returns the supported groups, with information like supported
// versions and the preferred version.
func (c *FakeDiscovery) ServerGroups() (*metav1.APIGroupList, error) {
	action := testing.ActionImpl{
		Verb:     "get",
		Resource: schema.GroupVersionResource{Resource: "group"},
	}
	c.Invokes(action, nil)

	groups := map[string]*metav1.APIGroup{}

	for _, res := range c.Resources {
		gv, err := schema.ParseGroupVersion(res.GroupVersion)
		if err != nil {
			return nil, err
		}
		group := groups[gv.Group]
		if group == nil {
			group = &metav1.APIGroup{
				Name: gv.Group,
				PreferredVersion: metav1.GroupVersionForDiscovery{
					GroupVersion: res.GroupVersion,
					Version:      gv.Version,
				},
			}
			groups[gv.Group] = group
		}

		group.Versions = append(group.Versions, metav1.GroupVersionForDiscovery{
			GroupVersion: res.GroupVersion,
			Version:      gv.Version,
		})
	}

	list := &metav1.APIGroupList{}
	for _, apiGroup := range groups {
		list.Groups = append(list.Groups, *apiGroup)
	}

	return list, nil

}

// ServerVersion retrieves and parses the server's version.
func (c *FakeDiscovery) ServerVersion() (*version.Info, error) {
	action := testing.ActionImpl{}
	action.Verb = "get"
	action.Resource = schema.GroupVersionResource{Resource: "version"}
	_, err := c.Invokes(action, nil)
	if err != nil {
		return nil, err
	}

	if c.FakedServerVersion != nil {
		return c.FakedServerVersion, nil
	}

	versionInfo := kubeversion.Get()
	return &versionInfo, nil
}

// OpenAPISchema retrieves and parses the swagger API schema the server supports.
func (c *FakeDiscovery) OpenAPISchema() (*openapi_v2.Document, error) {
	return &openapi_v2.Document{}, nil
}

func (c *FakeDiscovery) OpenAPIV3() openapi.Client {
	panic("unimplemented")
}

// RESTClient returns a RESTClient that is used to communicate with API server
// by this client implementation.
func (c *FakeDiscovery) RESTClient() restclient.Interface {
	return nil
}

func (c *FakeDiscovery) WithLegacy() discovery.DiscoveryInterface {
	panic("unimplemented")
}
