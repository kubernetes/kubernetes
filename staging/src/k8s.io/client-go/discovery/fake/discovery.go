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

	"github.com/googleapis/gnostic/OpenAPIv2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
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
	return nil, fmt.Errorf("GroupVersion %q not found", groupVersion)
}

func (c *FakeDiscovery) ServerResources() ([]*metav1.APIResourceList, error) {
	action := testing.ActionImpl{
		Verb:     "get",
		Resource: schema.GroupVersionResource{Resource: "resource"},
	}
	c.Invokes(action, nil)
	return c.Resources, nil
}

func (c *FakeDiscovery) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

func (c *FakeDiscovery) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

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

func (c *FakeDiscovery) ServerVersion() (*version.Info, error) {
	action := testing.ActionImpl{}
	action.Verb = "get"
	action.Resource = schema.GroupVersionResource{Resource: "version"}
	c.Invokes(action, nil)

	if c.FakedServerVersion != nil {
		return c.FakedServerVersion, nil
	}

	versionInfo := kubeversion.Get()
	return &versionInfo, nil
}

func (c *FakeDiscovery) OpenAPISchema() (*openapi_v2.Document, error) {
	return &openapi_v2.Document{}, nil
}

func (c *FakeDiscovery) RESTClient() restclient.Interface {
	return nil
}
