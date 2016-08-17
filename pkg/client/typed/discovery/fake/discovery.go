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
	"github.com/emicklei/go-restful/swagger"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/version"
)

type FakeDiscovery struct {
	*core.Fake
}

func (c *FakeDiscovery) ServerResourcesForGroupVersion(groupVersion string) (*unversioned.APIResourceList, error) {
	action := core.ActionImpl{
		Verb:     "get",
		Resource: unversioned.GroupVersionResource{Resource: "resource"},
	}
	c.Invokes(action, nil)
	return c.Resources[groupVersion], nil
}

func (c *FakeDiscovery) ServerResources() (map[string]*unversioned.APIResourceList, error) {
	action := core.ActionImpl{
		Verb:     "get",
		Resource: unversioned.GroupVersionResource{Resource: "resource"},
	}
	c.Invokes(action, nil)
	return c.Resources, nil
}

func (c *FakeDiscovery) ServerPreferredResources() ([]unversioned.GroupVersionResource, error) {
	return nil, nil
}

func (c *FakeDiscovery) ServerPreferredNamespacedResources() ([]unversioned.GroupVersionResource, error) {
	return nil, nil
}

func (c *FakeDiscovery) ServerGroups() (*unversioned.APIGroupList, error) {
	return nil, nil
}

func (c *FakeDiscovery) ServerVersion() (*version.Info, error) {
	action := core.ActionImpl{}
	action.Verb = "get"
	action.Resource = unversioned.GroupVersionResource{Resource: "version"}

	c.Invokes(action, nil)
	versionInfo := version.Get()
	return &versionInfo, nil
}

func (c *FakeDiscovery) SwaggerSchema(version unversioned.GroupVersion) (*swagger.ApiDeclaration, error) {
	action := core.ActionImpl{}
	action.Verb = "get"
	if version == v1.SchemeGroupVersion {
		action.Resource = unversioned.GroupVersionResource{Resource: "/swaggerapi/api/" + version.Version}
	} else {
		action.Resource = unversioned.GroupVersionResource{Resource: "/swaggerapi/apis/" + version.Group + "/" + version.Version}
	}

	c.Invokes(action, nil)
	return &swagger.ApiDeclaration{}, nil
}
