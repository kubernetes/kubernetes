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

package util

import (
	"io/ioutil"
	"os"
	"testing"
	"time"

	"github.com/emicklei/go-restful/swagger"
	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/version"
)

func TestCachedDiscoveryClient_Fresh(t *testing.T) {
	assert := assert.New(t)

	d, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(d)

	c := fakeDiscoveryClient{}
	cdc := NewCachedDiscoveryClient(&c, d, 60*time.Second)
	assert.True(cdc.Fresh(), "should be fresh after creation")

	cdc.ServerGroups()
	assert.True(cdc.Fresh(), "should be fresh after groups call without cache")
	assert.Equal(c.groupCalls, 1)

	cdc.ServerGroups()
	assert.True(cdc.Fresh(), "should be fresh after another groups call")
	assert.Equal(c.groupCalls, 1)

	cdc.ServerResources()
	assert.True(cdc.Fresh(), "should be fresh after resources call")
	assert.Equal(c.resourceCalls, 1)

	cdc.ServerResources()
	assert.True(cdc.Fresh(), "should be fresh after another resources call")
	assert.Equal(c.resourceCalls, 1)

	cdc = NewCachedDiscoveryClient(&c, d, 60*time.Second)
	cdc.ServerGroups()
	assert.False(cdc.Fresh(), "should NOT be fresh after recreation with existing groups cache")
	assert.Equal(c.groupCalls, 1)

	cdc.ServerResources()
	assert.False(cdc.Fresh(), "should NOT be fresh after recreation with existing resources cache")
	assert.Equal(c.resourceCalls, 1)

	cdc.Invalidate()
	assert.True(cdc.Fresh(), "should be fresh after cache invalidation")

	cdc.ServerResources()
	assert.True(cdc.Fresh(), "should ignore existing resources cache after invalidation")
	assert.Equal(c.resourceCalls, 2)
}

func TestNewCachedDiscoveryClient_TTL(t *testing.T) {
	assert := assert.New(t)

	d, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(d)

	c := fakeDiscoveryClient{}
	cdc := NewCachedDiscoveryClient(&c, d, 1*time.Nanosecond)
	cdc.ServerGroups()
	assert.Equal(c.groupCalls, 1)

	time.Sleep(1 * time.Second)

	cdc.ServerGroups()
	assert.Equal(c.groupCalls, 2)
}

type fakeDiscoveryClient struct {
	groupCalls    int
	resourceCalls int
	versionCalls  int
	swaggerCalls  int
}

var _ discovery.DiscoveryInterface = &fakeDiscoveryClient{}

func (c *fakeDiscoveryClient) RESTClient() restclient.Interface {
	return &fake.RESTClient{}
}

func (c *fakeDiscoveryClient) ServerGroups() (*unversioned.APIGroupList, error) {
	c.groupCalls = c.groupCalls + 1
	return &unversioned.APIGroupList{
		Groups: []unversioned.APIGroup{
			{
				Name: "a",
				Versions: []unversioned.GroupVersionForDiscovery{
					{
						GroupVersion: "a/v1",
						Version:      "v1",
					},
				},
				PreferredVersion: unversioned.GroupVersionForDiscovery{
					GroupVersion: "a/v1",
					Version:      "v1",
				},
			},
		},
	}, nil
}

func (c *fakeDiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (*unversioned.APIResourceList, error) {
	c.resourceCalls = c.resourceCalls + 1
	if groupVersion == "a/v1" {
		return &unversioned.APIResourceList{}, nil
	}

	return nil, errors.NewNotFound(unversioned.GroupResource{}, "")
}

func (c *fakeDiscoveryClient) ServerResources() (map[string]*unversioned.APIResourceList, error) {
	c.resourceCalls = c.resourceCalls + 1
	return map[string]*unversioned.APIResourceList{}, nil
}

func (c *fakeDiscoveryClient) ServerPreferredResources() ([]unversioned.GroupVersionResource, error) {
	c.resourceCalls = c.resourceCalls + 1
	return []unversioned.GroupVersionResource{}, nil
}

func (c *fakeDiscoveryClient) ServerPreferredNamespacedResources() ([]unversioned.GroupVersionResource, error) {
	c.resourceCalls = c.resourceCalls + 1
	return []unversioned.GroupVersionResource{}, nil
}

func (c *fakeDiscoveryClient) ServerVersion() (*version.Info, error) {
	c.versionCalls = c.versionCalls + 1
	return &version.Info{}, nil
}

func (c *fakeDiscoveryClient) SwaggerSchema(version unversioned.GroupVersion) (*swagger.ApiDeclaration, error) {
	c.swaggerCalls = c.swaggerCalls + 1
	return &swagger.ApiDeclaration{}, nil
}
