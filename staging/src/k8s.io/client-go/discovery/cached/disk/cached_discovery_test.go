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

package disk

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"time"

	openapi_v2 "github.com/google/gnostic/openapiv2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/rest"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	testutil "k8s.io/client-go/util/testing"
)

// Tests that schema instances returned by openapi cached and returned after
// successive calls
func TestOpenAPIDiskCache(t *testing.T) {
	// Create discovery cache dir (unused)
	discoCache, err := ioutil.TempDir("", "")
	require.NoError(t, err)
	os.RemoveAll(discoCache)
	defer os.RemoveAll(discoCache)

	// Create http cache dir
	httpCache, err := ioutil.TempDir("", "")
	require.NoError(t, err)
	os.RemoveAll(httpCache)
	defer os.RemoveAll(httpCache)

	// Start test OpenAPI server
	fakeServer, err := testutil.NewFakeOpenAPIV3Server("../../testdata")
	require.NoError(t, err)
	defer fakeServer.HttpServer.Close()

	require.Greater(t, len(fakeServer.ServedDocuments), 0)

	client, err := NewCachedDiscoveryClientForConfig(
		&rest.Config{Host: fakeServer.HttpServer.URL},
		discoCache,
		httpCache,
		1*time.Nanosecond,
	)
	require.NoError(t, err)

	openapiClient := client.OpenAPIV3()

	// Ensure initial Paths call hits server
	_, err = openapiClient.Paths()
	require.NoError(t, err)
	assert.Equal(t, 1, fakeServer.RequestCounters["/openapi/v3"])

	// Ensure Paths call does hits server again
	// This is expected since openapiClient is the same instance, so Paths()
	// should be cached in memory.
	paths, err := openapiClient.Paths()
	require.NoError(t, err)
	assert.Equal(t, 2, fakeServer.RequestCounters["/openapi/v3"])

	require.Greater(t, len(paths), 0)
	i := 0
	for k, v := range paths {
		i++

		_, err = v.Schema()
		assert.NoError(t, err)

		path := "/openapi/v3/" + strings.TrimPrefix(k, "/")
		assert.Equal(t, 1, fakeServer.RequestCounters[path])

		// Ensure schema call is served from memory
		_, err = v.Schema()
		assert.NoError(t, err)
		assert.Equal(t, 1, fakeServer.RequestCounters[path])

		// Refetch the schema from a new openapi client to try to force a new
		// http request
		newPaths, err := client.OpenAPIV3().Paths()
		if !assert.NoError(t, err) {
			continue
		}

		// Ensure schema call is still served from disk
		_, err = newPaths[k].Schema()
		assert.NoError(t, err)
		assert.Equal(t, 2+i, fakeServer.RequestCounters["/openapi/v3"])
		assert.Equal(t, 1, fakeServer.RequestCounters[path])
	}
}

type fakeDiscoveryClient struct {
	groupCalls    int
	resourceCalls int
	versionCalls  int
	openAPICalls  int

	serverResourcesHandler func() ([]*metav1.APIResourceList, error)
}

var _ discovery.DiscoveryInterface = &fakeDiscoveryClient{}

func (c *fakeDiscoveryClient) RESTClient() restclient.Interface {
	return &fake.RESTClient{}
}

func (c *fakeDiscoveryClient) ServerGroups() (*metav1.APIGroupList, error) {
	c.groupCalls = c.groupCalls + 1
	return c.serverGroups()
}

func (c *fakeDiscoveryClient) serverGroups() (*metav1.APIGroupList, error) {
	return &metav1.APIGroupList{
		Groups: []metav1.APIGroup{
			{
				Name: "a",
				Versions: []metav1.GroupVersionForDiscovery{
					{
						GroupVersion: "a/v1",
						Version:      "v1",
					},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{
					GroupVersion: "a/v1",
					Version:      "v1",
				},
			},
		},
	}, nil
}

func (c *fakeDiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	c.resourceCalls = c.resourceCalls + 1
	if groupVersion == "a/v1" {
		return &metav1.APIResourceList{APIResources: []metav1.APIResource{{Name: "widgets", Kind: "Widget"}}}, nil
	}

	return nil, errors.NewNotFound(schema.GroupResource{}, "")
}

func (c *fakeDiscoveryClient) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	c.resourceCalls = c.resourceCalls + 1

	gs, _ := c.serverGroups()
	resultGroups := []*metav1.APIGroup{}
	for i := range gs.Groups {
		resultGroups = append(resultGroups, &gs.Groups[i])
	}

	if c.serverResourcesHandler != nil {
		rs, err := c.serverResourcesHandler()
		return resultGroups, rs, err
	}
	return resultGroups, []*metav1.APIResourceList{}, nil
}

func (c *fakeDiscoveryClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	c.resourceCalls = c.resourceCalls + 1
	return nil, nil
}

func (c *fakeDiscoveryClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	c.resourceCalls = c.resourceCalls + 1
	return nil, nil
}

func (c *fakeDiscoveryClient) ServerVersion() (*version.Info, error) {
	c.versionCalls = c.versionCalls + 1
	return &version.Info{}, nil
}

func (c *fakeDiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	c.openAPICalls = c.openAPICalls + 1
	return &openapi_v2.Document{}, nil
}

func (d *fakeDiscoveryClient) OpenAPIV3() openapi.Client {
	panic("unimplemented")
}
