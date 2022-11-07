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

package v2

import (
	"bytes"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
)

var apiDiscoveryJSON string = `{"openapi":"3.0.0","info":{"title":"Kubernetes","version":"v1.26.0"},"paths":{"/apis/discovery.k8s.io/":{"get":{"tags":["discovery"],"description":"get information of a group","operationId":"getDiscoveryAPIGroup","responses":{"200":{"description":"OK","content":{"application/json":{"schema":{"$ref":"#/components/schemas/io.k8s.apimachinery.pkg.apis.meta.v1.APIGroup"}},"application/vnd.kubernetes.protobuf":{"schema":{"$ref":"#/components/schemas/io.k8s.apimachinery.pkg.apis.meta.v1.APIGroup"}},"application/yaml":{"schema":{"$ref":"#/components/schemas/io.k8s.apimachinery.pkg.apis.meta.v1.APIGroup"}}}},"401":{"description":"Unauthorized"}}}}},"components":{"schemas":{"io.k8s.apimachinery.pkg.apis.meta.v1.APIGroup":{"description":"APIGroup contains the name, the supported versions, and the preferred version of a group.","type":"object","required":["name","versions"],"properties":{"apiVersion":{"description":"APIVersion defines the versioned schema of this representation of an object. Servers should convert recognized schemas to the latest internal value, and may reject unrecognized values. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources","type":"string"},"kind":{"description":"Kind is a string value representing the REST resource this object represents. Servers may infer this from the endpoint the client submits requests to. Cannot be updated. In CamelCase. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds","type":"string"},"name":{"description":"name is the name of the group.","type":"string","default":""},"preferredVersion":{"description":"preferredVersion is the version preferred by the API server, which probably is the storage version.","default":{},"allOf":[{"$ref":"#/components/schemas/io.k8s.apimachinery.pkg.apis.meta.v1.GroupVersionForDiscovery"}]},"serverAddressByClientCIDRs":{"description":"a map of client CIDR to server address that is serving this group. This is to help clients reach servers in the most network-efficient way possible. Clients can use the appropriate server address as per the CIDR that they match. In case of multiple matches, clients should use the longest matching CIDR. The server returns only those CIDRs that it thinks that the client can match. For example: the master will return an internal IP CIDR only, if the client reaches the server using an internal IP. Server looks at X-Forwarded-For header or X-Real-Ip header or request.RemoteAddr (in that order) to get the client IP.","type":"array","items":{"default":{},"allOf":[{"$ref":"#/components/schemas/io.k8s.apimachinery.pkg.apis.meta.v1.ServerAddressByClientCIDR"}]}},"versions":{"description":"versions are the versions supported in this group.","type":"array","items":{"default":{},"allOf":[{"$ref":"#/components/schemas/io.k8s.apimachinery.pkg.apis.meta.v1.GroupVersionForDiscovery"}]}}},"x-kubernetes-group-version-kind":[{"group":"","kind":"APIGroup","version":"v1"}]},"io.k8s.apimachinery.pkg.apis.meta.v1.GroupVersionForDiscovery":{"description":"GroupVersion contains the \"group/version\" and \"version\" string of a version. It is made a struct to keep extensibility.","type":"object","required":["groupVersion","version"],"properties":{"groupVersion":{"description":"groupVersion specifies the API group and version in the form \"group/version\"","type":"string","default":""},"version":{"description":"version specifies the version in the form of \"version\". This is to save the clients the trouble of splitting the GroupVersion.","type":"string","default":""}}},"io.k8s.apimachinery.pkg.apis.meta.v1.ServerAddressByClientCIDR":{"description":"ServerAddressByClientCIDR helps the client to determine the server address that they should use, depending on the clientCIDR that they match.","type":"object","required":["clientCIDR","serverAddress"],"properties":{"clientCIDR":{"description":"The CIDR with which clients can match their IP to figure out the server address that they should use.","type":"string","default":""},"serverAddress":{"description":"Address of this server, suitable for a client that matches the above CIDR. This can be a hostname, hostname:port, IP or IP:port.","type":"string","default":""}}}},"securitySchemes":{"BearerToken":{"type":"apiKey","description":"Bearer Token authentication","name":"authorization","in":"header"}}}}`
var apiGroupsGVR schema.GroupVersionResource = schema.GroupVersionResource{
	Group:    "discovery.k8s.io",
	Version:  "v1",
	Resource: "apigroups",
}
var apiGroupsDocument map[string]interface{} = func() map[string]interface{} {
	var doc map[string]interface{}

	err := json.Unmarshal([]byte(apiDiscoveryJSON), &doc)
	if err != nil {
		panic(err)
	}

	return doc
}()

type FakeOpenAPIV3Client struct {
	// Path:
	//		ContentType:
	//			OpenAPIV3 Schema bytes
	Values      map[string]map[string][]byte
	FetchCounts map[string]map[string]int
	lock        sync.Mutex
}

type FakeGroupVersion struct {
	Data        map[string][]byte
	FetchCounts map[string]int
	Lock        *sync.Mutex
}

func (f *FakeGroupVersion) Schema(contentType string) ([]byte, error) {
	f.Lock.Lock()
	defer f.Lock.Unlock()

	if count, ok := f.FetchCounts[contentType]; ok {
		f.FetchCounts[contentType] = count + 1
	} else {
		f.FetchCounts[contentType] = 1
	}

	data, ok := f.Data[contentType]
	if !ok {
		return nil, errors.New("not found")
	}
	return data, nil
}

func (f *FakeOpenAPIV3Client) Paths() (map[string]openapi.GroupVersion, error) {
	if f.Values == nil {
		return nil, errors.New("values is nil")
	}

	res := map[string]openapi.GroupVersion{}
	if f.FetchCounts == nil {
		f.FetchCounts = map[string]map[string]int{}
	}

	for k, v := range f.Values {
		counts, ok := f.FetchCounts[k]
		if !ok {
			counts = map[string]int{}
			f.FetchCounts[k] = counts
		}
		res[k] = &FakeGroupVersion{Data: v, FetchCounts: counts, Lock: &f.lock}
	}
	return res, nil
}

func TestExplainErrors(t *testing.T) {
	var buf bytes.Buffer

	// A client with nil `Values` will return error on returning paths
	failFetchPaths := &FakeOpenAPIV3Client{}

	err := PrintModelDescription(nil, &buf, failFetchPaths, apiGroupsGVR, false, "unknown-format")
	require.ErrorContains(t, err, "failed to fetch list of groupVersions")

	// Missing Schema
	fakeClient := &FakeOpenAPIV3Client{
		Values: map[string]map[string][]byte{
			"apis/test1.example.com/v1": {
				"unknown/content-type": []byte(apiDiscoveryJSON),
			},
			"apis/test2.example.com/v1": {
				runtime.ContentTypeJSON: []byte(`<some invalid json!>`),
			},
			"apis/discovery.k8s.io/v1": {
				runtime.ContentTypeJSON: []byte(apiDiscoveryJSON),
			},
		},
	}

	err = PrintModelDescription(nil, &buf, fakeClient, schema.GroupVersionResource{
		Group:    "test0.example.com",
		Version:  "v1",
		Resource: "doesntmatter",
	}, false, "unknown-format")
	require.ErrorContains(t, err, "could not locate schema")

	// Missing JSON
	err = PrintModelDescription(nil, &buf, fakeClient, schema.GroupVersionResource{
		Group:    "test1.example.com",
		Version:  "v1",
		Resource: "doesntmatter",
	}, false, "unknown-format")
	require.ErrorContains(t, err, "failed to fetch openapi schema ")

	err = PrintModelDescription(nil, &buf, fakeClient, schema.GroupVersionResource{
		Group:    "test2.example.com",
		Version:  "v1",
		Resource: "doesntmatter",
	}, false, "unknown-format")
	require.ErrorContains(t, err, "failed to parse openapi schema")

	err = PrintModelDescription(nil, &buf, fakeClient, apiGroupsGVR, false, "unknown-format")
	require.ErrorContains(t, err, "unrecognized format: unknown-format")
}

// Shows that the correct GVR is fetched from the open api client when
// given to explain
func TestExplainOpenAPIClient(t *testing.T) {
	var buf bytes.Buffer

	fakeClient := &FakeOpenAPIV3Client{
		Values: map[string]map[string][]byte{
			"apis/discovery.k8s.io/v1": {
				runtime.ContentTypeJSON: []byte(apiDiscoveryJSON),
			},
		},
	}

	gen := NewGenerator()
	err := gen.AddTemplate("Context", "{{ toJson . }}")
	require.NoError(t, err)

	expectedContext := TemplateContext{
		Document:  apiGroupsDocument,
		GVR:       apiGroupsGVR,
		Recursive: false,
		FieldPath: nil,
	}

	err = printModelDescriptionWithGenerator(gen, nil, &buf, fakeClient, apiGroupsGVR, false, "Context")
	require.NoError(t, err)

	var actualContext TemplateContext
	err = json.Unmarshal(buf.Bytes(), &actualContext)
	require.NoError(t, err)
	require.Equal(t, expectedContext, actualContext)
	require.Equal(t, fakeClient.FetchCounts["apis/discovery.k8s.io/v1"][runtime.ContentTypeJSON], 1)
}
