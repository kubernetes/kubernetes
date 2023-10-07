/*
Copyright 2017 The Kubernetes Authors.

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

package openapi

import (
	"context"
	"encoding/json"
	"strings"

	"k8s.io/client-go/rest"
	"k8s.io/kube-openapi/pkg/handler3"
)

type Client interface {
	Paths() (map[string]GroupVersion, error)
}

type client struct {
	// URL includes the `hash` query param to take advantage of cache busting
	restClient rest.Interface
}

func NewClient(restClient rest.Interface) Client {
	return &client{
		restClient: restClient,
	}
}

func (c *client) Paths() (map[string]GroupVersion, error) {
	data, err := c.restClient.Get().
		AbsPath("/openapi/v3").
		Do(context.TODO()).
		Raw()

	if err != nil {
		return nil, err
	}

	discoMap := &handler3.OpenAPIV3Discovery{}
	err = json.Unmarshal(data, discoMap)
	if err != nil {
		return nil, err
	}

	// Create GroupVersions for each element of the result
	result := map[string]GroupVersion{}
	for k, v := range discoMap.Paths {
		// If the server returned a URL rooted at /openapi/v3, preserve any additional client-side prefix.
		// If the server returned a URL not rooted at /openapi/v3, treat it as an actual server-relative URL.
		// See https://github.com/kubernetes/kubernetes/issues/117463 for details
		useClientPrefix := strings.HasPrefix(v.ServerRelativeURL, "/openapi/v3")
		result[k] = newGroupVersion(c, v, useClientPrefix)
	}
	return result, nil
}
