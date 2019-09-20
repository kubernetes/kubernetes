/*
Copyright 2019 The Kubernetes Authors.

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

package client

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"time"

	"k8s.io/apimachinery/pkg/version"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
)

const defaultRequestTimeout time.Duration = 5 * time.Second
const defaultCacheMaxAge = 60 * 60 // One hour in seconds

// Encapsulates the client which fetches the server version. Implements
// the discovery.ServerVersionInterface, allowing the creation of a
// mock or fake for testing.
type ServerVersionClient struct {
	flags          *genericclioptions.ConfigFlags
	delegate       restclient.Interface
	requestTimeout time.Duration // Query timeout duration
	cacheMaxAge    uint64        // Maximum cache age allowed in seconds
}

var _ discovery.ServerVersionInterface = &ServerVersionClient{}

func NewServerVersionClient(kubeConfigFlags *genericclioptions.ConfigFlags) *ServerVersionClient {
	return &ServerVersionClient{
		flags:          kubeConfigFlags,
		delegate:       nil,
		requestTimeout: defaultRequestTimeout,
		cacheMaxAge:    defaultCacheMaxAge,
	}
}

func (c *ServerVersionClient) GetRequestTimeout() time.Duration {
	return c.requestTimeout
}

func (c *ServerVersionClient) SetRequestTimeout(requestTimeout string) error {
	timeout, err := time.ParseDuration(requestTimeout)
	if err != nil {
		return err
	}
	c.requestTimeout = timeout
	return nil
}

func (c *ServerVersionClient) GetCacheMaxAge() uint64 {
	return c.cacheMaxAge
}

func (c *ServerVersionClient) SetCacheMaxAge(cacheMaxAge uint64) {
	c.cacheMaxAge = cacheMaxAge
}

func (c *ServerVersionClient) ServerVersion() (*version.Info, error) {
	request, err := c.createRequest()
	if err != nil {
		return nil, err
	}
	body, err := request.DoRaw(context.TODO())
	if err != nil {
		return nil, err
	}
	var info version.Info
	err = json.Unmarshal(body, &info)
	if err != nil {
		return nil, fmt.Errorf("got '%s': %v", string(body), err)
	}
	return &info, nil
}

const (
	userAgentHeader    = "User-Agent"
	cacheControlHeader = "Cache-Control"
	serverVersionPath  = "/version"
)

func (c *ServerVersionClient) createRequest() (*restclient.Request, error) {
	if c.delegate == nil {
		discoveryClient, err := c.flags.ToDiscoveryClient()
		if err != nil {
			return nil, err
		}
		c.delegate = discoveryClient.RESTClient()
	}
	request := c.delegate.Get()
	request.SetHeader(userAgentHeader, c.getUserAgent())
	request.SetHeader(cacheControlHeader, fmt.Sprintf("max-age=%d", c.GetCacheMaxAge()))
	request.Timeout(c.GetRequestTimeout())
	request.AbsPath(serverVersionPath)
	return request, nil
}

const (
	dispatcherVersion   = "1.0"
	dispatcherUserAgent = "kubectl-dispatcher/v%s (%s/%s)"
)

func (c *ServerVersionClient) getUserAgent() string {
	os := runtime.GOOS
	arch := runtime.GOARCH
	return fmt.Sprintf(dispatcherUserAgent, dispatcherVersion, os, arch)
}
