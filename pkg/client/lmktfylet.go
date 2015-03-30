/*
Copyright 2014 Google Inc. All rights reserved.

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
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strconv"

	"github.com/GoogleCloudPlatform/lmktfy/pkg/api"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/api/latest"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/probe"
	httprobe "github.com/GoogleCloudPlatform/lmktfy/pkg/probe/http"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/runtime"
)

// ErrPodInfoNotAvailable may be returned when the requested pod info is not available.
var ErrPodInfoNotAvailable = errors.New("no pod info available")

// LMKTFYletClient is an interface for all lmktfylet functionality
type LMKTFYletClient interface {
	LMKTFYletHealthChecker
	PodInfoGetter
	NodeInfoGetter
	ConnectionInfoGetter
}

// LMKTFYletHealthchecker is an interface for healthchecking lmktfylets
type LMKTFYletHealthChecker interface {
	HealthCheck(host string) (probe.Result, error)
}

// PodInfoGetter is an interface for things that can get information about a pod's containers.
// Injectable for easy testing.
type PodInfoGetter interface {
	// GetPodStatus returns information about all containers which are part
	// Returns an api.PodStatus, or an error if one occurs.
	GetPodStatus(host, podNamespace, podID string) (api.PodStatusResult, error)
}

type NodeInfoGetter interface {
	GetNodeInfo(host string) (api.NodeInfo, error)
}

type ConnectionInfoGetter interface {
	GetConnectionInfo(host string) (scheme string, port uint, transport http.RoundTripper, error error)
}

// HTTPLMKTFYletClient is the default implementation of PodInfoGetter and LMKTFYletHealthchecker, accesses the lmktfylet over HTTP.
type HTTPLMKTFYletClient struct {
	Client      *http.Client
	Port        uint
	EnableHttps bool
}

// TODO: this structure is questionable, it should be using client.Config and overriding defaults.
func NewLMKTFYletClient(config *LMKTFYletConfig) (LMKTFYletClient, error) {
	transport := http.DefaultTransport

	tlsConfig, err := TLSConfigFor(&Config{
		TLSClientConfig: config.TLSClientConfig,
	})
	if err != nil {
		return nil, err
	}
	if tlsConfig != nil {
		transport = &http.Transport{
			TLSClientConfig: tlsConfig,
		}
	}

	c := &http.Client{
		Transport: transport,
		Timeout:   config.HTTPTimeout,
	}
	return &HTTPLMKTFYletClient{
		Client:      c,
		Port:        config.Port,
		EnableHttps: config.EnableHttps,
	}, nil
}

func (c *HTTPLMKTFYletClient) GetConnectionInfo(host string) (string, uint, http.RoundTripper, error) {
	scheme := "http"
	if c.EnableHttps {
		scheme = "https"
	}
	return scheme, c.Port, c.Client.Transport, nil
}

func (c *HTTPLMKTFYletClient) url(host, path, query string) string {
	scheme := "http"
	if c.EnableHttps {
		scheme = "https"
	}

	return (&url.URL{
		Scheme:   scheme,
		Host:     net.JoinHostPort(host, strconv.FormatUint(uint64(c.Port), 10)),
		Path:     path,
		RawQuery: query,
	}).String()
}

// GetPodInfo gets information about the specified pod.
func (c *HTTPLMKTFYletClient) GetPodStatus(host, podNamespace, podID string) (api.PodStatusResult, error) {
	status := api.PodStatusResult{}
	query := url.Values{"podID": {podID}, "podNamespace": {podNamespace}}
	response, err := c.getEntity(host, "/api/v1beta1/podInfo", query.Encode(), &status)
	if response != nil && response.StatusCode == http.StatusNotFound {
		return status, ErrPodInfoNotAvailable
	}
	return status, err
}

// GetNodeInfo gets information about the specified node.
func (c *HTTPLMKTFYletClient) GetNodeInfo(host string) (api.NodeInfo, error) {
	info := api.NodeInfo{}
	_, err := c.getEntity(host, "/api/v1beta1/nodeInfo", "", &info)
	return info, err
}

// getEntity might return a nil response.
func (c *HTTPLMKTFYletClient) getEntity(host, path, query string, entity runtime.Object) (*http.Response, error) {
	request, err := http.NewRequest("GET", c.url(host, path, query), nil)
	if err != nil {
		return nil, err
	}
	response, err := c.Client.Do(request)
	if err != nil {
		return response, err
	}
	defer response.Body.Close()
	if response.StatusCode >= 300 || response.StatusCode < 200 {
		return response, fmt.Errorf("lmktfylet %q server responded with HTTP error code %d", host, response.StatusCode)
	}
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return response, err
	}
	err = latest.Codec.DecodeInto(body, entity)
	return response, err
}

func (c *HTTPLMKTFYletClient) HealthCheck(host string) (probe.Result, error) {
	return httprobe.DoHTTPProbe(c.url(host, "/healthz", ""), c.Client)
}

// FakeLMKTFYletClient is a fake implementation of LMKTFYletClient which returns an error
// when called.  It is useful to pass to the master in a test configuration with
// no lmktfylets.
type FakeLMKTFYletClient struct{}

// GetPodInfo is a fake implementation of PodInfoGetter.GetPodInfo.
func (c FakeLMKTFYletClient) GetPodStatus(host, podNamespace string, podID string) (api.PodStatusResult, error) {
	return api.PodStatusResult{}, errors.New("Not Implemented")
}

// GetNodeInfo is a fake implementation of PodInfoGetter.GetNodeInfo
func (c FakeLMKTFYletClient) GetNodeInfo(host string) (api.NodeInfo, error) {
	return api.NodeInfo{}, errors.New("Not Implemented")
}

func (c FakeLMKTFYletClient) HealthCheck(host string) (probe.Result, error) {
	return probe.Unknown, errors.New("Not Implemented")
}

func (c FakeLMKTFYletClient) GetConnectionInfo(host string) (string, uint, http.RoundTripper, error) {
	return "", 0, nil, errors.New("Not Implemented")
}
