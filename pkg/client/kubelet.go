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
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
)

// ErrPodInfoNotAvailable may be returned when the requested pod info is not available.
var ErrPodInfoNotAvailable = errors.New("no pod info available")

// KubeletClient is an interface for all kubelet functionality
type KubeletClient interface {
	KubeletHealthChecker
	PodInfoGetter
}

// KubeletHealthchecker is an interface for healthchecking kubelets
type KubeletHealthChecker interface {
	HealthCheck(host string) (health.Status, error)
}

// PodInfoGetter is an interface for things that can get information about a pod's containers.
// Injectable for easy testing.
type PodInfoGetter interface {
	// GetPodInfo returns information about all containers which are part
	// Returns an api.PodInfo, or an error if one occurs.
	GetPodInfo(host, podNamespace, podID string) (api.PodInfo, error)
}

// HTTPKubeletClient is the default implementation of PodInfoGetter and KubeletHealthchecker, accesses the kubelet over HTTP.
type HTTPKubeletClient struct {
	Client      *http.Client
	Port        uint
	EnableHttps bool
}

func NewKubeletClient(config *KubeletConfig) (KubeletClient, error) {
	transport := http.DefaultTransport
	if config.CAFile != "" {
		t, err := NewClientCertTLSTransport(config.CertFile, config.KeyFile, config.CAFile)
		if err != nil {
			return nil, err
		}
		transport = t
	}

	c := &http.Client{Transport: transport}
	return &HTTPKubeletClient{
		Client:      c,
		Port:        config.Port,
		EnableHttps: config.EnableHttps,
	}, nil
}

func (c *HTTPKubeletClient) url(host string) string {
	scheme := "http://"
	if c.EnableHttps {
		scheme = "https://"
	}

	return fmt.Sprintf(
		"%s%s",
		scheme,
		net.JoinHostPort(host, strconv.FormatUint(uint64(c.Port), 10)))
}

// GetPodInfo gets information about the specified pod.
func (c *HTTPKubeletClient) GetPodInfo(host, podNamespace, podID string) (api.PodInfo, error) {
	request, err := http.NewRequest(
		"GET",
		fmt.Sprintf(
			"%s/podInfo?podID=%s&podNamespace=%s",
			c.url(host),
			podID,
			podNamespace),
		nil)
	if err != nil {
		return nil, err
	}
	response, err := c.Client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return nil, ErrPodInfoNotAvailable
	}
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	// Check that this data can be unmarshalled
	info := api.PodInfo{}
	err = json.Unmarshal(body, &info)
	if err != nil {
		return nil, err
	}
	return info, nil
}

func (c *HTTPKubeletClient) HealthCheck(host string) (health.Status, error) {
	return health.DoHTTPCheck(fmt.Sprintf("%s/healthz", c.url(host)), c.Client)
}

// FakeKubeletClient is a fake implementation of KubeletClient which returns an error
// when called.  It is useful to pass to the master in a test configuration with
// no kubelets.
type FakeKubeletClient struct{}

// GetPodInfo is a fake implementation of PodInfoGetter.GetPodInfo.
func (c FakeKubeletClient) GetPodInfo(host, podNamespace string, podID string) (api.PodInfo, error) {
	return api.PodInfo{}, errors.New("Not Implemented")
}

func (c FakeKubeletClient) HealthCheck(host string) (health.Status, error) {
	return health.Unknown, errors.New("Not Implemented")
}
