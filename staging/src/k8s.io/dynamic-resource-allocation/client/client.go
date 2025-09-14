/*
Copyright 2025 The Kubernetes Authors.

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
	"sync/atomic"

	"k8s.io/client-go/kubernetes"
	cgoresource "k8s.io/client-go/kubernetes/typed/resource/v1"
	rest "k8s.io/client-go/rest"
)

// Enumerate all supported APIs, most preferred first.
const (
	useLatestAPI = int32(iota)
	useV1beta2API
	useV1beta1API
	numAPIs
)

func apiName(apiNumber int32) string {
	switch apiNumber {
	case useLatestAPI:
		return "V1"
	case useV1beta2API:
		return "V1beta2"
	case useV1beta1API:
		return "V1beta1"
	default:
		return "???"
	}
}

var (
	ErrNotImplemented = errors.New("not implemented in k8s.io/dynamic-resource-allocation/client")
)

func New(clientSet kubernetes.Interface) *Client {
	return &Client{
		clientSet: clientSet,
	}
}

type Client struct {
	clientSet kubernetes.Interface
	useAPI    atomic.Int32
}

var _ cgoresource.ResourceV1Interface = &Client{}

func (c *Client) CurrentAPI() string {
	return apiName(c.useAPI.Load())
}

func (c *Client) RESTClient() rest.Interface {
	return c.clientSet.ResourceV1().RESTClient()
}

func (c *Client) DeviceClasses() cgoresource.DeviceClassInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1().DeviceClasses(),
		c.clientSet.ResourceV1beta1().DeviceClasses(),
		c.clientSet.ResourceV1beta2().DeviceClasses(),
	)
}

func (c *Client) ResourceClaims(namespace string) cgoresource.ResourceClaimInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1().ResourceClaims(namespace),
		c.clientSet.ResourceV1beta1().ResourceClaims(namespace),
		c.clientSet.ResourceV1beta2().ResourceClaims(namespace),
	)
}

func (c *Client) ResourceClaimTemplates(namespace string) cgoresource.ResourceClaimTemplateInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1().ResourceClaimTemplates(namespace),
		c.clientSet.ResourceV1beta1().ResourceClaimTemplates(namespace),
		c.clientSet.ResourceV1beta2().ResourceClaimTemplates(namespace),
	)
}

func (c *Client) ResourceSlices() cgoresource.ResourceSliceInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1().ResourceSlices(),
		c.clientSet.ResourceV1beta1().ResourceSlices(),
		c.clientSet.ResourceV1beta2().ResourceSlices(),
	)
}
