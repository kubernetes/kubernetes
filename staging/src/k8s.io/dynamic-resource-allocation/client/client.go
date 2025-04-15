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
	cgoresource "k8s.io/client-go/kubernetes/typed/resource/v1beta2"
	rest "k8s.io/client-go/rest"
)

const (
	useLatestAPI = int32(iota)
	useV1beta1API
	numAPIs
)

var (
	ErrNotImplemented = errors.New("not implemented in k8s.io/dynamic-resource-allocation/client")
)

func New(clientSet kubernetes.Interface) cgoresource.ResourceV1beta2Interface {
	return &client{
		clientSet: clientSet,
	}
}

type client struct {
	clientSet kubernetes.Interface
	useAPI    atomic.Int32
}

var _ cgoresource.ResourceV1beta2Interface = &client{}

func (c *client) RESTClient() rest.Interface {
	return c.clientSet.ResourceV1beta2().RESTClient()
}

func (c *client) DeviceClasses() cgoresource.DeviceClassInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1beta2().DeviceClasses(),
		c.clientSet.ResourceV1beta1().DeviceClasses(),
	)
}

func (c *client) ResourceClaims(namespace string) cgoresource.ResourceClaimInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1beta2().ResourceClaims(namespace),
		c.clientSet.ResourceV1beta1().ResourceClaims(namespace),
	)
}

func (c *client) ResourceClaimTemplates(namespace string) cgoresource.ResourceClaimTemplateInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1beta2().ResourceClaimTemplates(namespace),
		c.clientSet.ResourceV1beta1().ResourceClaimTemplates(namespace),
	)
}

func (c *client) ResourceSlices() cgoresource.ResourceSliceInterface {
	return newConvertingClient(c,
		c.clientSet.ResourceV1beta2().ResourceSlices(),
		c.clientSet.ResourceV1beta1().ResourceSlices(),
	)
}
