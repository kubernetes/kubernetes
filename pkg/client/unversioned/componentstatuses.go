/*
Copyright 2015 The Kubernetes Authors.

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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
)

type ComponentStatusesInterface interface {
	ComponentStatuses() ComponentStatusInterface
}

// ComponentStatusInterface contains methods to retrieve ComponentStatus
type ComponentStatusInterface interface {
	List(opts api.ListOptions) (*api.ComponentStatusList, error)
	Get(name string) (*api.ComponentStatus, error)

	// TODO: It'd be nice to have watch support at some point
	//Watch(opts api.ListOptions) (watch.Interface, error)
}

// componentStatuses implements ComponentStatusesInterface
type componentStatuses struct {
	client *Client
}

func newComponentStatuses(c *Client) *componentStatuses {
	return &componentStatuses{c}
}

func (c *componentStatuses) List(opts api.ListOptions) (result *api.ComponentStatusList, err error) {
	result = &api.ComponentStatusList{}
	err = c.client.Get().
		Resource("componentStatuses").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)

	return result, err
}

func (c *componentStatuses) Get(name string) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Get().Resource("componentStatuses").Name(name).Do().Into(result)
	return
}
