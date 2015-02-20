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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
)

type OAuthClientsInterface interface {
	OAuthClients() OAuthClientInterface
}

// PodInterface has methods to work with Pod resources.
type OAuthClientInterface interface {
	List(selector labels.Selector) (*api.OAuthClientList, error)
	Get(name string) (*api.OAuthClient, error)
	Delete(name string) error
	Create(token *api.OAuthClient) (*api.OAuthClient, error)
	Update(token *api.OAuthClient) (*api.OAuthClient, error)
}

// pods implements PodsNamespacer interface
type Clients struct {
	r *Client
}

// newPods returns a pods
func newClients(c *Client) *Clients {
	return &Clients{
		r: c,
	}
}

// ListPods takes a selector, and returns the list of pods that match that selector.
func (c *Clients) List(selector labels.Selector) (*api.OAuthClientList, error) {
	result := &api.OAuthClientList{}
	err := c.r.Get().Resource(OAuthClientsPath).SelectorParam("labels", selector).Do().Into(result)
	return result, err
}

// GetPod takes the name of the pod, and returns the corresponding Pod object, and an error if it occurs
func (c *Clients) Get(name string) (*api.OAuthClient, error) {
	if len(name) == 0 {
		return nil, errors.New("name is required parameter to Get")
	}

	result := &api.OAuthClient{}
	err := c.r.Get().Resource(OAuthClientsPath).Name(name).Do().Into(result)
	return result, err
}

// DeletePod takes the name of the pod, and returns an error if one occurs
func (c *Clients) Delete(name string) error {
	return c.r.Delete().Resource(OAuthClientsPath).Name(name).Do().Error()
}

// CreatePod takes the representation of a pod.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *Clients) Create(oauthclient *api.OAuthClient) (*api.OAuthClient, error) {
	result := &api.OAuthClient{}
	err := c.r.Post().Resource(OAuthClientsPath).Body(oauthclient).Do().Into(result)
	return result, err
}

// UpdatePod takes the representation of a pod to update.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *Clients) Update(oauthclient *api.OAuthClient) (*api.OAuthClient, error) {
	result := &api.OAuthClient{}
	if len(oauthclient.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", oauthclient)
		return result, err
	}
	err := c.r.Put().Resource(OAuthClientsPath).Name(oauthclient.Name).Body(oauthclient).Do().Into(result)
	return result, err
}
