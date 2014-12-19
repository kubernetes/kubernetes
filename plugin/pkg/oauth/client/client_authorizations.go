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

type OAuthClientAuthorizationsInterface interface {
	OAuthClientAuthorizations() OAuthClientAuthorizationInterface
}

// PodInterface has methods to work with Pod resources.
type OAuthClientAuthorizationInterface interface {
	Name(username, clientname string) string
	List(selector labels.Selector) (*api.OAuthClientAuthorizationList, error)
	Get(name string) (*api.OAuthClientAuthorization, error)
	Delete(name string) error
	Create(token *api.OAuthClientAuthorization) (*api.OAuthClientAuthorization, error)
	Update(token *api.OAuthClientAuthorization) (*api.OAuthClientAuthorization, error)
}

// pods implements PodsNamespacer interface
type ClientAuthorizations struct {
	r *Client
}

// newPods returns a pods
func newClientAuthorizations(c *Client) *ClientAuthorizations {
	return &ClientAuthorizations{
		r: c,
	}
}

func (c *ClientAuthorizations) Name(username, clientname string) string {
	return username + ":" + clientname
}

// ListPods takes a selector, and returns the list of pods that match that selector.
func (c *ClientAuthorizations) List(selector labels.Selector) (*api.OAuthClientAuthorizationList, error) {
	result := &api.OAuthClientAuthorizationList{}
	err := c.r.Get().Resource(OAuthClientAuthorizationsPath).SelectorParam("labels", selector).Do().Into(result)
	return result, err
}

// GetPod takes the name of the pod, and returns the corresponding Pod object, and an error if it occurs
func (c *ClientAuthorizations) Get(name string) (*api.OAuthClientAuthorization, error) {
	if len(name) == 0 {
		return nil, errors.New("name is required parameter to Get")
	}

	result := &api.OAuthClientAuthorization{}
	err := c.r.Get().Resource(OAuthClientAuthorizationsPath).Name(name).Do().Into(result)
	return result, err
}

// DeletePod takes the name of the pod, and returns an error if one occurs
func (c *ClientAuthorizations) Delete(name string) error {
	return c.r.Delete().Resource(OAuthClientAuthorizationsPath).Name(name).Do().Error()
}

// CreatePod takes the representation of a pod.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *ClientAuthorizations) Create(auth *api.OAuthClientAuthorization) (*api.OAuthClientAuthorization, error) {
	result := &api.OAuthClientAuthorization{}
	err := c.r.Post().Resource(OAuthClientAuthorizationsPath).Body(auth).Do().Into(result)
	return result, err
}

// UpdatePod takes the representation of a pod to update.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *ClientAuthorizations) Update(auth *api.OAuthClientAuthorization) (*api.OAuthClientAuthorization, error) {
	result := &api.OAuthClientAuthorization{}
	if len(auth.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", auth)
		return result, err
	}
	err := c.r.Put().Resource(OAuthClientAuthorizationsPath).Name(auth.Name).Body(auth).Do().Into(result)
	return result, err
}
