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

type OAuthAuthorizeTokensInterface interface {
	OAuthAuthorizeTokens() OAuthAuthorizeTokenInterface
}

// PodInterface has methods to work with Pod resources.
type OAuthAuthorizeTokenInterface interface {
	List(selector labels.Selector) (*api.OAuthAuthorizeTokenList, error)
	Get(name string) (*api.OAuthAuthorizeToken, error)
	Delete(name string) error
	Create(token *api.OAuthAuthorizeToken) (*api.OAuthAuthorizeToken, error)
	Update(token *api.OAuthAuthorizeToken) (*api.OAuthAuthorizeToken, error)
}

// pods implements PodsNamespacer interface
type AuthorizeTokens struct {
	r *Client
}

// newPods returns a pods
func newAuthorizeTokens(c *Client) *AuthorizeTokens {
	return &AuthorizeTokens{
		r: c,
	}
}

// ListPods takes a selector, and returns the list of pods that match that selector.
func (c *AuthorizeTokens) List(selector labels.Selector) (*api.OAuthAuthorizeTokenList, error) {
	result := &api.OAuthAuthorizeTokenList{}
	err := c.r.Get().Resource(OAuthAuthorizeTokensPath).SelectorParam("labels", selector).Do().Into(result)
	return result, err
}

// GetPod takes the name of the pod, and returns the corresponding Pod object, and an error if it occurs
func (c *AuthorizeTokens) Get(name string) (*api.OAuthAuthorizeToken, error) {
	if len(name) == 0 {
		return nil, errors.New("name is required parameter to Get")
	}

	result := &api.OAuthAuthorizeToken{}
	err := c.r.Get().Resource(OAuthAuthorizeTokensPath).Name(name).Do().Into(result)
	return result, err
}

// DeletePod takes the name of the pod, and returns an error if one occurs
func (c *AuthorizeTokens) Delete(name string) error {
	return c.r.Delete().Resource(OAuthAuthorizeTokensPath).Name(name).Do().Error()
}

// CreatePod takes the representation of a pod.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *AuthorizeTokens) Create(token *api.OAuthAuthorizeToken) (*api.OAuthAuthorizeToken, error) {
	result := &api.OAuthAuthorizeToken{}
	err := c.r.Post().Resource(OAuthAuthorizeTokensPath).Body(token).Do().Into(result)
	return result, err
}

// UpdatePod takes the representation of a pod to update.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *AuthorizeTokens) Update(token *api.OAuthAuthorizeToken) (*api.OAuthAuthorizeToken, error) {
	result := &api.OAuthAuthorizeToken{}
	if len(token.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", token)
		return result, err
	}
	err := c.r.Put().Resource(OAuthAuthorizeTokensPath).Name(token.Name).Body(token).Do().Into(result)
	return result, err
}
