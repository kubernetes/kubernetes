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

type OAuthAccessTokensInterface interface {
	OAuthAccessTokens() OAuthAccessTokenInterface
}

// PodInterface has methods to work with Pod resources.
type OAuthAccessTokenInterface interface {
	List(selector labels.Selector) (*api.OAuthAccessTokenList, error)
	Get(name string) (*api.OAuthAccessToken, error)
	Delete(name string) error
	Create(token *api.OAuthAccessToken) (*api.OAuthAccessToken, error)
	Update(token *api.OAuthAccessToken) (*api.OAuthAccessToken, error)
}

// pods implements PodsNamespacer interface
type accessTokens struct {
	r *Client
}

// newPods returns a pods
func newAccessTokens(c *Client) *accessTokens {
	return &accessTokens{
		r: c,
	}
}

// ListPods takes a selector, and returns the list of pods that match that selector.
func (c *accessTokens) List(selector labels.Selector) (*api.OAuthAccessTokenList, error) {
	result := &api.OAuthAccessTokenList{}
	err := c.r.Get().Resource(OAuthAccessTokensPath).SelectorParam("labels", selector).Do().Into(result)
	return result, err
}

// GetPod takes the name of the pod, and returns the corresponding Pod object, and an error if it occurs
func (c *accessTokens) Get(name string) (*api.OAuthAccessToken, error) {
	if len(name) == 0 {
		return nil, errors.New("name is required parameter to Get")
	}

	result := &api.OAuthAccessToken{}
	err := c.r.Get().Resource(OAuthAccessTokensPath).Name(name).Do().Into(result)
	return result, err
}

// DeletePod takes the name of the pod, and returns an error if one occurs
func (c *accessTokens) Delete(name string) error {
	return c.r.Delete().Resource(OAuthAccessTokensPath).Name(name).Do().Error()
}

// CreatePod takes the representation of a pod.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *accessTokens) Create(token *api.OAuthAccessToken) (*api.OAuthAccessToken, error) {
	result := &api.OAuthAccessToken{}
	err := c.r.Post().Resource(OAuthAccessTokensPath).Body(token).Do().Into(result)
	return result, err
}

// UpdatePod takes the representation of a pod to update.  Returns the server's representation of the pod, and an error, if it occurs.
func (c *accessTokens) Update(token *api.OAuthAccessToken) (*api.OAuthAccessToken, error) {
	result := &api.OAuthAccessToken{}
	if len(token.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", token)
		return result, err
	}
	err := c.r.Put().Resource(OAuthAccessTokensPath).Name(token.Name).Body(token).Do().Into(result)
	return result, err
}
