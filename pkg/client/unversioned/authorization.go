/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/authorization"
	"k8s.io/kubernetes/pkg/client/restclient"
)

type AuthorizationInterface interface {
	SubjectAccessReviewsInterface
}

// AuthorizationClient is used to interact with Kubernetes authorization features.
type AuthorizationClient struct {
	*restclient.RESTClient
}

func (c *AuthorizationClient) SubjectAccessReviews() SubjectAccessReviewInterface {
	return newSubjectAccessReviews(c)
}

func NewAuthorization(c *restclient.Config) (*AuthorizationClient, error) {
	config := *c
	if err := setAuthorizationDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &AuthorizationClient{client}, nil
}

func NewAuthorizationOrDie(c *restclient.Config) *AuthorizationClient {
	client, err := NewAuthorization(c)
	if err != nil {
		panic(err)
	}
	return client
}

func setAuthorizationDefaults(config *restclient.Config) error {
	// if authorization group is not registered, return an error
	g, err := registered.Group(authorization.GroupName)
	if err != nil {
		return err
	}
	config.APIPath = defaultAPIPath
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

	config.NegotiatedSerializer = api.Codecs
	return nil
}
