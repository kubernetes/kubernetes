/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/client/restclient"
	// Import solely to initialize client auth plugins.
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

const (
	legacyAPIPath  = "/api"
	defaultAPIPath = "/apis"
)

// SetKubernetesDefaults sets default values on the provided client config for accessing the
// Kubernetes API or returns an error if any of the defaults are impossible or invalid.
// TODO: this method needs to be split into one that sets defaults per group, expected to be fix in PR "Refactoring clientcache.go and helper.go #14592"
func SetKubernetesDefaults(config *restclient.Config) error {
	if config.APIPath == "" {
		config.APIPath = legacyAPIPath
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != api.GroupName {
		g, err := api.Registry.Group(api.GroupName)
		if err != nil {
			return err
		}
		copyGroupVersion := g.GroupVersion
		config.GroupVersion = &copyGroupVersion
	}
	if config.NegotiatedSerializer == nil {
		config.NegotiatedSerializer = api.Codecs
	}
	return restclient.SetKubernetesDefaults(config)
}

func setGroupDefaults(groupName string, config *restclient.Config) error {
	config.APIPath = defaultAPIPath
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != groupName {
		g, err := api.Registry.Group(groupName)
		if err != nil {
			return err
		}
		copyGroupVersion := g.GroupVersion
		config.GroupVersion = &copyGroupVersion
	}
	if config.NegotiatedSerializer == nil {
		config.NegotiatedSerializer = api.Codecs
	}
	return nil
}
