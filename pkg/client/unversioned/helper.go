/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"
	// Import solely to initialize client auth plugins.
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

const (
	legacyAPIPath  = "/api"
	defaultAPIPath = "/apis"
)

// New creates a Kubernetes client for the given config. This client works with pods,
// replication controllers, daemons, and services. It allows operations such as list, get, update
// and delete on these objects. An error is returned if the provided configuration
// is not valid.
func New(c *restclient.Config) (*Client, error) {
	config := *c
	if err := SetKubernetesDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}

	discoveryConfig := *c
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(&discoveryConfig)
	if err != nil {
		return nil, err
	}

	var autoscalingClient *AutoscalingClient
	if registered.IsRegistered(autoscaling.GroupName) {
		autoscalingConfig := *c
		autoscalingClient, err = NewAutoscaling(&autoscalingConfig)
		if err != nil {
			return nil, err
		}
	}

	var batchClient *BatchClient
	if registered.IsRegistered(batch.GroupName) {
		batchConfig := *c
		batchClient, err = NewBatch(&batchConfig)
		if err != nil {
			return nil, err
		}
	}

	var extensionsClient *ExtensionsClient
	if registered.IsRegistered(extensions.GroupName) {
		extensionsConfig := *c
		extensionsClient, err = NewExtensions(&extensionsConfig)
		if err != nil {
			return nil, err
		}
	}
	var appsClient *AppsClient
	if registered.IsRegistered(apps.GroupName) {
		appsConfig := *c
		appsClient, err = NewApps(&appsConfig)
		if err != nil {
			return nil, err
		}
	}

	return &Client{RESTClient: client, AutoscalingClient: autoscalingClient, BatchClient: batchClient, ExtensionsClient: extensionsClient, DiscoveryClient: discoveryClient, AppsClient: appsClient}, nil
}

// MatchesServerVersion queries the server to compares the build version
// (git hash) of the client with the server's build version. It returns an error
// if it failed to contact the server or if the versions are not an exact match.
func MatchesServerVersion(client *Client, c *restclient.Config) error {
	var err error
	if client == nil {
		client, err = New(c)
		if err != nil {
			return err
		}
	}
	cVer := version.Get()
	sVer, err := client.Discovery().ServerVersion()
	if err != nil {
		return fmt.Errorf("couldn't read version from server: %v\n", err)
	}
	// GitVersion includes GitCommit and GitTreeState, but best to be safe?
	if cVer.GitVersion != sVer.GitVersion || cVer.GitCommit != sVer.GitCommit || cVer.GitTreeState != cVer.GitTreeState {
		return fmt.Errorf("server version (%#v) differs from client version (%#v)!\n", sVer, cVer)
	}

	return nil
}

// NegotiateVersion queries the server's supported api versions to find
// a version that both client and server support.
// - If no version is provided, try registered client versions in order of
//   preference.
// - If version is provided, but not default config (explicitly requested via
//   commandline flag), and is unsupported by the server, print a warning to
//   stderr and try client's registered versions in order of preference.
// - If version is config default, and the server does not support it,
//   return an error.
func NegotiateVersion(client *Client, c *restclient.Config, requestedGV *unversioned.GroupVersion, clientRegisteredGVs []unversioned.GroupVersion) (*unversioned.GroupVersion, error) {
	var err error
	if client == nil {
		client, err = New(c)
		if err != nil {
			return nil, err
		}
	}
	clientVersions := sets.String{}
	for _, gv := range clientRegisteredGVs {
		clientVersions.Insert(gv.String())
	}
	groups, err := client.ServerGroups()
	if err != nil {
		// This is almost always a connection error, and higher level code should treat this as a generic error,
		// not a negotiation specific error.
		return nil, err
	}
	versions := unversioned.ExtractGroupVersions(groups)
	serverVersions := sets.String{}
	for _, v := range versions {
		serverVersions.Insert(v)
	}

	// If no version requested, use config version (may also be empty).
	// make a copy of the original so we don't risk mutating input here or in the returned value
	var preferredGV *unversioned.GroupVersion
	switch {
	case requestedGV != nil:
		t := *requestedGV
		preferredGV = &t
	case c.GroupVersion != nil:
		t := *c.GroupVersion
		preferredGV = &t
	}

	// If version explicitly requested verify that both client and server support it.
	// If server does not support warn, but try to negotiate a lower version.
	if preferredGV != nil {
		if !clientVersions.Has(preferredGV.String()) {
			return nil, fmt.Errorf("client does not support API version %q; client supported API versions: %v", preferredGV, clientVersions)

		}
		if serverVersions.Has(preferredGV.String()) {
			return preferredGV, nil
		}
		// If we are using an explicit config version the server does not support, fail.
		if (c.GroupVersion != nil) && (*preferredGV == *c.GroupVersion) {
			return nil, fmt.Errorf("server does not support API version %q", preferredGV)
		}
	}

	for _, clientGV := range clientRegisteredGVs {
		if serverVersions.Has(clientGV.String()) {
			// Version was not explicitly requested in command config (--api-version).
			// Ok to fall back to a supported version with a warning.
			// TODO: caesarxuchao: enable the warning message when we have
			// proper fix. Please refer to issue #14895.
			// if len(version) != 0 {
			// 	glog.Warningf("Server does not support API version '%s'. Falling back to '%s'.", version, clientVersion)
			// }
			t := clientGV
			return &t, nil
		}
	}
	return nil, fmt.Errorf("failed to negotiate an api version; server supports: %v, client supports: %v",
		serverVersions, clientVersions)
}

// NewOrDie creates a Kubernetes client and panics if the provided API version is not recognized.
func NewOrDie(c *restclient.Config) *Client {
	client, err := New(c)
	if err != nil {
		panic(err)
	}
	return client
}

// NewInCluster is a shortcut for calling InClusterConfig() and then New().
func NewInCluster() (*Client, error) {
	cc, err := restclient.InClusterConfig()
	if err != nil {
		return nil, err
	}
	return New(cc)
}

// SetKubernetesDefaults sets default values on the provided client config for accessing the
// Kubernetes API or returns an error if any of the defaults are impossible or invalid.
// TODO: this method needs to be split into one that sets defaults per group, expected to be fix in PR "Refactoring clientcache.go and helper.go #14592"
func SetKubernetesDefaults(config *restclient.Config) error {
	if config.APIPath == "" {
		config.APIPath = legacyAPIPath
	}
	g, err := registered.Group(api.GroupName)
	if err != nil {
		return err
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	if config.NegotiatedSerializer == nil {
		config.NegotiatedSerializer = api.Codecs
	}
	if config.Codec == nil {
		config.Codec = api.Codecs.LegacyCodec(*config.GroupVersion)
	}

	return restclient.SetKubernetesDefaults(config)
}
