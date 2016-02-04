/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package release_1_2

import (
	"github.com/golang/glog"
	extensions_unversioned "k8s.io/kubernetes/pkg/client/typed/generated/extensions/unversioned"
	legacy_unversioned "k8s.io/kubernetes/pkg/client/typed/generated/legacy/unversioned"
	unversioned "k8s.io/kubernetes/pkg/client/unversioned"
)

type Interface interface {
	Discovery() unversioned.DiscoveryInterface
	Legacy() legacy_unversioned.LegacyInterface
	Extensions() extensions_unversioned.ExtensionsInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*unversioned.DiscoveryClient
	*legacy_unversioned.LegacyClient
	*extensions_unversioned.ExtensionsClient
}

// Legacy retrieves the LegacyClient
func (c *Clientset) Legacy() legacy_unversioned.LegacyInterface {
	return c.LegacyClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() extensions_unversioned.ExtensionsInterface {
	return c.ExtensionsClient
}

// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() unversioned.DiscoveryInterface {
	return c.DiscoveryClient
}

// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *unversioned.Config) (*Clientset, error) {
	var clientset Clientset
	var err error
	clientset.LegacyClient, err = legacy_unversioned.NewForConfig(c)
	if err != nil {
		return &clientset, err
	}
	clientset.ExtensionsClient, err = extensions_unversioned.NewForConfig(c)
	if err != nil {
		return &clientset, err
	}

	clientset.DiscoveryClient, err = unversioned.NewDiscoveryClientForConfig(c)
	if err != nil {
		glog.Errorf("failed to create the DiscoveryClient: %v", err)
	}
	return &clientset, err
}

// NewForConfigOrDie creates a new Clientset for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *unversioned.Config) *Clientset {
	var clientset Clientset
	clientset.LegacyClient = legacy_unversioned.NewForConfigOrDie(c)
	clientset.ExtensionsClient = extensions_unversioned.NewForConfigOrDie(c)

	clientset.DiscoveryClient = unversioned.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c *unversioned.RESTClient) *Clientset {
	var clientset Clientset
	clientset.LegacyClient = legacy_unversioned.New(c)
	clientset.ExtensionsClient = extensions_unversioned.New(c)

	clientset.DiscoveryClient = unversioned.NewDiscoveryClient(c)
	return &clientset
}
