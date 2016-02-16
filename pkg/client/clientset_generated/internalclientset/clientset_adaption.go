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

package internalclientset

import (
	core_unversioned "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"
	extensions_unversioned "k8s.io/kubernetes/pkg/client/typed/generated/extensions/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned"
)

// FromUnversionedClient adapts a pkg/client/unversioned#Client to a Clientset.
// This function is temporary. We will remove it when everyone has moved to using
// Clientset. New code should NOT use this function.
func FromUnversionedClient(c *unversioned.Client) *Clientset {
	var clientset Clientset
	if c != nil {
		clientset.CoreClient = core_unversioned.New(c.RESTClient)
	} else {
		clientset.CoreClient = core_unversioned.New(nil)
	}
	if c != nil && c.ExtensionsClient != nil {
		clientset.ExtensionsClient = extensions_unversioned.New(c.ExtensionsClient.RESTClient)
	} else {
		clientset.ExtensionsClient = extensions_unversioned.New(nil)
	}

	if c != nil && c.DiscoveryClient != nil {
		clientset.DiscoveryClient = unversioned.NewDiscoveryClient(c.DiscoveryClient.RESTClient)
	} else {
		clientset.DiscoveryClient = unversioned.NewDiscoveryClient(nil)
	}

	return &clientset
}
