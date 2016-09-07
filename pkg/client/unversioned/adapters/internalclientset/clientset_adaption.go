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

package internalclientset

import (
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedauthentication "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/unversioned"
	unversionedauthorization "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/unversioned"
	unversionedautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned"
	unversionedbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/unversioned"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	unversionedstorage "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/unversioned"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/unversioned"
)

// FromUnversionedClient adapts a unversioned.Client to a internalclientset.Clientset.
// This function is temporary. We will remove it when everyone has moved to using
// Clientset. New code should NOT use this function.
func FromUnversionedClient(c *unversioned.Client) *internalclientset.Clientset {
	var clientset internalclientset.Clientset
	if c != nil {
		clientset.CoreClient = unversionedcore.New(c.RESTClient)
	} else {
		clientset.CoreClient = unversionedcore.New(nil)
	}
	if c != nil && c.ExtensionsClient != nil {
		clientset.ExtensionsClient = unversionedextensions.New(c.ExtensionsClient.RESTClient)
	} else {
		clientset.ExtensionsClient = unversionedextensions.New(nil)
	}
	if c != nil && c.BatchClient != nil {
		clientset.BatchClient = unversionedbatch.New(c.BatchClient.RESTClient)
	} else {
		clientset.BatchClient = unversionedbatch.New(nil)
	}
	if c != nil && c.AuthorizationClient != nil {
		clientset.AuthorizationClient = unversionedauthorization.New(c.AuthorizationClient.RESTClient)
	} else {
		clientset.AuthorizationClient = unversionedauthorization.New(nil)
	}
	if c != nil && c.AutoscalingClient != nil {
		clientset.AutoscalingClient = unversionedautoscaling.New(c.AutoscalingClient.RESTClient)
	} else {
		clientset.AutoscalingClient = unversionedautoscaling.New(nil)
	}
	if c != nil && c.AuthenticationClient != nil {
		clientset.AuthenticationClient = unversionedauthentication.New(c.AuthenticationClient.RESTClient)
	} else {
		clientset.AuthenticationClient = unversionedauthentication.New(nil)
	}
	if c != nil && c.DiscoveryClient != nil {
		clientset.DiscoveryClient = discovery.NewDiscoveryClient(c.DiscoveryClient.RESTClient)
	} else {
		clientset.DiscoveryClient = discovery.NewDiscoveryClient(nil)
	}
	if c != nil && c.StorageClient != nil {
		clientset.StorageClient = unversionedstorage.New(c.StorageClient.RESTClient)
	} else {
		clientset.StorageClient = unversionedstorage.New(nil)
	}

	return &clientset
}
