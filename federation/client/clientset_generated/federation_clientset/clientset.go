/*
Copyright 2017 The Kubernetes Authors.

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

package federation_clientset

import (
	glog "github.com/golang/glog"
	discovery "k8s.io/client-go/discovery"
	rest "k8s.io/client-go/rest"
	flowcontrol "k8s.io/client-go/util/flowcontrol"
	autoscalingv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1"
	batchv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/batch/v1"
	corev1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/core/v1"
	extensionsv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1"
	federationv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	AutoscalingV1() autoscalingv1.AutoscalingV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Autoscaling() autoscalingv1.AutoscalingV1Interface
	BatchV1() batchv1.BatchV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Batch() batchv1.BatchV1Interface
	CoreV1() corev1.CoreV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Core() corev1.CoreV1Interface
	ExtensionsV1beta1() extensionsv1beta1.ExtensionsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() extensionsv1beta1.ExtensionsV1beta1Interface
	FederationV1beta1() federationv1beta1.FederationV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Federation() federationv1beta1.FederationV1beta1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	autoscalingV1     *autoscalingv1.AutoscalingV1Client
	batchV1           *batchv1.BatchV1Client
	coreV1            *corev1.CoreV1Client
	extensionsV1beta1 *extensionsv1beta1.ExtensionsV1beta1Client
	federationV1beta1 *federationv1beta1.FederationV1beta1Client
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() autoscalingv1.AutoscalingV1Interface {
	return c.autoscalingV1
}

// Deprecated: Autoscaling retrieves the default version of AutoscalingClient.
// Please explicitly pick a version.
func (c *Clientset) Autoscaling() autoscalingv1.AutoscalingV1Interface {
	return c.autoscalingV1
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() batchv1.BatchV1Interface {
	return c.batchV1
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() batchv1.BatchV1Interface {
	return c.batchV1
}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() corev1.CoreV1Interface {
	return c.coreV1
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() corev1.CoreV1Interface {
	return c.coreV1
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() extensionsv1beta1.ExtensionsV1beta1Interface {
	return c.extensionsV1beta1
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() extensionsv1beta1.ExtensionsV1beta1Interface {
	return c.extensionsV1beta1
}

// FederationV1beta1 retrieves the FederationV1beta1Client
func (c *Clientset) FederationV1beta1() federationv1beta1.FederationV1beta1Interface {
	return c.federationV1beta1
}

// Deprecated: Federation retrieves the default version of FederationClient.
// Please explicitly pick a version.
func (c *Clientset) Federation() federationv1beta1.FederationV1beta1Interface {
	return c.federationV1beta1
}

// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	if c == nil {
		return nil
	}
	return c.DiscoveryClient
}

// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *rest.Config) (*Clientset, error) {
	configShallowCopy := *c
	if configShallowCopy.RateLimiter == nil && configShallowCopy.QPS > 0 {
		configShallowCopy.RateLimiter = flowcontrol.NewTokenBucketRateLimiter(configShallowCopy.QPS, configShallowCopy.Burst)
	}
	var cs Clientset
	var err error
	cs.autoscalingV1, err = autoscalingv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.batchV1, err = batchv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.coreV1, err = corev1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.extensionsV1beta1, err = extensionsv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.federationV1beta1, err = federationv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}

	cs.DiscoveryClient, err = discovery.NewDiscoveryClientForConfig(&configShallowCopy)
	if err != nil {
		glog.Errorf("failed to create the DiscoveryClient: %v", err)
		return nil, err
	}
	return &cs, nil
}

// NewForConfigOrDie creates a new Clientset for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *rest.Config) *Clientset {
	var cs Clientset
	cs.autoscalingV1 = autoscalingv1.NewForConfigOrDie(c)
	cs.batchV1 = batchv1.NewForConfigOrDie(c)
	cs.coreV1 = corev1.NewForConfigOrDie(c)
	cs.extensionsV1beta1 = extensionsv1beta1.NewForConfigOrDie(c)
	cs.federationV1beta1 = federationv1beta1.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.autoscalingV1 = autoscalingv1.New(c)
	cs.batchV1 = batchv1.New(c)
	cs.coreV1 = corev1.New(c)
	cs.extensionsV1beta1 = extensionsv1beta1.New(c)
	cs.federationV1beta1 = federationv1beta1.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
