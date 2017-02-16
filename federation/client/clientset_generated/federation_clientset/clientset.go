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
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	discovery "k8s.io/client-go/discovery"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	rest "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/scheme"
	clientautoscalingv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1"
	clientbatchv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/batch/v1"
	clientcorev1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/core/v1"
	clientextensionsv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1"
	clientfederationv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	CoreV1() clientcorev1.CoreV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Core() clientcorev1.CoreV1Interface
	AutoscalingV1() clientautoscalingv1.AutoscalingV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Autoscaling() clientautoscalingv1.AutoscalingV1Interface
	BatchV1() clientbatchv1.BatchV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Batch() clientbatchv1.BatchV1Interface
	ExtensionsV1beta1() clientextensionsv1beta1.ExtensionsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() clientextensionsv1beta1.ExtensionsV1beta1Interface
	FederationV1beta1() clientfederationv1beta1.FederationV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Federation() clientfederationv1beta1.FederationV1beta1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*clientcorev1.CoreV1Client
	*clientautoscalingv1.AutoscalingV1Client
	*clientbatchv1.BatchV1Client
	*clientextensionsv1beta1.ExtensionsV1beta1Client
	*clientfederationv1beta1.FederationV1beta1Client
}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() clientcorev1.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() clientcorev1.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() clientautoscalingv1.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// Deprecated: Autoscaling retrieves the default version of AutoscalingClient.
// Please explicitly pick a version.
func (c *Clientset) Autoscaling() clientautoscalingv1.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() clientbatchv1.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() clientbatchv1.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// FederationV1beta1 retrieves the FederationV1beta1Client
func (c *Clientset) FederationV1beta1() clientfederationv1beta1.FederationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.FederationV1beta1Client
}

// Deprecated: Federation retrieves the default version of FederationClient.
// Please explicitly pick a version.
func (c *Clientset) Federation() clientfederationv1beta1.FederationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.FederationV1beta1Client
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
	if configShallowCopy.ParameterCodec == nil {
		configShallowCopy.ParameterCodec = scheme.ParameterCodec
	}
	if configShallowCopy.NegotiatedSerializer == nil {
		configShallowCopy.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: scheme.Codecs}
	}
	var cs Clientset
	var err error
	cs.CoreV1Client, err = clientcorev1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingV1Client, err = clientautoscalingv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV1Client, err = clientbatchv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsV1beta1Client, err = clientextensionsv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.FederationV1beta1Client, err = clientfederationv1beta1.NewForConfig(&configShallowCopy)
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
	cs.CoreV1Client = clientcorev1.NewForConfigOrDie(c)
	cs.AutoscalingV1Client = clientautoscalingv1.NewForConfigOrDie(c)
	cs.BatchV1Client = clientbatchv1.NewForConfigOrDie(c)
	cs.ExtensionsV1beta1Client = clientextensionsv1beta1.NewForConfigOrDie(c)
	cs.FederationV1beta1Client = clientfederationv1beta1.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.CoreV1Client = clientcorev1.New(c)
	cs.AutoscalingV1Client = clientautoscalingv1.New(c)
	cs.BatchV1Client = clientbatchv1.New(c)
	cs.ExtensionsV1beta1Client = clientextensionsv1beta1.New(c)
	cs.FederationV1beta1Client = clientfederationv1beta1.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
