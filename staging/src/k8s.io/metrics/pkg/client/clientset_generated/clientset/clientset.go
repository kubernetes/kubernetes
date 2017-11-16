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

package clientset

import (
	glog "github.com/golang/glog"
	discovery "k8s.io/client-go/discovery"
	rest "k8s.io/client-go/rest"
	flowcontrol "k8s.io/client-go/util/flowcontrol"
	metricsv1alpha1 "k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1alpha1"
	metricsv1beta1 "k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1beta1"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	MetricsV1alpha1() metricsv1alpha1.MetricsV1alpha1Interface
	MetricsV1beta1() metricsv1beta1.MetricsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Metrics() metricsv1beta1.MetricsV1beta1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	metricsV1alpha1 *metricsv1alpha1.MetricsV1alpha1Client
	metricsV1beta1  *metricsv1beta1.MetricsV1beta1Client
}

// MetricsV1alpha1 retrieves the MetricsV1alpha1Client
func (c *Clientset) MetricsV1alpha1() metricsv1alpha1.MetricsV1alpha1Interface {
	return c.metricsV1alpha1
}

// MetricsV1beta1 retrieves the MetricsV1beta1Client
func (c *Clientset) MetricsV1beta1() metricsv1beta1.MetricsV1beta1Interface {
	return c.metricsV1beta1
}

// Deprecated: Metrics retrieves the default version of MetricsClient.
// Please explicitly pick a version.
func (c *Clientset) Metrics() metricsv1beta1.MetricsV1beta1Interface {
	return c.metricsV1beta1
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
	cs.metricsV1alpha1, err = metricsv1alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.metricsV1beta1, err = metricsv1beta1.NewForConfig(&configShallowCopy)
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
	cs.metricsV1alpha1 = metricsv1alpha1.NewForConfigOrDie(c)
	cs.metricsV1beta1 = metricsv1beta1.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.metricsV1alpha1 = metricsv1alpha1.New(c)
	cs.metricsV1beta1 = metricsv1beta1.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
