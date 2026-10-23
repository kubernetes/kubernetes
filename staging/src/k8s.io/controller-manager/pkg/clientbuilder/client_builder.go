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

package clientbuilder

import (
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/scale"
	"k8s.io/klog/v2"
)

// ControllerClientBuilder allows you to get clients for controllers.
// Please note a subset copy also exists in staging/src/k8s.io/cloud-provider/cloud.go
// TODO: Extract this into a separate controller utilities repo (issues/68947)
type ControllerClientBuilder interface {
	// Config returns a client config for the named controller.
	//
	// Deprecated: a raw config lets a controller build a client that bypasses
	// the transport wrappers the controller manager applies to the clients it
	// builds, such as the leader election write gate. Build clients through
	// the builder instead; TypedClientBuilder covers the clients controllers
	// commonly need. Config and ConfigOrDie will be removed following the
	// deprecation policy.
	Config(name string) (*restclient.Config, error)
	// ConfigOrDie is Config, exiting the process on error.
	//
	// Deprecated: see Config.
	ConfigOrDie(name string) *restclient.Config
	Client(name string) (clientset.Interface, error)
	ClientOrDie(name string) clientset.Interface
	DiscoveryClient(name string) (discovery.DiscoveryInterface, error)
	DiscoveryClientOrDie(name string) discovery.DiscoveryInterface
}

// TypedClientBuilder is a ControllerClientBuilder that also constructs the
// other clients controllers commonly need, so that controllers never handle a
// rest.Config themselves. Both builders in this package implement it and the
// controller managers hand it to their controllers.
type TypedClientBuilder interface {
	ControllerClientBuilder
	MetadataClient(name string) (metadata.Interface, error)
	DynamicClient(name string) (dynamic.Interface, error)
	ScaleClient(name string, mapper scale.PreferredResourceMapper, resolver dynamic.APIPathResolverFunc, scaleKindResolver scale.ScaleKindResolver) (scale.ScalesGetter, error)
	// WithRateLimits returns a builder whose clients use the given QPS and
	// burst instead of the configured defaults.
	WithRateLimits(qps float32, burst int) TypedClientBuilder
	// WithDropManagedFields returns a builder whose clients ask the server to
	// omit metadata.managedFields from responses.
	WithDropManagedFields() TypedClientBuilder
}

// SimpleControllerClientBuilder returns a fixed client with different user agents
type SimpleControllerClientBuilder struct {
	// ClientConfig is a skeleton config to clone and use as the basis for each controller client
	ClientConfig *restclient.Config
}

// Config returns a client config for a fixed client
//
// Deprecated: see ControllerClientBuilder.Config.
func (b SimpleControllerClientBuilder) Config(name string) (*restclient.Config, error) {
	clientConfig := *b.ClientConfig
	return restclient.AddUserAgent(&clientConfig, name), nil
}

// ConfigOrDie returns a client config if no error from previous config func.
// If it gets an error getting the client, it will log the error and kill the process it's running in.
//
// Deprecated: see ControllerClientBuilder.Config.
func (b SimpleControllerClientBuilder) ConfigOrDie(name string) *restclient.Config {
	clientConfig, err := b.Config(name)
	if err != nil {
		klog.Fatal(err)
	}
	return clientConfig
}

// WithRateLimits returns a TypedClientBuilder whose clients use the given QPS and burst.
func (b SimpleControllerClientBuilder) WithRateLimits(qps float32, burst int) TypedClientBuilder {
	clientConfig := *b.ClientConfig
	clientConfig.QPS = qps
	clientConfig.Burst = burst
	return SimpleControllerClientBuilder{
		ClientConfig: &clientConfig,
	}
}

// WithDropManagedFields returns a TypedClientBuilder whose clients ask the server to omit managedFields.
func (b SimpleControllerClientBuilder) WithDropManagedFields() TypedClientBuilder {
	clientConfig := *b.ClientConfig
	clientConfig.DropManagedFields = true
	return SimpleControllerClientBuilder{
		ClientConfig: &clientConfig,
	}
}

// Client returns a clientset.Interface built from the ClientBuilder
func (b SimpleControllerClientBuilder) Client(name string) (clientset.Interface, error) {
	clientConfig, err := b.Config(name)
	if err != nil {
		return nil, err
	}
	return clientset.NewForConfig(clientConfig)
}

// ClientOrDie returns a clientset.interface built from the ClientBuilder with no error.
// If it gets an error getting the client, it will log the error and kill the process it's running in.
func (b SimpleControllerClientBuilder) ClientOrDie(name string) clientset.Interface {
	client, err := b.Client(name)
	if err != nil {
		klog.Fatal(err)
	}
	return client
}

// DiscoveryClient returns a discovery.DiscoveryInterface built from the ClientBuilder
// Discovery is special because it will artificially pump the burst quite high to handle the many discovery requests.
func (b SimpleControllerClientBuilder) DiscoveryClient(name string) (discovery.DiscoveryInterface, error) {
	clientConfig, err := b.Config(name)
	if err != nil {
		return nil, err
	}
	// Discovery makes a lot of requests infrequently.  This allows the burst to succeed and refill to happen
	// in just a few seconds.
	if clientConfig.Burst < 200 {
		clientConfig.Burst = 200
	}
	if clientConfig.QPS < 20 {
		clientConfig.QPS = 20
	}
	return clientset.NewForConfig(clientConfig)
}

// DiscoveryClientOrDie returns a discovery.DiscoveryInterface built from the ClientBuilder with no error.
// Discovery is special because it will artificially pump the burst quite high to handle the many discovery requests.
// If it gets an error getting the client, it will log the error and kill the process it's running in.
func (b SimpleControllerClientBuilder) DiscoveryClientOrDie(name string) discovery.DiscoveryInterface {
	client, err := b.DiscoveryClient(name)
	if err != nil {
		klog.Fatal(err)
	}
	return client
}

// MetadataClient returns a metadata.Interface built from the ClientBuilder.
func (b SimpleControllerClientBuilder) MetadataClient(name string) (metadata.Interface, error) {
	clientConfig, err := b.Config(name)
	if err != nil {
		return nil, err
	}
	return metadata.NewForConfig(clientConfig)
}

// DynamicClient returns a dynamic.Interface built from the ClientBuilder.
func (b SimpleControllerClientBuilder) DynamicClient(name string) (dynamic.Interface, error) {
	clientConfig, err := b.Config(name)
	if err != nil {
		return nil, err
	}
	return dynamic.NewForConfig(clientConfig)
}

// ScaleClient returns a scale.ScalesGetter built from the ClientBuilder.
func (b SimpleControllerClientBuilder) ScaleClient(name string, mapper scale.PreferredResourceMapper, resolver dynamic.APIPathResolverFunc, scaleKindResolver scale.ScaleKindResolver) (scale.ScalesGetter, error) {
	clientConfig, err := b.Config(name)
	if err != nil {
		return nil, err
	}
	return scale.NewForConfig(clientConfig, mapper, resolver, scaleKindResolver)
}
