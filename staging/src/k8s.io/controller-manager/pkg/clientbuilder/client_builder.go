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
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

// ControllerClientBuilder allows you to get clients and configs for controllers
// Please note a copy also exists in staging/src/k8s.io/cloud-provider/cloud.go
// TODO: Extract this into a separate controller utilities repo (issues/68947)
type ControllerClientBuilder interface {
	Config(logger klog.Logger, name string) (*restclient.Config, error)
	ConfigOrDie(logger klog.Logger, name string) *restclient.Config
	Client(logger klog.Logger, name string) (clientset.Interface, error)
	ClientOrDie(logger klog.Logger, name string) clientset.Interface
	DiscoveryClient(logger klog.Logger, name string) (discovery.DiscoveryInterface, error)
	DiscoveryClientOrDie(logger klog.Logger, name string) discovery.DiscoveryInterface
}

// SimpleControllerClientBuilder returns a fixed client with different user agents
type SimpleControllerClientBuilder struct {
	// ClientConfig is a skeleton config to clone and use as the basis for each controller client
	ClientConfig *restclient.Config
}

// Config returns a client config for a fixed client
func (b SimpleControllerClientBuilder) Config(logger klog.Logger, name string) (*restclient.Config, error) {
	clientConfig := *b.ClientConfig
	return restclient.AddUserAgent(&clientConfig, name), nil
}

// ConfigOrDie returns a client config if no error from previous config func.
// If it gets an error getting the client, it will log the error and kill the process it's running in.
func (b SimpleControllerClientBuilder) ConfigOrDie(logger klog.Logger, name string) *restclient.Config {
	clientConfig, err := b.Config(logger, name)
	if err != nil {
		logger.Error(err, "Error in getting client config")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	return clientConfig
}

// Client returns a clientset.Interface built from the ClientBuilder
func (b SimpleControllerClientBuilder) Client(logger klog.Logger, name string) (clientset.Interface, error) {
	clientConfig, err := b.Config(logger, name)
	if err != nil {
		return nil, err
	}
	return clientset.NewForConfig(clientConfig)
}

// ClientOrDie returns a clientset.interface built from the ClientBuilder with no error.
// If it gets an error getting the client, it will log the error and kill the process it's running in.
func (b SimpleControllerClientBuilder) ClientOrDie(logger klog.Logger, name string) clientset.Interface {
	client, err := b.Client(logger, name)
	if err != nil {
		logger.Error(err, "Error in getting client")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	return client
}

// DiscoveryClient returns a discovery.DiscoveryInterface built from the ClientBuilder
// Discovery is special because it will artificially pump the burst quite high to handle the many discovery requests.
func (b SimpleControllerClientBuilder) DiscoveryClient(logger klog.Logger, name string) (discovery.DiscoveryInterface, error) {
	clientConfig, err := b.Config(logger, name)
	if err != nil {
		return nil, err
	}
	// Discovery makes a lot of requests infrequently.  This allows the burst to succeed and refill to happen
	// in just a few seconds.
	clientConfig.Burst = 200
	clientConfig.QPS = 20
	return clientset.NewForConfig(clientConfig)
}

// DiscoveryClientOrDie returns a discovery.DiscoveryInterface built from the ClientBuilder with no error.
// Discovery is special because it will artificially pump the burst quite high to handle the many discovery requests.
// If it gets an error getting the client, it will log the error and kill the process it's running in.
func (b SimpleControllerClientBuilder) DiscoveryClientOrDie(logger klog.Logger, name string) discovery.DiscoveryInterface {
	client, err := b.DiscoveryClient(logger, name)
	if err != nil {
		logger.Error(err, "Error in getting client")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	return client
}
