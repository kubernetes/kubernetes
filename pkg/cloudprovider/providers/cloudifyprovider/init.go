/*
Copyright (c) 2017 GigaSpaces Technologies Ltd. All rights reserved

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

package cloudifyprovider

import (
	cloudify "github.com/cloudify-incubator/cloudify-rest-go-client/cloudify"
	"github.com/golang/glog"
	"io"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	providerName = "cloudify"
)

// CloudProvider implents Instances, Zones, and LoadBalancer
type CloudProvider struct {
	deployment string
	scaleGroup string
	client     *cloudify.Client
	instances  *Instances
	balancers  *Balancer
	zones      *Zones
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (r *CloudProvider) Initialize(clientBuilder controller.ControllerClientBuilder) {
	glog.V(4).Info("Initialize")
}

// ProviderName returns the cloud provider ID.
func (r *CloudProvider) ProviderName() string {
	return providerName
}

// LoadBalancer returns a balancer interface. Also returns true if the interface is supported, false otherwise.
func (r *CloudProvider) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("LoadBalancer")
	if r.client != nil {
		if r.balancers != nil {
			return r.balancers, true
		}

		r.balancers = NewBalancer(r.client, r.deployment)
		return r.balancers, true
	}
	return nil, false
}

// Zones returns a zones interface. Also returns true if the interface is supported, false otherwise.
func (r *CloudProvider) Zones() (cloudprovider.Zones, bool) {
	glog.V(4).Info("Zones")
	if r.client != nil {
		if r.zones != nil {
			return r.zones, true
		}

		r.zones = NewZones(r.client)
		return r.zones, true
	}
	return nil, false
}

// Instances returns an instances interface. Also returns true if the interface is supported, false otherwise.
func (r *CloudProvider) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("Instances")
	if r.client != nil {
		if r.instances != nil {
			return r.instances, true
		}

		r.instances = NewInstances(r.client, r.deployment)
		return r.instances, true
	}
	return nil, false
}

// Clusters returns a clusters interface.  Also returns true if the interface is supported, false otherwise.
func (r *CloudProvider) Clusters() (cloudprovider.Clusters, bool) {
	glog.Error("?Clusters")
	return nil, false
}

// Routes returns a routes interface along with whether the interface is supported.
func (r *CloudProvider) Routes() (cloudprovider.Routes, bool) {
	glog.Error("?Routers")
	return nil, false
}

// HasClusterID returns true if a ClusterID is required and set
func (r *CloudProvider) HasClusterID() bool {
	// required by kubernetes-1.9
	return true
}

// ScrubDNS provides an opportunity for cloud-provider-specific code to process DNS settings for pods.
func (r *CloudProvider) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	glog.Errorf("?ScrubDNS: Name Servers: %+v, Searches: %+v", nameservers, searches)
	return nameservers, searches
}

// newCloudifyCloud - load connection configuration from file
func newCloudifyCloud(config io.Reader) (cloudprovider.Interface, error) {
	glog.V(4).Info("New Cloudify client")

	cloudConfig, err := cloudify.ServiceClientInit(config)
	if err != nil {
		return nil, err
	}
	return &CloudProvider{
		deployment: cloudConfig.DeploymentsFile,
		client:     cloudify.NewClient(cloudConfig.ClientConfig),
	}, nil
}

// init - code for register cloudify as provider
func init() {
	glog.V(4).Info("Cloudify init")
	cloudprovider.RegisterCloudProvider(providerName, func(config io.Reader) (cloudprovider.Interface, error) {
		return newCloudifyCloud(config)
	})
}
