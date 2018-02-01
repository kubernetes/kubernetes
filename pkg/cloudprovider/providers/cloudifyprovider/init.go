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
	"encoding/json"
	"fmt"
	cloudify "github.com/cloudify-incubator/cloudify-rest-go-client/cloudify"
	"github.com/golang/glog"
	"io"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"os"
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

		r.balancers = NewBalancer(r.client, r.deployment, r.scaleGroup)
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
	return false
}

// ScrubDNS provides an opportunity for cloud-provider-specific code to process DNS settings for pods.
func (r *CloudProvider) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	glog.Errorf("?ScrubDNS: Name Servers: %+v ", nameservers)
	glog.Errorf("?ScrubDNS: Searches: %+v ", searches)
	return nameservers, searches
}

// Config - settings for connect to cloudify
type Config struct {
	Host               string `json:"host,omitempty"`
	User               string `json:"user,omitempty"`
	Password           string `json:"password,omitempty"`
	Tenant             string `json:"tenant,omitempty"`
	Deployment         string `json:"deployment,omitempty"`
	LoadBalancersScale string `json:"loadbalancer,omitempty"`
}

// newCloudifyCloud - load connection configuration from file
func newCloudifyCloud(config io.Reader) (cloudprovider.Interface, error) {
	glog.V(4).Info("New Cloudify client")

	var cloudConfig Config
	cloudConfig.Host = os.Getenv("CFY_HOST")
	cloudConfig.User = os.Getenv("CFY_USER")
	cloudConfig.Password = os.Getenv("CFY_PASSWORD")
	cloudConfig.Tenant = os.Getenv("CFY_TENANT")
	cloudConfig.LoadBalancersScale = os.Getenv("CFY_LOADBALANCER")
	if config != nil {
		err := json.NewDecoder(config).Decode(&cloudConfig)
		if err != nil {
			return nil, err
		}
	}

	if len(cloudConfig.Host) == 0 {
		return nil, fmt.Errorf("You have empty host")
	}

	if len(cloudConfig.User) == 0 {
		return nil, fmt.Errorf("You have empty user")
	}

	if len(cloudConfig.Password) == 0 {
		return nil, fmt.Errorf("You have empty password")
	}

	if len(cloudConfig.Tenant) == 0 {
		return nil, fmt.Errorf("You have empty tenant")
	}

	if len(cloudConfig.Deployment) == 0 {
		return nil, fmt.Errorf("You have empty deployment")
	}

	if len(cloudConfig.LoadBalancersScale) == 0 {
		cloudConfig.LoadBalancersScale = "k8s_load_scale_group"
	}

	glog.V(4).Infof("Config %+v", cloudConfig)
	return &CloudProvider{
		deployment: cloudConfig.Deployment,
		scaleGroup: cloudConfig.LoadBalancersScale,
		client: cloudify.NewClient(
			cloudConfig.Host, cloudConfig.User,
			cloudConfig.Password, cloudConfig.Tenant),
	}, nil
}

// init - code for register cloudify as provider
func init() {
	glog.V(4).Info("Cloudify init")
	cloudprovider.RegisterCloudProvider(providerName, func(config io.Reader) (cloudprovider.Interface, error) {
		return newCloudifyCloud(config)
	})
}
