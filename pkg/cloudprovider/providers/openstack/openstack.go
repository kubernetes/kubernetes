/*
Copyright 2014 The Kubernetes Authors.

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

package openstack

import (
	"crypto/tls"
	"io"
	"net/http"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/extensions/trusts"

	"github.com/golang/glog"
	netutil "k8s.io/apimachinery/pkg/util/net"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
)

const ProviderName = "openstack"

// OpenStack is an implementation of cloud provider Interface for OpenStack.
type OpenStack struct {
	provider  *gophercloud.ProviderClient
	region    string
	lbOpts    LoadBalancerOpts
	bsOpts    BlockStorageOpts
	routeOpts RouterOpts
	// InstanceID of the server where this OpenStack object is instantiated.
	localInstanceID string
}

func init() {
	RegisterMetrics()

	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newOpenStack(cfg)
	})
}

func newOpenStack(cfg Config) (*OpenStack, error) {
	provider, err := openstack.NewClient(cfg.Global.AuthUrl)
	if err != nil {
		return nil, err
	}
	if cfg.Global.CAFile != "" {
		roots, err := certutil.NewPool(cfg.Global.CAFile)
		if err != nil {
			return nil, err
		}
		config := &tls.Config{}
		config.RootCAs = roots
		provider.HTTPClient.Transport = netutil.SetOldTransportDefaults(&http.Transport{TLSClientConfig: config})

	}
	if cfg.Global.TrustId != "" {
		opts := cfg.toAuth3Options()
		authOptsExt := trusts.AuthOptsExt{
			TrustID:            cfg.Global.TrustId,
			AuthOptionsBuilder: &opts,
		}
		err = openstack.AuthenticateV3(provider, authOptsExt, gophercloud.EndpointOpts{})
	} else {
		err = openstack.Authenticate(provider, cfg.toAuthOptions())
	}

	if err != nil {
		return nil, err
	}

	os := OpenStack{
		provider:  provider,
		region:    cfg.Global.Region,
		lbOpts:    cfg.LoadBalancer,
		bsOpts:    cfg.BlockStorage,
		routeOpts: cfg.Route,
	}

	err = checkOpenStackOpts(&os)
	if err != nil {
		return nil, err
	}

	return &os, nil
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (os *OpenStack) Initialize(clientBuilder controller.ControllerClientBuilder) {}

func (os *OpenStack) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (os *OpenStack) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (os *OpenStack) ScrubDNS(nameServers, searches []string) ([]string, []string) {
	return nameServers, searches
}

// HasClusterID returns true if the cluster has a clusterID
func (os *OpenStack) HasClusterID() bool {
	return true
}

type LoadBalancer struct {
	network *gophercloud.ServiceClient
	compute *gophercloud.ServiceClient
	opts    LoadBalancerOpts
}

func (os *OpenStack) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("openstack.LoadBalancer() called")

	// TODO: Search for and support Rackspace loadbalancer API, and others.
	network, err := os.NewNetworkV2()
	if err != nil {
		return nil, false
	}

	compute, err := os.NewComputeV2()
	if err != nil {
		return nil, false
	}

	lbVersion := os.lbOpts.LBVersion
	if lbVersion == "" {
		// No version specified, try newest supported by server
		netExts, err := networkExtensions(network)
		if err != nil {
			glog.Warningf("Failed to list neutron extensions: %v", err)
			return nil, false
		}

		if netExts["lbaasv2"] {
			lbVersion = "v2"
		} else if netExts["lbaas"] {
			lbVersion = "v1"
		} else {
			glog.Warningf("Failed to find neutron LBaaS extension (v1 or v2)")
			return nil, false
		}
		glog.V(3).Infof("Using LBaaS extension %v", lbVersion)
	}

	glog.V(1).Info("Claiming to support LoadBalancer")

	if lbVersion == "v2" {
		return &LbaasV2{LoadBalancer{network, compute, os.lbOpts}}, true
	} else if lbVersion == "v1" {
		return &LbaasV1{LoadBalancer{network, compute, os.lbOpts}}, true
	} else {
		glog.Warningf("Config error: unrecognised lb-version \"%v\"", lbVersion)
		return nil, false
	}
}

// Instances returns an implementation of Instances for OpenStack.
func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("openstack.Instances() called")

	compute, err := os.NewComputeV2()
	if err != nil {
		return nil, false
	}

	glog.V(1).Info("Claiming to support Instances")

	return &Instances{compute}, true
}

func (os *OpenStack) Zones() (cloudprovider.Zones, bool) {
	glog.V(1).Info("Claiming to support Zones")

	return os, true
}

func (os *OpenStack) Routes() (cloudprovider.Routes, bool) {
	glog.V(4).Info("openstack.Routes() called")

	network, err := os.NewNetworkV2()
	if err != nil {
		return nil, false
	}

	netExts, err := networkExtensions(network)
	if err != nil {
		glog.Warningf("Failed to list neutron extensions: %v", err)
		return nil, false
	}

	if !netExts["extraroute"] {
		glog.V(3).Infof("Neutron extraroute extension not found, required for Routes support")
		return nil, false
	}

	compute, err := os.NewComputeV2()
	if err != nil {
		return nil, false
	}

	r, err := NewRoutes(compute, network, os.routeOpts)
	if err != nil {
		glog.Warningf("Error initialising Routes support: %v", err)
		return nil, false
	}

	glog.V(1).Info("Claiming to support Routes")

	return r, true
}
