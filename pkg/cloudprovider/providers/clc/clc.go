/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package clc

import (
	"fmt"
	"io"

	"encoding/base64"

	"github.com/golang/glog"
	"gopkg.in/gcfg.v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
)

const (
	// ProviderName clc for CenturyLinkCloud
	ProviderName = "clc"
)

// CLCCloud is an implementation of Interface, LoadBalancer and Zones for CenturyLinkCloud.
type CLCCloud struct {
	clcClient CenturyLinkClient
	clcLB     *clcProviderLB // cloudprovider's LoadBalancer interface is implemented here
	clcConfig Config         // loaded in readConfig
}

func init() {
	glog.Info("registering CLC provider")

	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			glog.Info(fmt.Sprintf("CLC provider could not read config: err=%s", err))
			return nil, err
		}

		newClient := makeCenturyLinkClient()

		newCloud := CLCCloud{
			clcClient: newClient,
			clcLB:     makeProviderLB(newClient),
			clcConfig: cfg,
		}

		newClient.GetCreds().CredsLogin(cfg.Global.Username, cfg.Global.Password) // try the login, but accept that it may fail
		if !newCloud.clcLB.clcClient.GetCreds().IsValid() {
			glog.Info("CLC: initial login is not valid, either it failed or was aliased")
		}

		return &newCloud, nil
	})
}

// Config holds CenturyLinkCloud configuration parameters.  The password is
// base64-encoded not for security but to escape special characters (#) from gcfg parsing
type Config struct {
	Global struct {
		Username   string `gcfg:"username"`
		Password   string `gcfg:"password-base64"`
		Alias      string `gcfg:"alias"`
		Token      string
		Datacenter string `gcfg:"datacenter"`
	}
	// LoadBalancer LoadBalancerOpts
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := clcError("no CenturyLinkCloud provider config file given")
		return Config{}, err
	}

	var cfg Config
	err := gcfg.ReadInto(&cfg, config)
	if err != nil {
		return cfg, err
	}

	password, err := base64.StdEncoding.DecodeString(cfg.Global.Password)
	if err != nil {
		return cfg, err
	}

	cfg.Global.Password = string(password)
	return cfg, err
}

// LoadBalancer returns a balancer interface. Also returns true if the interface is supported, false otherwise.
func (clc *CLCCloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	var ret cloudprovider.LoadBalancer = clc
	if ret == nil {
		glog.Info("CLC LoadBalancer call failed to convert types")
		return ret, true
	}

	glog.Info("CLC LoadBalancer interface successfully returned")
	return ret, true
}

// Instances returns an instances interface. Also returns true if the interface is supported, false otherwise.
func (clc *CLCCloud) Instances() (cloudprovider.Instances, bool) {
	return nil, false
}

// Zones returns a zones interface. Also returns true if the interface is supported, false otherwise.
func (clc *CLCCloud) Zones() (cloudprovider.Zones, bool) {
	var ret cloudprovider.Zones = clc
	if ret == nil {
		glog.Info("CLC Zones call failed to convert types")
		return ret, true
	}

	glog.Info("CLC Zones interface successfully returned")
	return ret, true
}

// Clusters returns a clusters interface.  Also returns true if the interface is supported, false otherwise.
func (clc *CLCCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// Routes returns a routes interface along with whether the interface is supported.
func (clc *CLCCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (clc *CLCCloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS provides an opportunity for cloud-provider-specific code to process DNS settings for pods.
func (clc *CLCCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nil, nil
}

// GetLoadBalancer is an implementation of LoadBalancer.GetLoadBalancer
func (clc *CLCCloud) GetLoadBalancer(service *api.Service) (*api.LoadBalancerStatus, bool, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	zone, err := clc.GetZone()
	if err != nil {
		return nil, false, err
	}
	return clc.clcLB.GetLoadBalancer(loadBalancerName, zone.Region)
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
func (clc *CLCCloud) EnsureLoadBalancer(service *api.Service, hosts []string, annotations map[string]string) (*api.LoadBalancerStatus, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	loadBalancerIP := service.Spec.LoadBalancerIP
	ports := service.Spec.Ports

	zone, err := clc.GetZone()
	if err != nil {
		return nil, err
	}
	affinityType := service.Spec.SessionAffinity

	serviceName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}

	return clc.clcLB.EnsureLoadBalancer(loadBalancerName, zone.Region, loadBalancerIP, ports, hosts, serviceName, affinityType, annotations)
}

// UpdateLoadBalancer is an implementation of LoadBalancer.UpdateLoadBalancer.
func (clc *CLCCloud) UpdateLoadBalancer(service *api.Service, hostNames []string) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	zone, err := clc.GetZone()
	if err != nil {
		return err
	}
	return clc.clcLB.UpdateLoadBalancer(loadBalancerName, zone.Region, hostNames)
}

// EnsureLoadBalancerDeleted is an implementation of LoadBalancer.EnsureLoadBalancerDeleted.
func (clc *CLCCloud) EnsureLoadBalancerDeleted(service *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	zone, err := clc.GetZone()
	if err != nil {
		return err
	}
	glog.V(2).Infof("EnsureLoadBalancerDeleted(%v, %v, %v, %v)", service.Namespace, service.Name, loadBalancerName, zone.Region)
	return clc.clcLB.EnsureLoadBalancerDeleted(loadBalancerName, zone.Region)
}

// GetZone returns the Zone containing the current failure zone and locality region that the program is running in
func (clc *CLCCloud) GetZone() (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: clc.clcConfig.Global.Datacenter,
		Region:        clc.clcConfig.Global.Datacenter,
	}, nil
}
