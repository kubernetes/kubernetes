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

package remote

import (
	"errors"
	"github.com/golang/glog"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/networkprovider"
	"strings"
)

func ProbeNetworkProviders() {
	files, _ := ioutil.ReadDir(PluginsPath)
	for _, f := range files {
		// only treat .sock/.spec as plugins
		if !f.IsDir() {
			if strings.HasSuffix(f.Name(), ".sock") || strings.HasSuffix(f.Name(), ".spec") {
				pluginName := f.Name()[:len(f.Name())-5]
				networkprovider.RegisterNetworkProvider(pluginName, func() (networkprovider.Interface, error) {
					plugin, err := GetPlugin(pluginName)
					if err != nil {
						glog.Warningf("Initialize network provider %s failed: %v", pluginName, err)
						return nil, err
					} else {
						glog.V(4).Infof("Network provider %s initialized", pluginName)
					}

					return plugin, nil
				})
			}
		}
	}
}

func (p *Plugin) ProviderName() string {
	return p.Name
}

func (p *Plugin) CheckTenantID(tenantID string) (bool, error) {
	ret := CheckTenantIDResponse{}
	args := CheckTenantIDRequest{TenantID: tenantID}
	err := p.Client.Call(CheckTenantIDMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider check tenant id %s failed: %v", tenantID, err)
		return false, err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider check tenant id %s failed: %v", tenantID, ret.GetError())
		return false, errors.New(ret.GetError())
	}

	return ret.Result, nil
}

// Networks returns an network interface
func (p *Plugin) Networks() networkprovider.Networks {
	return p
}

// LoadBalancer returns a balancer interface
func (p *Plugin) LoadBalancers() networkprovider.LoadBalancers {
	return p
}

// Get network by networkName
func (p *Plugin) GetNetwork(networkName string) (*networkprovider.Network, error) {
	ret := GetNetworkResponse{}
	args := GetNetworkRequest{Name: networkName}
	err := p.Client.Call(GetNetworkMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider get network %s failed: %v", networkName, err)
		return nil, err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider get network %s failed: %v", networkName, ret.GetError())
		return nil, errors.New(ret.GetError())
	}

	return ret.Result, nil
}

// Get network by networkID
func (p *Plugin) GetNetworkByID(networkID string) (*networkprovider.Network, error) {
	ret := GetNetworkResponse{}
	args := GetNetworkRequest{ID: networkID}
	err := p.Client.Call(GetNetworkMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider get network %s failed: %v", networkID, err)
		return nil, err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider get network %s failed: %v", networkID, ret.GetError())
		return nil, errors.New(ret.GetError())
	}

	return ret.Result, nil
}

// Create network
func (p *Plugin) CreateNetwork(network *networkprovider.Network) error {
	ret := CreateNetworkResponse{}
	args := CreateNetworkRequest{Network: network}
	err := p.Client.Call(CreateNetworkMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider create network %s failed: %v", network.Name, err)
		return err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider get network %s failed: %v", network.Name, ret.GetError())
		return errors.New(ret.GetError())
	}

	return nil
}

// Update network
func (p *Plugin) UpdateNetwork(network *networkprovider.Network) error {
	ret := UpdateNetworkResponse{}
	args := UpdateNetworkRequest{Network: network}
	err := p.Client.Call(UpdateNetworkMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider update network %s failed: %v", network.Name, err)
		return err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider update network %s failed: %v", network.Name, ret.GetError())
		return errors.New(ret.GetError())
	}

	return nil
}

// Delete network by networkName
func (p *Plugin) DeleteNetwork(networkName string) error {
	ret := DeleteNetworkResponse{}
	args := DeleteNetworkRequest{Name: networkName}
	err := p.Client.Call(DeleteNetworkMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider delete network %s failed: %v", networkName, err)
		return err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider delete  network %s failed: %v", networkName, ret.GetError())
		return errors.New(ret.GetError())
	}

	return nil
}

// Get load balancer by name
func (p *Plugin) GetLoadBalancer(name string) (*networkprovider.LoadBalancer, error) {
	ret := GetLoadBalancerResponse{}
	args := GetLoadBalancerRequest{Name: name}
	err := p.Client.Call(GetLoadBalancerMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider get loadbalancer %s failed: %v", name, err)
		return nil, err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider get loadbalancer %s failed: %v", name, ret.GetError())
		return nil, errors.New(ret.GetError())
	}

	return ret.Result, nil
}

// Create load balancer, return ip and externalIP
func (p *Plugin) CreateLoadBalancer(loadBalancer *networkprovider.LoadBalancer, affinity api.ServiceAffinity) (string, error) {
	ret := CreateLoadBalancerResponse{}
	args := CreateLoadBalancerRequest{
		LoadBalancer: loadBalancer,
		Affinity:     affinity,
	}
	err := p.Client.Call(CreateLoadBalancerMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider create loadbalancer %s failed: %v", loadBalancer.Name, err)
		return "", err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider create loadbalancer %s failed: %v", loadBalancer.Name, ret.GetError())
		return "", errors.New(ret.GetError())
	}

	return ret.Result.VIP, nil
}

// Update load balancer, return externalIP
func (p *Plugin) UpdateLoadBalancer(name string, hosts []*networkprovider.HostPort, externalIPs []string) (string, error) {
	ret := UpdateLoadBalancerResponse{}
	args := UpdateLoadBalancerRequest{
		Name:        name,
		Hosts:       hosts,
		ExternalIPs: externalIPs,
	}
	err := p.Client.Call(UpdateLoadBalancerMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider update loadbalancer %s failed: %v", name, err)
		return "", err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider update loadbalancer %s failed: %v", name, ret.GetError())
		return "", errors.New(ret.GetError())
	}

	return ret.Result.VIP, nil
}

// Delete load balancer
func (p *Plugin) DeleteLoadBalancer(name string) error {
	ret := DeleteLoadBalancerResponse{}
	args := DeleteLoadBalancerRequest{Name: name}
	err := p.Client.Call(DeleteLoadBalancerMethod, args, &ret)
	if err != nil {
		glog.Warningf("NetworkProvider delete loadbalancer %s failed: %v", name, err)
		return err
	}

	if ret.GetError() != "" {
		glog.Warningf("NetworkProvider delete  loadbalancer %s failed: %v", name, ret.GetError())
		return errors.New(ret.GetError())
	}

	return nil
}
