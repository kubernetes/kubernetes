/*
Copyright 2014 Google Inc. All rights reserved.

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
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"regexp"

	"code.google.com/p/gcfg"
	"github.com/rackspace/gophercloud"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var ErrServerNotFound = errors.New("Server not found")
var ErrMultipleServersFound = errors.New("Multiple servers matched query")
var ErrFlavorNotFound = errors.New("Flavor not found")

// OpenStack is an implementation of cloud provider Interface for OpenStack.
type OpenStack struct {
	provider string
	authOpt  gophercloud.AuthOptions
	region   string
	access   *gophercloud.Access
}

type Config struct {
	Global struct {
		AuthUrl              string
		Username, Password   string
		ApiKey               string
		TenantId, TenantName string
		Region               string
	}
}

func init() {
	cloudprovider.RegisterCloudProvider("openstack", func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newOpenStack(cfg)
	})
}

func (cfg Config) toAuthOptions() gophercloud.AuthOptions {
	return gophercloud.AuthOptions{
		Username:   cfg.Global.Username,
		Password:   cfg.Global.Password,
		ApiKey:     cfg.Global.ApiKey,
		TenantId:   cfg.Global.TenantId,
		TenantName: cfg.Global.TenantName,

		// Persistent service, so we need to be able to renew tokens
		AllowReauth: true,
	}
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("No OpenStack cloud provider config file given")
		return Config{}, err
	}

	var cfg Config
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func newOpenStack(cfg Config) (*OpenStack, error) {
	os := OpenStack{
		provider: cfg.Global.AuthUrl,
		authOpt:  cfg.toAuthOptions(),
		region:   cfg.Global.Region,
	}

	access, err := gophercloud.Authenticate(os.provider, os.authOpt)
	os.access = access

	return &os, err
}

type Instances struct {
	servers            gophercloud.CloudServersProvider
	flavor_to_resource map[string]*api.NodeResources // keyed by flavor id
}

// Instances returns an implementation of Instances for OpenStack.
func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	servers, err := gophercloud.ServersApi(os.access, gophercloud.ApiCriteria{
		Type:      "compute",
		UrlChoice: gophercloud.PublicURL,
		Region:    os.region,
	})

	if err != nil {
		return nil, false
	}

	flavors, err := servers.ListFlavors()
	if err != nil {
		return nil, false
	}
	flavor_to_resource := make(map[string]*api.NodeResources, len(flavors))
	for _, flavor := range flavors {
		rsrc := api.NodeResources{
			Capacity: api.ResourceList{
				"cpu":                      util.NewIntOrStringFromInt(flavor.VCpus),
				"memory":                   util.NewIntOrStringFromString(fmt.Sprintf("%dMiB", flavor.Ram)),
				"openstack.org/disk":       util.NewIntOrStringFromString(fmt.Sprintf("%dGB", flavor.Disk)),
				"openstack.org/rxTxFactor": util.NewIntOrStringFromInt(int(flavor.RxTxFactor * 1000)),
				"openstack.org/swap":       util.NewIntOrStringFromString(fmt.Sprintf("%dMiB", flavor.Swap)),
			},
		}
		flavor_to_resource[flavor.Id] = &rsrc
	}

	return &Instances{servers, flavor_to_resource}, true
}

func (i *Instances) List(name_filter string) ([]string, error) {
	filter := url.Values{}
	filter.Set("name", name_filter)
	filter.Set("status", "ACTIVE")

	servers, err := i.servers.ListServersByFilter(filter)
	if err != nil {
		return nil, err
	}

	ret := make([]string, len(servers))
	for idx, srv := range servers {
		ret[idx] = srv.Name
	}
	return ret, nil
}

func getServerByName(api gophercloud.CloudServersProvider, name string) (*gophercloud.Server, error) {
	filter := url.Values{}
	filter.Set("name", fmt.Sprintf("^%s$", regexp.QuoteMeta(name)))
	filter.Set("status", "ACTIVE")

	servers, err := api.ListServersByFilter(filter)
	if err != nil {
		return nil, err
	}

	if len(servers) == 0 {
		return nil, ErrServerNotFound
	} else if len(servers) > 1 {
		return nil, ErrMultipleServersFound
	}

	return &servers[0], nil
}

func (i *Instances) IPAddress(name string) (net.IP, error) {
	srv, err := getServerByName(i.servers, name)
	if err != nil {
		return nil, err
	}

	var s string
	if len(srv.Addresses.Private) > 0 {
		s = srv.Addresses.Private[0].Addr
	} else if len(srv.Addresses.Public) > 0 {
		s = srv.Addresses.Public[0].Addr
	} else if srv.AccessIPv4 != "" {
		s = srv.AccessIPv4
	} else {
		s = srv.AccessIPv6
	}
	return net.ParseIP(s), nil
}

func (i *Instances) GetNodeResources(name string) (*api.NodeResources, error) {
	srv, err := getServerByName(i.servers, name)
	if err != nil {
		return nil, err
	}

	rsrc, ok := i.flavor_to_resource[srv.Flavor.Id]
	if !ok {
		return nil, ErrFlavorNotFound
	}

	return rsrc, nil
}

func (os *OpenStack) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

func (os *OpenStack) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}
