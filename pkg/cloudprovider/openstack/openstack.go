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
	"fmt"
	"io"

	"code.google.com/p/gcfg"
	"github.com/rackspace/gophercloud"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

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

func (os *OpenStack) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	return nil, false
}

func (os *OpenStack) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}
