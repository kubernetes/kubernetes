/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package vsphere

import (
	"fmt"
	"io"
	"net/url"

	"github.com/vmware/govmomi"
	"golang.org/x/net/context"
	"gopkg.in/gcfg.v1"

	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "vsphere"

// VSphere is an implementation of Interface, LoadBalancer and Instances for vSphere.
type VSphere struct {
	provider   *govmomi.Client
	region     string
	datacenter string
	// cfg      *VSphereConfig
	// Add vSphere instance
}

type VSphereConfig struct {
	Global struct {
		User          string `gcfg:"user"`
		Password      string `gcfg:"password"`
		VSphereServer string `gcfg:"server"`
		InsecureFlag  bool   `gcfg:"insecure-flag"`
		Datacenter    string `gcfg:"datacenter"`
	}
}

func readConfig(config io.Reader) (VSphereConfig, error) {
	if config == nil {
		err := fmt.Errorf("no vSphere cloud provider config file given")
		return VSphereConfig{}, err
	}

	var cfg VSphereConfig
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newVSphere(cfg)
	})
}

func newVSphere(cfg VSphereConfig) (*VSphere, error) {

	u, err := url.Parse("https://" + cfg.Global.VSphereServer + "/sdk")
	if err != nil {
		return nil, fmt.Errorf("Error parse url: %s", err)
	}

	u.User = url.UserPassword(cfg.Global.User, cfg.Global.Password)

	provider, err := govmomi.NewClient(context.TODO(), u, cfg.Global.InsecureFlag)
	if err != nil {
		return nil, err
	}

	vs := VSphere{
		provider:   provider,
		datacenter: cfg.Global.Datacenter,
		// cfg              *VSphereConfig,
	}
	return &vs, nil
}

type Instances struct {
}

// Instances returns an implementation of Instances for vSphere.
func (vs *VSphere) Instances() (cloudprovider.Instances, bool) {
	return nil, true
}

func (i *Instances) List(nameFilter string) ([]string, error) {
	return nil, nil
}

func (vs *VSphere) Clusters() (cloudprovider.Clusters, bool) {
	return nil, true
}

// ProviderName returns the cloud provider ID.
func (vs *VSphere) ProviderName() string {
	return ProviderName
}

// LoadBalancer returns an implementation of LoadBalancer for vSphere.
func (vs *VSphere) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}

// Zones returns an implementation of Zones for Google vSphere.
func (vs *VSphere) Zones() (cloudprovider.Zones, bool) {
	return nil, true
}

// Routes returns an implementation of Routes for vSphere.
func (vs *VSphere) Routes() (cloudprovider.Routes, bool) {
	return nil, true
}

// ScrubDNS filters DNS settings for pods.
func (vs *VSphere) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}
