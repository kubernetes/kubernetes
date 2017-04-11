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

package cloudstack

import (
	"fmt"
	"io"

	"github.com/golang/glog"
	"github.com/xanzy/go-cloudstack/cloudstack"
	"gopkg.in/gcfg.v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// ProviderName is the name of this cloud provider.
const ProviderName = "cloudstack"

// CSConfig wraps the config for the CloudStack cloud provider.
type CSConfig struct {
	Global struct {
		APIURL      string `gcfg:"api-url"`
		APIKey      string `gcfg:"api-key"`
		SecretKey   string `gcfg:"secret-key"`
		SSLNoVerify bool   `gcfg:"ssl-no-verify"`
		ProjectID   string `gcfg:"project-id"`
		Zone        string `gcfg:"zone"`
	}
}

// CSCloud is an implementation of Interface for CloudStack.
type CSCloud struct {
	client    *cloudstack.CloudStackClient
	projectID string // If non-"", all resources will be created within this project
	zone      string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}

		return newCSCloud(cfg)
	})
}

func readConfig(config io.Reader) (*CSConfig, error) {
	if config == nil {
		err := fmt.Errorf("no cloud provider config given")
		return nil, err
	}

	cfg := &CSConfig{}
	if err := gcfg.ReadInto(cfg, config); err != nil {
		glog.Errorf("Couldn't parse config: %v", err)
		return nil, err
	}

	return cfg, nil
}

// newCSCloud creates a new instance of CSCloud.
func newCSCloud(cfg *CSConfig) (*CSCloud, error) {
	client := cloudstack.NewAsyncClient(cfg.Global.APIURL, cfg.Global.APIKey, cfg.Global.SecretKey, !cfg.Global.SSLNoVerify)

	return &CSCloud{client, cfg.Global.ProjectID, cfg.Global.Zone}, nil
}

// LoadBalancer returns an implementation of LoadBalancer for CloudStack.
func (cs *CSCloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return cs, true
}

// Instances returns an implementation of Instances for CloudStack.
func (cs *CSCloud) Instances() (cloudprovider.Instances, bool) {
	return nil, false
}

// Zones returns an implementation of Zones for CloudStack.
func (cs *CSCloud) Zones() (cloudprovider.Zones, bool) {
	return cs, true
}

// Clusters returns an implementation of Clusters for CloudStack.
func (cs *CSCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// Routes returns an implementation of Routes for CloudStack.
func (cs *CSCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (cs *CSCloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (cs *CSCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// GetZone returns the Zone containing the region that the program is running in.
func (cs *CSCloud) GetZone() (cloudprovider.Zone, error) {
	glog.V(2).Infof("Current zone is %v", cs.zone)
	return cloudprovider.Zone{Region: cs.zone}, nil
}
