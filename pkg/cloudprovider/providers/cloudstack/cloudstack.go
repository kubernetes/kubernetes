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
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/kardianos/osext"
	"github.com/xanzy/go-cloudstack/cloudstack"
	"gopkg.in/gcfg.v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
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
	metadata  *metadata
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
	cfg := &CSConfig{}

	if config == nil {
		return cfg, nil
	}

	if err := gcfg.ReadInto(cfg, config); err != nil {
		return nil, fmt.Errorf("could not parse cloud provider config: %v", err)
	}

	return cfg, nil
}

// newCSCloud creates a new instance of CSCloud.
func newCSCloud(cfg *CSConfig) (*CSCloud, error) {
	cs := &CSCloud{
		projectID: cfg.Global.ProjectID,
		zone:      cfg.Global.Zone,
	}

	exe, err := osext.Executable()
	if err != nil {
		return nil, fmt.Errorf("cloud not find the service executable: %v", err)
	}

	// When running the kubelet service it's fine to not specify a config file (or only a
	// partial config file) as all needed info can be retrieved anonymously using metadata.
	if filepath.Base(exe) == "kubelet" || filepath.Base(exe) == "kubelet.exe" {
		// In CloudStack your metadata is always served by the DHCP server.
		dhcpServer, err := findDHCPServer()
		if err == nil {
			glog.V(4).Infof("Found metadata server: %v", dhcpServer)
			cs.metadata = &metadata{dhcpServer: dhcpServer, zone: cs.zone}
		} else {
			glog.Errorf("Error searching metadata server: %v", err)
		}
	}

	if cfg.Global.APIURL != "" && cfg.Global.APIKey != "" && cfg.Global.SecretKey != "" {
		cs.client = cloudstack.NewAsyncClient(cfg.Global.APIURL, cfg.Global.APIKey, cfg.Global.SecretKey, !cfg.Global.SSLNoVerify)
	}

	if cs.client == nil {
		if cs.metadata != nil {
			glog.V(2).Infof("No API URL, key and secret are provided, so only using metadata!")
		} else {
			return nil, errors.New("no cloud provider config given")
		}
	}

	return cs, nil
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (cs *CSCloud) Initialize(clientBuilder controller.ControllerClientBuilder) {}

// LoadBalancer returns an implementation of LoadBalancer for CloudStack.
func (cs *CSCloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	if cs.client == nil {
		return nil, false
	}

	return cs, true
}

// Instances returns an implementation of Instances for CloudStack.
func (cs *CSCloud) Instances() (cloudprovider.Instances, bool) {
	if cs.metadata != nil {
		return cs.metadata, true
	}

	if cs.client == nil {
		return nil, false
	}

	return cs, true
}

// Zones returns an implementation of Zones for CloudStack.
func (cs *CSCloud) Zones() (cloudprovider.Zones, bool) {
	if cs.metadata != nil {
		return cs.metadata, true
	}

	if cs.client == nil {
		return nil, false
	}

	return cs, true
}

// Clusters returns an implementation of Clusters for CloudStack.
func (cs *CSCloud) Clusters() (cloudprovider.Clusters, bool) {
	if cs.client == nil {
		return nil, false
	}

	return nil, false
}

// Routes returns an implementation of Routes for CloudStack.
func (cs *CSCloud) Routes() (cloudprovider.Routes, bool) {
	if cs.client == nil {
		return nil, false
	}

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

// HasClusterID returns true if the cluster has a clusterID
func (cs *CSCloud) HasClusterID() bool {
	return true
}

// GetZone returns the Zone containing the region that the program is running in.
func (cs *CSCloud) GetZone() (cloudprovider.Zone, error) {
	zone := cloudprovider.Zone{}

	if cs.zone == "" {
		hostname, err := os.Hostname()
		if err != nil {
			return zone, fmt.Errorf("failed to get hostname for retrieving the zone: %v", err)
		}

		instance, count, err := cs.client.VirtualMachine.GetVirtualMachineByName(hostname)
		if err != nil {
			if count == 0 {
				return zone, fmt.Errorf("could not find instance for retrieving the zone: %v", err)
			}
			return zone, fmt.Errorf("error getting instance for retrieving the zone: %v", err)
		}

		cs.zone = instance.Zonename
	}

	glog.V(2).Infof("Current zone is %v", cs.zone)
	zone.FailureDomain = cs.zone
	zone.Region = cs.zone

	return zone, nil
}

// GetZoneByProviderID returns the Zone, found by using the provider ID.
func (cs *CSCloud) GetZoneByProviderID(providerID string) (cloudprovider.Zone, error) {
	zone := cloudprovider.Zone{}

	instance, count, err := cs.client.VirtualMachine.GetVirtualMachineByID(
		providerID,
		cloudstack.WithProject(cs.projectID),
	)
	if err != nil {
		if count == 0 {
			return zone, fmt.Errorf("could not find node by ID: %v", providerID)
		}
		return zone, fmt.Errorf("error retrieving zone: %v", err)
	}

	glog.V(2).Infof("Current zone is %v", cs.zone)
	zone.FailureDomain = instance.Zonename
	zone.Region = instance.Zonename

	return zone, nil
}

// GetZoneByNodeName returns the Zone, found by using the node name.
func (cs *CSCloud) GetZoneByNodeName(nodeName types.NodeName) (cloudprovider.Zone, error) {
	zone := cloudprovider.Zone{}

	instance, count, err := cs.client.VirtualMachine.GetVirtualMachineByName(
		string(nodeName),
		cloudstack.WithProject(cs.projectID),
	)
	if err != nil {
		if count == 0 {
			return zone, fmt.Errorf("could not find node: %v", nodeName)
		}
		return zone, fmt.Errorf("error retrieving zone: %v", err)
	}

	glog.V(2).Infof("Current zone is %v", cs.zone)
	zone.FailureDomain = instance.Zonename
	zone.Region = instance.Zonename

	return zone, nil
}
