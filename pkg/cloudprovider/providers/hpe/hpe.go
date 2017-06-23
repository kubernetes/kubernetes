/*
Copyright 2017 The Kubernetes Authors.

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

package hpe

import (
	"fmt"
	"io"
	"os"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
)

const ProviderName = "hpe"

type LoadBalancer struct{}

type HpeCloud struct {
	region string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		startIP := os.Getenv("START_IP")
		endIP := os.Getenv("END_IP")
		glog.Errorf("IPAM range [%s - %s]", startIP, endIP)
		ipamStatus := initializeIpam(startIP, endIP)
		if ipamStatus == false {
			fmt.Errorf("Invalid IPAM range")
		}
		return newHpeCloud()
	})
}

func newHpeCloud() (*HpeCloud, error) {
	return &HpeCloud{}, nil
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (hpe *HpeCloud) Initialize(clientBuilder controller.ControllerClientBuilder) {}

func (hpe *HpeCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (hpe *HpeCloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (hpe *HpeCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

func (hpe *HpeCloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("hpe.LoadBalancer() called")
	return &Lbaas{LoadBalancer{}}, true
}

func (hpe *HpeCloud) Zones() (cloudprovider.Zones, bool) {
	return hpe, true
}

func (hpe *HpeCloud) Instances() (cloudprovider.Instances, bool) {
	return nil, false
}

func (hpe *HpeCloud) GetZone() (cloudprovider.Zone, error) {
	zone := cloudprovider.Zone{
		FailureDomain: "zone-1",
		Region:        hpe.region,
	}
	glog.V(1).Infof("Current zone is %v", zone)

	return zone, nil
}

func (hpe *HpeCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}
