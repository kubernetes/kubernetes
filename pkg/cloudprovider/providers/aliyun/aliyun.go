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

package aliyun

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/ecs"
	"github.com/denverdino/aliyungo/slb"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const (
	ProviderName = "aliyun"
)

type LoadBalancerOpts struct {
	// internet | intranet, default: internet
	AddressType        slb.AddressType           `json:"addressType"`
	InternetChargeType common.InternetChargeType `json:"internetChargeType"`
	// Bandwidth peak of the public network instance charged per fixed bandwidth.
	// Value:1-1000(in Mbps), default: 1
	Bandwidth int `json:"bandwidth"`
}

type Config struct {
	Global struct {
		AccessKeyID     string `json:"accessKeyID"`
		AccessKeySecret string `json:"accessKeySecret"`
		RegionID        string `json:"regionID"`
	}
	LoadBalancer LoadBalancerOpts
}

// A single Kubernetes cluster can run in multiple zones,
// but only within the same region (and cloud provider).
type Aliyun struct {
	ecsClient *ecs.Client
	slbClient *slb.Client
	regionID  string
	lbOpts    LoadBalancerOpts
	// InstanceID of the server where this Aliyun object is instantiated.
	localInstanceID string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newAliyun(cfg)
	})
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("No cloud provider config given")
		return Config{}, err
	}

	cfg := Config{}
	if err := json.NewDecoder(config).Decode(&cfg); err != nil {
		glog.Errorf("Couldn't parse config: %v", err)
		return Config{}, err
	}

	return cfg, nil
}

// newAliyun returns a new instance of Aliyun cloud provider.
func newAliyun(config Config) (cloudprovider.Interface, error) {
	ecsClient := ecs.NewClient(config.Global.AccessKeyID, config.Global.AccessKeySecret)
	slbClient := slb.NewClient(config.Global.AccessKeyID, config.Global.AccessKeySecret)

	// Get the local instance by it's hostname.
	hostname, err := os.Hostname()
	if err != nil {
		glog.Errorf("Error get os.Hostname: %v", err)
		return nil, err
	}

	glog.V(4).Infof("Get the local instance hostname: %s", hostname)

	if config.LoadBalancer.AddressType == "" {
		config.LoadBalancer.AddressType = slb.InternetAddressType
	}

	if config.LoadBalancer.InternetChargeType == "" {
		/* Valid value: paybytraffic|paybybandwidth
		 *  https://help.aliyun.com/document_detail/27577.html?spm=5176.product27537.6.118.R6Bqe6
		 *
		 * aliyun bug:
		 * We cloudn't use common.PayByBandwidth:
		 *     PayByBandwidth = InternetChargeType("PayByBandwidth"))
		 * but InternetChargeType("paybybandwidth")
		 */
		config.LoadBalancer.InternetChargeType = common.InternetChargeType("paybytraffic")
	}

	if config.LoadBalancer.AddressType == slb.InternetAddressType && config.LoadBalancer.InternetChargeType == common.InternetChargeType("paybybandwidth") {
		if config.LoadBalancer.Bandwidth == 0 {
			config.LoadBalancer.Bandwidth = 1
		}

		if config.LoadBalancer.Bandwidth < 1 || config.LoadBalancer.Bandwidth > 1000 {
			return nil, fmt.Errorf("LoadBalancer.Bandwidth '%d' is out of range [1, 1000]", config.LoadBalancer.Bandwidth)
		}
	}

	aly := Aliyun{
		ecsClient: ecsClient,
		slbClient: slbClient,
		regionID:  config.Global.RegionID,
		lbOpts:    config.LoadBalancer,
	}

	glog.V(4).Infof("new Aliyun: '%v'", aly)

	return &aly, nil
}

func (aly *Aliyun) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("aliyun.LoadBalancer() called")
	return aly, true
}

// Instances returns an implementation of Interface.Instances for Aliyun cloud.
func (aly *Aliyun) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("aliyun.Instances() called")
	return aly, true
}

func (aly *Aliyun) Zones() (cloudprovider.Zones, bool) {
	return aly, true
}

func (aly *Aliyun) Clusters() (cloudprovider.Clusters, bool) {
	glog.V(4).Info("aliyun.Clusters() called")
	return nil, false
}

func (aly *Aliyun) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

func (aly *Aliyun) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (aly *Aliyun) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

func (aly *Aliyun) GetZone() (cloudprovider.Zone, error) {
	glog.V(1).Infof("Current zone is %v", aly.regionID)

	return cloudprovider.Zone{Region: aly.regionID}, nil
}
