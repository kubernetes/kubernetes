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

package qingcloud

// Please see qingcloud document: https://docs.qingcloud.com/index.html
// and must pay attention to your account resource quota limit.

import (
	"fmt"
	"io"
	"os"

	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"github.com/magicshui/qingcloud-go"
	"github.com/magicshui/qingcloud-go/instance"
	lb "github.com/magicshui/qingcloud-go/loadbalancer"
	"github.com/magicshui/qingcloud-go/volume"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const (
	ProviderName = "qingcloud"
)

type Config struct {
	Global struct {
		AccessKeyID     string `gcfg:"accessKeyID"`
		SecretAccessKey string `gcfg:"secretAccessKey"`
		Zone            string `gcfg:"zone"`
	}
}

// A single Kubernetes cluster can run in multiple zones,
// but only within the same region (and cloud provider).
type Qingcloud struct {
	instanceClient *instance.INSTANCE
	lbClient       *lb.LOADBALANCER
	volumeClient   *volume.VOLUME
	zone           string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newQingcloud(cfg)
	})
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no qingcloud provider config file given")
		return Config{}, err
	}

	var cfg Config
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

// newQingcloud returns a new instance of Qingcloud cloud provider.
func newQingcloud(config Config) (cloudprovider.Interface, error) {
	clt := qingcloud.NewClient()
	clt.ConnectToZone(config.Global.Zone, config.Global.AccessKeyID, config.Global.SecretAccessKey)

	qc := Qingcloud{
		instanceClient: instance.NewClient(clt),
		lbClient:       lb.NewClient(clt),
		volumeClient:   volume.NewClient(clt),
		zone:           config.Global.Zone,
	}

	host, err := os.Hostname()
	if err != nil {
		return nil, err
	}
	instancesN := qingcloud.NumberedString{}
	instancesN.Add(host)
	resp, err := qc.instanceClient.DescribeInstances(instance.DescribeInstanceRequest{
		InstancesN: instancesN,
	})
	if err != nil {
		return nil, err
	}
	if len(resp.InstanceSet) == 0 {
		return nil, fmt.Errorf("this host(%v) not in qingcloud in zone %v", host, qc.zone)
	}

	glog.V(3).Infof("new Qingcloud: %v", qc)

	return &qc, nil
}

// LoadBalancer returns an implementation of LoadBalancer for Qingcloud.
func (qc *Qingcloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(3).Info("LoadBalancer() called")
	return qc, true
}

// Instances returns an implementation of Instances for Qingcloud.
func (qc *Qingcloud) Instances() (cloudprovider.Instances, bool) {
	glog.V(3).Info("Instances() called")
	return qc, true
}

func (qc *Qingcloud) Zones() (cloudprovider.Zones, bool) {
	return qc, true
}

func (qc *Qingcloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

func (qc *Qingcloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

func (qc *Qingcloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (qc *Qingcloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

func (qc *Qingcloud) GetZone() (cloudprovider.Zone, error) {
	glog.V(3).Infof("GetZone() called, current zone is %v", qc.zone)

	return cloudprovider.Zone{Region: qc.zone}, nil
}
