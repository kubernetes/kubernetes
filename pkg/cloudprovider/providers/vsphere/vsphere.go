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
	"errors"
	"fmt"
	"io"
	"net/url"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "vsphere"

// VSphere is an implementation of Interface, LoadBalancer and Instances for vSphere.
type VSphere struct {
	cfg *VSphereConfig
}

type VSphereConfig struct {
	Global struct {
		User         string `gcfg:"user"`
		Password     string `gcfg:"password"`
		VCenterIp    string `gcfg:"server"`
		VCenterPort  string `gcfg:"port"`
		InsecureFlag bool   `gcfg:"insecure-flag"`
		Datacenter   string `gcfg:"datacenter"`
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
	vs := VSphere{
		cfg: &cfg,
	}
	return &vs, nil
}

func vsphereLogin(cfg *VSphereConfig, ctx context.Context) (*govmomi.Client, error) {

	// Parse URL from string
	u, err := url.Parse(fmt.Sprintf("https://%s:%s/sdk", cfg.Global.VCenterIp, cfg.Global.VCenterPort))
	if err != nil {
		return nil, err
	}
	// set username and password for the URL
	u.User = url.UserPassword(cfg.Global.User, cfg.Global.Password)

	// Connect and log in to ESX or vCenter
	c, err := govmomi.NewClient(ctx, u, cfg.Global.InsecureFlag)
	if err != nil {
		return nil, err
	}

	return c, nil
}

type Instances struct {
	cfg *VSphereConfig
}

// Instances returns an implementation of Instances for vSphere.
func (vs *VSphere) Instances() (cloudprovider.Instances, bool) {
    
	return &Instances{vs.cfg}, true
}

func getInstances(c *govmomi.Client, finder *find.Finder, ctx context.Context, nameFilter string) ([]string, error) {
	//TODO: get all vms inside subfolders
	vms, err := finder.VirtualMachineList(ctx, nameFilter)
	if err != nil {
		return nil, err
	}

	var vmRef []types.ManagedObjectReference
	for _, vm := range vms {
		vmRef = append(vmRef, vm.Reference())
	}

	pc := property.DefaultCollector(c.Client)

	var vmt []mo.VirtualMachine
	err = pc.Retrieve(ctx, vmRef, []string{"name", "summary"}, &vmt)
	if err != nil {
		return nil, err
	}

	var vmList []string
	for _, vm := range vmt {
		vmPowerstate := vm.Summary.Runtime.PowerState
		if vmPowerstate == "poweredOn" {
			vmList = append(vmList, vm.Name)
		}
	}
	return vmList, nil
}

func (i *Instances) List(nameFilter string) ([]string, error) {

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c, err := vsphereLogin(i.cfg, ctx)
	if err != nil {
		fmt.Errorf("Failed to create vSpere client: %s", err)
	}
	defer c.Logout(ctx)

	fo := find.NewFinder(c.Client, true)
	dc, err := fo.Datacenter(ctx, i.cfg.Global.Datacenter)
	if err != nil {
		glog.Warningf("Failed to find %v", err)
		return nil, err
	}

	finderObj := fo.SetDatacenter(dc)
	vmList, err := getInstances(c, finderObj, ctx, nameFilter)
	if err != nil {
		return nil, err
	}

	glog.V(3).Infof("Found %v instances matching %v: %v",
		len(vmList), nameFilter, vmList)

	return vmList, nil
}

func (i *Instances) NodeAddresses(name string) ([]api.NodeAddress, error) {
	return nil, nil
}

func (i *Instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

func (i *Instances) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

func (i *Instances) ExternalID(name string) (string, error) {
	return "", nil
}

func (i *Instances) InstanceType(name string) (string, error) {
	return "", nil
}

func (i *Instances) InstanceID(name string) (string, error) {
	return "/" + " ", nil
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
	glog.V(1).Info("Claiming to support Zones")

	return vs, true
}

func (vs *VSphere) GetZone() (cloudprovider.Zone, error) {
	glog.V(1).Infof("Current zone is %v", vs.cfg.Global.Datacenter)

	return cloudprovider.Zone{Region: vs.cfg.Global.Datacenter}, nil
}

// Routes returns an implementation of Routes for vSphere.
func (vs *VSphere) Routes() (cloudprovider.Routes, bool) {
	return nil, true
}

// ScrubDNS filters DNS settings for pods.
func (vs *VSphere) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}
