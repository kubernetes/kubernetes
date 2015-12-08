/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package bootstrap

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
)

// Bootstrapper is the main bootstrapping manager class
type Bootstrapper struct {
	ConfigFile      string
	config          Configuration
	cloudProvider   cloudprovider.Interface
	masterBootstrap cloudprovider.MasterBootstrap
}

// NewBootstrapper creates a Bootstrapper instance.
func NewBootstrapper(configFile string) *Bootstrapper {
	b := &Bootstrapper{ConfigFile: configFile}
	return b
}

// RunOnce is the main entry point for Bootstrapper; it performs all configuration
func (b *Bootstrapper) RunOnce() error {
	err := b.readConfig()
	if err != nil {
		return fmt.Errorf("unable to read configuration: %v", err)
	}

	err = b.buildCloudProvider()
	if err != nil {
		return fmt.Errorf("unable to build cloud provider: %v", err)
	}

	// The master volume acts as a mutex, so we always mount it first
	err = b.mountMasterVolume()
	if err != nil {
		return fmt.Errorf("unable to mount master volume: %v", err)
	}

	err = b.configureMasterPublicIP()
	if err != nil {
		return fmt.Errorf("unable to associate master IP: %v", err)
	}

	err = b.configureMasterPrivateIP()
	if err != nil {
		return fmt.Errorf("unable to associate master IP: %v", err)
	}

	err = b.configureMasterRoute()
	if err != nil {
		return fmt.Errorf("unable to configure master route: %v", err)
	}

	// All is well
	glog.Info("Bootstrapping completed successfully")
	return nil
}

func (b *Bootstrapper) readConfig() error {
	configBytes, err := ioutil.ReadFile(b.ConfigFile)
	if err != nil {
		return fmt.Errorf("error reading config file (%s): %v", b.ConfigFile, err)
	}

	err = json.Unmarshal(configBytes, &b.config)
	if err != nil {
		return fmt.Errorf("error parsing config file (%s): %v", b.ConfigFile, err)
	}

	return nil
}

func (b *Bootstrapper) buildCloudProvider() error {
	cloudProvider, err := cloudprovider.GetCloudProvider(b.config.CloudProvider, strings.NewReader(b.config.CloudProviderConfig))
	if err != nil {
		return fmt.Errorf("error initializing cloud provider: %v", err)
	}
	if cloudProvider == nil {
		return fmt.Errorf("cloud provider not found (%s)", b.config.CloudProvider)
	}

	masterBootstrap, ok := cloudProvider.MasterBootstrap()
	if !ok {
		return fmt.Errorf("cloud provider does not support master bootstrap (%s)", b.config.CloudProvider)
	}
	b.cloudProvider = cloudProvider
	b.masterBootstrap = masterBootstrap

	return nil
}

func (b *Bootstrapper) configureMasterPublicIP() error {
	ipString := b.config.MasterPublicIP

	ip := net.ParseIP(ipString)
	if ip == nil {
		return fmt.Errorf("error parsing public IP (%q)", ipString)
	}

	err := b.masterBootstrap.AttachPublicIP(ip)
	if err != nil {
		// TODO: This is retryable
		return fmt.Errorf("error attaching public IP (%q): %v", ipString, err)
	}

	return nil
}

func (b *Bootstrapper) configureMasterPrivateIP() error {
	ipString := b.config.MasterPrivateIP

	ip, ipNet, err := net.ParseCIDR(ipString)
	if err != nil {
		return fmt.Errorf("error parsing private IP CIDR (%q): %v", ipString, err)
	}

	err = b.masterBootstrap.AttachPrivateIP(ip, ipNet)
	if err != nil {
		// TODO: This is retryable
		return fmt.Errorf("error attaching private IP (%q): %v", ipString, err)
	}

	return nil
}

func (b *Bootstrapper) mountMasterVolume() error {
	// We could discover the volume using metadata (at least on AWS),
	// but it is probably easier just to rely on it being present in config
	volumeID := b.config.MasterVolume

	device, err := b.masterBootstrap.AttachMasterVolume(volumeID)
	if err != nil {
		// TODO: This is retryable
		return fmt.Errorf("error attaching volume (%s): %v", volumeID, err)
	}

	mounter := mount.New()
	diskMounter := &mount.SafeFormatAndMount{mounter, exec.New()}

	target := "/mnt/master-pd"
	err = os.MkdirAll(target, 0755)
	if err != nil {
		return fmt.Errorf("error creating directory %q: %v", target, err)
	}

	fsType := "ext4"
	options := []string{"noatime"}
	err = diskMounter.Mount(device, target, fsType, options)
	if err != nil {
		// TODO: This is retryable (?)
		return fmt.Errorf("error mounting volume (%s): %v", volumeID, err)
	}

	return nil
}

func (b *Bootstrapper) configureMasterRoute() error {
	// Once we have it insert our entry into the routing table
	routes, ok := b.cloudProvider.Routes()
	if !ok {
		// TODO: Should this just be a warning?
		return fmt.Errorf("cloudprovider %s does not support routes", b.config.CloudProvider)
	}

	// TODO: Is there a better way to get "my name"?
	instances, ok := b.cloudProvider.Instances()
	if !ok {
		return fmt.Errorf("cloudprovider %s does not support instance info", b.config.CloudProvider)
	}

	hostname, err := os.Hostname()
	if err != nil {
		return fmt.Errorf("error getting hostname: %v", err)
	}

	myNodeName, err := instances.CurrentNodeName(hostname)
	if err != nil {
		return fmt.Errorf("unable to determine current node name: %v", err)
	}

	route := &cloudprovider.Route{
		TargetInstance:  myNodeName,
		DestinationCIDR: b.config.MasterCIDR,
	}

	nameHint := "" // TODO: master?
	err = routes.CreateRoute(b.config.ClusterID, nameHint, route)
	if err != nil {
		return fmt.Errorf("error creating route: %v", err)
	}

	return nil
}
