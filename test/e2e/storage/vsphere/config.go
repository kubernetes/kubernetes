/*
Copyright 2018 The Kubernetes Authors.

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
	"gopkg.in/gcfg.v1"
	"io"
	"k8s.io/kubernetes/test/e2e/framework"
	"os"
)

const (
	vSphereConfFileEnvVar = "VSPHERE_CONF_FILE"
)

var (
	confFileLocation = os.Getenv(vSphereConfFileEnvVar)
)

// Config represents vSphere configuration
type Config struct {
	Username          string
	Password          string
	Hostname          string
	Port              string
	Datacenters       string
	RoundTripperCount uint
	DefaultDatastore  string
	Folder            string
}

// ConfigFile represents the content of vsphere.conf file.
// Users specify the configuration of one or more vSphere instances in vsphere.conf where
// the Kubernetes master and worker nodes are running.
type ConfigFile struct {
	Global struct {
		// vCenter username.
		User string `gcfg:"user"`
		// vCenter password in clear text.
		Password string `gcfg:"password"`
		// vCenter port.
		VCenterPort string `gcfg:"port"`
		// True if vCenter uses self-signed cert.
		InsecureFlag bool `gcfg:"insecure-flag"`
		// Datacenter in which VMs are located.
		Datacenters string `gcfg:"datacenters"`
		// Soap round tripper count (retries = RoundTripper - 1)
		RoundTripperCount uint `gcfg:"soap-roundtrip-count"`
	}

	VirtualCenter map[string]*Config

	Network struct {
		// PublicNetwork is name of the network the VMs are joined to.
		PublicNetwork string `gcfg:"public-network"`
	}

	Disk struct {
		// SCSIControllerType defines SCSI controller to be used.
		SCSIControllerType string `dcfg:"scsicontrollertype"`
	}

	// Endpoint used to create volumes
	Workspace struct {
		VCenterIP        string `gcfg:"server"`
		Datacenter       string `gcfg:"datacenter"`
		Folder           string `gcfg:"folder"`
		DefaultDatastore string `gcfg:"default-datastore"`
		ResourcePoolPath string `gcfg:"resourcepool-path"`
	}
}

// GetVSphereInstances parses vsphere.conf and returns VSphere instances
func GetVSphereInstances() (map[string]*VSphere, error) {
	cfg, err := getConfig()
	if err != nil {
		return nil, err
	}
	return populateInstanceMap(cfg)
}

func getConfig() (*ConfigFile, error) {
	if confFileLocation == "" {
		return nil, fmt.Errorf("Env variable 'VSPHERE_CONF_FILE' is not set.")
	}
	confFile, err := os.Open(confFileLocation)
	if err != nil {
		return nil, err
	}
	defer confFile.Close()
	cfg, err := readConfig(confFile)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

// readConfig parses vSphere cloud config file into ConfigFile.
func readConfig(config io.Reader) (ConfigFile, error) {
	if config == nil {
		err := fmt.Errorf("no vSphere cloud provider config file given")
		return ConfigFile{}, err
	}

	var cfg ConfigFile
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func populateInstanceMap(cfg *ConfigFile) (map[string]*VSphere, error) {
	vsphereInstances := make(map[string]*VSphere)

	if cfg.Workspace.VCenterIP == "" || cfg.Workspace.DefaultDatastore == "" || cfg.Workspace.Folder == "" || cfg.Workspace.Datacenter == "" {
		msg := fmt.Sprintf("All fields in workspace are mandatory."+
			" vsphere.conf does not have the workspace specified correctly. cfg.Workspace: %+v", cfg.Workspace)
		framework.Logf(msg)
		return nil, errors.New(msg)
	}
	for vcServer, vcConfig := range cfg.VirtualCenter {
		framework.Logf("Initializing vc server %s", vcServer)
		if vcServer == "" {
			framework.Logf("vsphere.conf does not have the VirtualCenter IP address specified")
			return nil, errors.New("vsphere.conf does not have the VirtualCenter IP address specified")
		}
		vcConfig.Hostname = vcServer

		if vcConfig.Username == "" {
			vcConfig.Username = cfg.Global.User
		}
		if vcConfig.Password == "" {
			vcConfig.Password = cfg.Global.Password
		}
		if vcConfig.Username == "" {
			msg := fmt.Sprintf("vcConfig.User is empty for vc %s!", vcServer)
			framework.Logf(msg)
			return nil, errors.New(msg)
		}
		if vcConfig.Password == "" {
			msg := fmt.Sprintf("vcConfig.Password is empty for vc %s!", vcServer)
			framework.Logf(msg)
			return nil, errors.New(msg)
		}
		if vcConfig.Port == "" {
			vcConfig.Port = cfg.Global.VCenterPort
		}
		if vcConfig.Datacenters == "" && cfg.Global.Datacenters != "" {
			vcConfig.Datacenters = cfg.Global.Datacenters
		}
		if vcConfig.RoundTripperCount == 0 {
			vcConfig.RoundTripperCount = cfg.Global.RoundTripperCount
		}

		vcConfig.DefaultDatastore = cfg.Workspace.DefaultDatastore
		vcConfig.Folder = cfg.Workspace.Folder

		vsphereIns := VSphere{
			Config: vcConfig,
		}
		vsphereInstances[vcServer] = &vsphereIns
	}

	framework.Logf("ConfigFile %v \n vSphere instances %v", cfg, vsphereInstances)
	return vsphereInstances, nil
}
