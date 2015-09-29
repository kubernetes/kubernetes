// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package libcontainer

import (
	"encoding/json"
	"io/ioutil"
	"path"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/configs"
	"github.com/google/cadvisor/utils"
)

// State represents a running container's state
type preAPIState struct {
	// InitPid is the init process id in the parent namespace
	InitPid int `json:"init_pid,omitempty"`

	// InitStartTime is the init process start time
	InitStartTime string `json:"init_start_time,omitempty"`

	// Network runtime state.
	NetworkState preAPINetworkState `json:"network_state,omitempty"`

	// Path to all the cgroups setup for a container. Key is cgroup subsystem name.
	CgroupPaths map[string]string `json:"cgroup_paths,omitempty"`
}

// Struct describing the network specific runtime state that will be maintained by libcontainer for all running containers
// Do not depend on it outside of libcontainer.
type preAPINetworkState struct {
	// The name of the veth interface on the Host.
	VethHost string `json:"veth_host,omitempty"`
	// The name of the veth interface created inside the container for the child.
	VethChild string `json:"veth_child,omitempty"`
	// Net namespace path.
	NsPath string `json:"ns_path,omitempty"`
}

type preAPIConfig struct {
	// Pathname to container's root filesystem
	RootFs string `json:"root_fs,omitempty"`

	// Hostname optionally sets the container's hostname if provided
	Hostname string `json:"hostname,omitempty"`

	// User will set the uid and gid of the executing process running inside the container
	User string `json:"user,omitempty"`

	// WorkingDir will change the processes current working directory inside the container's rootfs
	WorkingDir string `json:"working_dir,omitempty"`

	// Env will populate the processes environment with the provided values
	// Any values from the parent processes will be cleared before the values
	// provided in Env are provided to the process
	Env []string `json:"environment,omitempty"`

	// Tty when true will allocate a pty slave on the host for access by the container's process
	// and ensure that it is mounted inside the container's rootfs
	Tty bool `json:"tty,omitempty"`

	// Namespaces specifies the container's namespaces that it should setup when cloning the init process
	// If a namespace is not provided that namespace is shared from the container's parent process
	Namespaces []configs.Namespace `json:"namespaces,omitempty"`

	// Capabilities specify the capabilities to keep when executing the process inside the container
	// All capbilities not specified will be dropped from the processes capability mask
	Capabilities []string `json:"capabilities,omitempty"`

	// Networks specifies the container's network setup to be created
	Networks []preAPINetwork `json:"networks,omitempty"`

	// Routes can be specified to create entries in the route table as the container is started
	Routes []*configs.Route `json:"routes,omitempty"`

	// Cgroups specifies specific cgroup settings for the various subsystems that the container is
	// placed into to limit the resources the container has available
	Cgroups *configs.Cgroup `json:"cgroups,omitempty"`

	// AppArmorProfile specifies the profile to apply to the process running in the container and is
	// change at the time the process is execed
	AppArmorProfile string `json:"apparmor_profile,omitempty"`

	// ProcessLabel specifies the label to apply to the process running in the container.  It is
	// commonly used by selinux
	ProcessLabel string `json:"process_label,omitempty"`

	// RestrictSys will remount /proc/sys, /sys, and mask over sysrq-trigger as well as /proc/irq and
	// /proc/bus
	RestrictSys bool `json:"restrict_sys,omitempty"`
}

// Network defines configuration for a container's networking stack
//
// The network configuration can be omited from a container causing the
// container to be setup with the host's networking stack
type preAPINetwork struct {
	// Type sets the networks type, commonly veth and loopback
	Type string `json:"type,omitempty"`

	// The bridge to use.
	Bridge string `json:"bridge,omitempty"`

	// Prefix for the veth interfaces.
	VethPrefix string `json:"veth_prefix,omitempty"`

	// MacAddress contains the MAC address to set on the network interface
	MacAddress string `json:"mac_address,omitempty"`

	// Address contains the IPv4 and mask to set on the network interface
	Address string `json:"address,omitempty"`

	// IPv6Address contains the IPv6 and mask to set on the network interface
	IPv6Address string `json:"ipv6_address,omitempty"`

	// Gateway sets the gateway address that is used as the default for the interface
	Gateway string `json:"gateway,omitempty"`

	// IPv6Gateway sets the ipv6 gateway address that is used as the default for the interface
	IPv6Gateway string `json:"ipv6_gateway,omitempty"`

	// Mtu sets the mtu value for the interface and will be mirrored on both the host and
	// container's interfaces if a pair is created, specifically in the case of type veth
	// Note: This does not apply to loopback interfaces.
	Mtu int `json:"mtu,omitempty"`

	// TxQueueLen sets the tx_queuelen value for the interface and will be mirrored on both the host and
	// container's interfaces if a pair is created, specifically in the case of type veth
	// Note: This does not apply to loopback interfaces.
	TxQueueLen int `json:"txqueuelen,omitempty"`
}

// Relative path to the libcontainer execdriver directory.
const libcontainerExecDriverPath = "execdriver/native"

// TODO(vmarmol): Deprecate over time as old Dockers are phased out.
func ReadConfig(dockerRoot, dockerRun, containerID string) (*configs.Config, error) {
	// Try using the new config if it is available.
	configPath := configPath(dockerRun, containerID)
	if utils.FileExists(configPath) {
		out, err := ioutil.ReadFile(configPath)
		if err != nil {
			return nil, err
		}

		var state libcontainer.State
		err = json.Unmarshal(out, &state)
		if err != nil {
			return nil, err
		}
		return &state.Config, nil
	}

	// Fallback to reading the old config which is comprised of the state and config files.
	oldConfigPath := oldConfigPath(dockerRoot, containerID)
	out, err := ioutil.ReadFile(oldConfigPath)
	if err != nil {
		return nil, err
	}

	// Try reading the preAPIConfig.
	var config preAPIConfig
	err = json.Unmarshal(out, &config)
	if err != nil {
		// Try to parse the old pre-API config. The main difference is that namespaces used to be a map, now it is a slice of structs.
		// The JSON marshaler will use the non-nested field before the nested one.
		type oldLibcontainerConfig struct {
			preAPIConfig
			OldNamespaces map[string]bool `json:"namespaces,omitempty"`
		}
		var oldConfig oldLibcontainerConfig
		err2 := json.Unmarshal(out, &oldConfig)
		if err2 != nil {
			// Use original error.
			return nil, err
		}

		// Translate the old pre-API config into the new config.
		config = oldConfig.preAPIConfig
		for ns := range oldConfig.OldNamespaces {
			config.Namespaces = append(config.Namespaces, configs.Namespace{
				Type: configs.NamespaceType(ns),
			})
		}
	}

	// Read the old state file as well.
	state, err := readState(dockerRoot, containerID)
	if err != nil {
		return nil, err
	}

	// Convert preAPIConfig + old state file to Config.
	// This only converts some of the fields, the ones we use.
	// You may need to add fields if the one you're interested in is not available.
	var result configs.Config
	result.Cgroups = new(configs.Cgroup)
	result.Rootfs = config.RootFs
	result.Hostname = config.Hostname
	result.Namespaces = config.Namespaces
	result.Capabilities = config.Capabilities
	for _, net := range config.Networks {
		n := &configs.Network{
			Name:              state.NetworkState.VethChild,
			Bridge:            net.Bridge,
			MacAddress:        net.MacAddress,
			Address:           net.Address,
			Gateway:           net.Gateway,
			IPv6Address:       net.IPv6Address,
			IPv6Gateway:       net.IPv6Gateway,
			HostInterfaceName: state.NetworkState.VethHost,
		}
		result.Networks = append(result.Networks, n)
	}
	result.Routes = config.Routes
	if config.Cgroups != nil {
		result.Cgroups = config.Cgroups
	}

	return &result, nil
}

func readState(dockerRoot, containerID string) (preAPIState, error) {
	// pre-API libcontainer changed how its state was stored, try the old way of a "pid" file
	statePath := path.Join(dockerRoot, libcontainerExecDriverPath, containerID, "state.json")
	if !utils.FileExists(statePath) {
		pidPath := path.Join(dockerRoot, libcontainerExecDriverPath, containerID, "pid")
		if utils.FileExists(pidPath) {
			// We don't need the old state, return an empty state and we'll gracefully degrade.
			return preAPIState{}, nil
		}
	}
	out, err := ioutil.ReadFile(statePath)
	if err != nil {
		return preAPIState{}, err
	}

	// Parse the state.
	var state preAPIState
	err = json.Unmarshal(out, &state)
	if err != nil {
		return preAPIState{}, err
	}

	return state, nil
}

// Gets the path to the libcontainer configuration.
func configPath(dockerRun, containerID string) string {
	return path.Join(dockerRun, libcontainerExecDriverPath, containerID, "state.json")
}

// Gets the path to the old libcontainer configuration.
func oldConfigPath(dockerRoot, containerID string) string {
	return path.Join(dockerRoot, libcontainerExecDriverPath, containerID, "container.json")
}

// Gets whether the specified container exists.
func Exists(dockerRoot, dockerRun, containerID string) bool {
	// New or old config must exist for the container to be considered alive.
	return utils.FileExists(configPath(dockerRun, containerID)) || utils.FileExists(oldConfigPath(dockerRoot, containerID))
}
