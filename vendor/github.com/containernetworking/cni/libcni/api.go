// Copyright 2015 CNI authors
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

package libcni

import (
	"os"
	"strings"

	"github.com/containernetworking/cni/pkg/invoke"
	"github.com/containernetworking/cni/pkg/types"
	"github.com/containernetworking/cni/pkg/version"
)

type RuntimeConf struct {
	ContainerID string
	NetNS       string
	IfName      string
	Args        [][2]string
}

type NetworkConfig struct {
	Network *types.NetConf
	Bytes   []byte
}

type NetworkConfigList struct {
	Name       string
	CNIVersion string
	Plugins    []*NetworkConfig
	Bytes      []byte
}

type CNI interface {
	AddNetworkList(net *NetworkConfigList, rt *RuntimeConf) (types.Result, error)
	DelNetworkList(net *NetworkConfigList, rt *RuntimeConf) error

	AddNetwork(net *NetworkConfig, rt *RuntimeConf) (types.Result, error)
	DelNetwork(net *NetworkConfig, rt *RuntimeConf) error
}

type CNIConfig struct {
	Path []string
}

// CNIConfig implements the CNI interface
var _ CNI = &CNIConfig{}

func buildOneConfig(list *NetworkConfigList, orig *NetworkConfig, prevResult types.Result) (*NetworkConfig, error) {
	var err error

	// Ensure every config uses the same name and version
	orig, err = InjectConf(orig, "name", list.Name)
	if err != nil {
		return nil, err
	}
	orig, err = InjectConf(orig, "cniVersion", list.CNIVersion)
	if err != nil {
		return nil, err
	}

	// Add previous plugin result
	if prevResult != nil {
		orig, err = InjectConf(orig, "prevResult", prevResult)
		if err != nil {
			return nil, err
		}
	}

	return orig, nil
}

// AddNetworkList executes a sequence of plugins with the ADD command
func (c *CNIConfig) AddNetworkList(list *NetworkConfigList, rt *RuntimeConf) (types.Result, error) {
	var prevResult types.Result
	for _, net := range list.Plugins {
		pluginPath, err := invoke.FindInPath(net.Network.Type, c.Path)
		if err != nil {
			return nil, err
		}

		newConf, err := buildOneConfig(list, net, prevResult)
		if err != nil {
			return nil, err
		}

		prevResult, err = invoke.ExecPluginWithResult(pluginPath, newConf.Bytes, c.args("ADD", rt))
		if err != nil {
			return nil, err
		}
	}

	return prevResult, nil
}

// DelNetworkList executes a sequence of plugins with the DEL command
func (c *CNIConfig) DelNetworkList(list *NetworkConfigList, rt *RuntimeConf) error {
	for i := len(list.Plugins) - 1; i >= 0; i-- {
		net := list.Plugins[i]

		pluginPath, err := invoke.FindInPath(net.Network.Type, c.Path)
		if err != nil {
			return err
		}

		newConf, err := buildOneConfig(list, net, nil)
		if err != nil {
			return err
		}

		if err := invoke.ExecPluginWithoutResult(pluginPath, newConf.Bytes, c.args("DEL", rt)); err != nil {
			return err
		}
	}

	return nil
}

// AddNetwork executes the plugin with the ADD command
func (c *CNIConfig) AddNetwork(net *NetworkConfig, rt *RuntimeConf) (types.Result, error) {
	pluginPath, err := invoke.FindInPath(net.Network.Type, c.Path)
	if err != nil {
		return nil, err
	}

	return invoke.ExecPluginWithResult(pluginPath, net.Bytes, c.args("ADD", rt))
}

// DelNetwork executes the plugin with the DEL command
func (c *CNIConfig) DelNetwork(net *NetworkConfig, rt *RuntimeConf) error {
	pluginPath, err := invoke.FindInPath(net.Network.Type, c.Path)
	if err != nil {
		return err
	}

	return invoke.ExecPluginWithoutResult(pluginPath, net.Bytes, c.args("DEL", rt))
}

// GetVersionInfo reports which versions of the CNI spec are supported by
// the given plugin.
func (c *CNIConfig) GetVersionInfo(pluginType string) (version.PluginInfo, error) {
	pluginPath, err := invoke.FindInPath(pluginType, c.Path)
	if err != nil {
		return nil, err
	}

	return invoke.GetVersionInfo(pluginPath)
}

// =====
func (c *CNIConfig) args(action string, rt *RuntimeConf) *invoke.Args {
	return &invoke.Args{
		Command:     action,
		ContainerID: rt.ContainerID,
		NetNS:       rt.NetNS,
		PluginArgs:  rt.Args,
		IfName:      rt.IfName,
		Path:        strings.Join(c.Path, string(os.PathListSeparator)),
	}
}
