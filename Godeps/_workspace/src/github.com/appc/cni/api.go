package cni

import (
	"strings"

	"github.com/appc/cni/pkg/invoke"
	"github.com/appc/cni/pkg/types"
)

type RuntimeConf struct {
	ContainerID string
	NetNS       string
	IfName      string
	Args        [][2]string
}

type NetworkConfig struct {
	types.NetConf
	Bytes []byte
}

type CNI interface {
	AddNetwork(net *NetworkConfig, rt *RuntimeConf) (*types.Result, error)
	DelNetwork(net *NetworkConfig, rt *RuntimeConf) error
}

type CNIConfig struct {
	Path []string
}

func (c *CNIConfig) AddNetwork(net *NetworkConfig, rt *RuntimeConf) (*types.Result, error) {
	return c.execPlugin("ADD", net, rt)
}

func (c *CNIConfig) DelNetwork(net *NetworkConfig, rt *RuntimeConf) error {
	_, err := c.execPlugin("DEL", net, rt)
	return err
}

// =====

func (c *CNIConfig) execPlugin(action string, conf *NetworkConfig, rt *RuntimeConf) (*types.Result, error) {
	pluginPath := invoke.FindInPath(conf.Type, c.Path)

	args := &invoke.Args{
		Command:     action,
		ContainerID: rt.ContainerID,
		NetNS:       rt.NetNS,
		PluginArgs:  rt.Args,
		IfName:      rt.IfName,
		Path:        strings.Join(c.Path, ":"),
	}
	return invoke.ExecPlugin(pluginPath, conf.Bytes, args)
}
