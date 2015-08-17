package cni

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/appc/cni/pkg/invoke"
	"github.com/appc/cni/pkg/types"
	"path/filepath"
	"sort"
)

type RuntimeConf struct {
	ContainerID string
	NetNS       string
	IfName      string
	Args        string
}

type NetworkConfig struct {
	types.NetConf
	Bytes []byte
}

func ConfFromFile(filename string) (*NetworkConfig, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading %s: %s", filename, err)
	}
	conf := &NetworkConfig{Bytes: bytes}
	if err = json.Unmarshal(bytes, conf); err != nil {
		return nil, fmt.Errorf("error parsing %s: %s", filename, err)
	}
	return conf, nil
}

func listConfFiles(dir string) ([]string, error) {
	// In part, from rkt/networking/podenv.go#listFiles
	files, err := ioutil.ReadDir(dir)
	switch {
	case err == nil: // break
	case os.IsNotExist(err):
		return nil, nil
	default:
		return nil, err
	}

	confFiles := []string{}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if filepath.Ext(f.Name()) == ".conf" {
			confFiles = append(confFiles, filepath.Join(dir, f.Name()))
		}
	}
	return confFiles, nil
}

func LoadNetConf(dir, name string) (*NetworkConfig, error) {
	files, err := listConfFiles(dir)
	switch {
	case err != nil:
		return nil, err
	case files == nil || len(files) == 0:
		return nil, fmt.Errorf("No net configurations found")
	}
	sort.Strings(files)

	for _, confFile := range files {
		conf, err := ConfFromFile(confFile)
		if err != nil {
			return nil, err
		}
		if conf.Name == name {
			return conf, nil
		}
	}
	return nil, fmt.Errorf(`no net configuration with name "%s" in %s`, name, dir)
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

// there's another in cni/pkg/plugin/ipam.go, but it assumes the
// environment variables are inherited from the current process
func (c *CNIConfig) execPlugin(action string, conf *NetworkConfig, rt *RuntimeConf) (*types.Result, error) {
	pluginPath := invoke.FindInPath(conf.Type, c.Path)

	vars := [][2]string{
		{"CNI_COMMAND", action},
		{"CNI_CONTAINERID", rt.ContainerID},
		{"CNI_NETNS", rt.NetNS},
		{"CNI_ARGS", rt.Args},
		{"CNI_IFNAME", rt.IfName},
		{"CNI_PATH", strings.Join(c.Path, ":")},
	}
	return invoke.ExecPlugin(pluginPath, conf.Bytes, envVars(vars))
}

// taken from rkt/networking/net_plugin.go
func envVars(vars [][2]string) []string {
	env := os.Environ()

	for _, kv := range vars {
		env = append(env, strings.Join(kv[:], "="))
	}

	return env
}
