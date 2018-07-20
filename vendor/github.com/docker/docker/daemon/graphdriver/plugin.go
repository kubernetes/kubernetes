package graphdriver

import (
	"fmt"
	"path/filepath"

	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/plugin/v2"
)

func lookupPlugin(name string, pg plugingetter.PluginGetter, config Options) (Driver, error) {
	if !config.ExperimentalEnabled {
		return nil, fmt.Errorf("graphdriver plugins are only supported with experimental mode")
	}
	pl, err := pg.Get(name, "GraphDriver", plugingetter.Acquire)
	if err != nil {
		return nil, fmt.Errorf("Error looking up graphdriver plugin %s: %v", name, err)
	}
	return newPluginDriver(name, pl, config)
}

func newPluginDriver(name string, pl plugingetter.CompatPlugin, config Options) (Driver, error) {
	home := config.Root
	if !pl.IsV1() {
		if p, ok := pl.(*v2.Plugin); ok {
			if p.PropagatedMount != "" {
				home = p.PluginObj.Config.PropagatedMount
			}
		}
	}
	proxy := &graphDriverProxy{name, pl, Capabilities{}}
	return proxy, proxy.Init(filepath.Join(home, name), config.DriverOptions, config.UIDMaps, config.GIDMaps)
}
