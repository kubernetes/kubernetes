// +build linux

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

package rkt

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/cni"
	"k8s.io/kubernetes/pkg/kubelet/network/kubenet"
)

const defaultNetworkName = "default"

type NetworkPluginConfig struct {
	NetworkPluginName string
	NetworkPluginDir  string
	NetworkPlugins    []network.NetworkPlugin
}

// init symlinks ${RKT_LOCAL_CONFIG}/net.d and ${RKT_LOCAL_CONFIG}/stage1/net.d
// to network plugin dir, so that the CNI/kubenet config files can be discoverd by rkt.
// If the plugin is not CNI or kubenet, that it returns an error.
func (cfg *NetworkPluginConfig) init(c *Config) error {
	switch cfg.NetworkPluginName {
	case cni.CNIPluginName, kubenet.KubenetPluginName:
	default:
		return fmt.Errorf("network plugin %q is not supported", cfg.NetworkPluginName)
	}

	if cfg.NetworkPluginDir == "" {
		return nil
	}

	// Older rkt version supports only ${LOCAL_CONFIG_DIR}/net.d.
	// Newer version supports both, and will deprecate the ${LOCAL_CONFIG_DIR}/net.d
	// in the future.
	// See https://github.com/coreos/rkt/pull/2312#issuecomment-210548103.
	netdirs := []string{
		filepath.Join(c.LocalConfigDir, "stage1", "net.d"),
		filepath.Join(c.LocalConfigDir, "net.d"),
	}

	succeeded := false
	for _, dir := range netdirs {
		if dir == cfg.NetworkPluginDir {
			// If the --network-plugin-dir is the net.d itself,
			// do nothing.
			continue
		}

		_, err := os.Lstat(dir)
		if err == nil {
			// If the net.d dir already exists, the original directory and all its contents
			// will be removed.
			// This is not ideal, the long term solution is to let rkt have the ability
			// to specify the network config directory.
			// See https://github.com/coreos/rkt/issues/2249#issuecomment-214528328
			if err = os.RemoveAll(dir); err != nil {
				glog.Errorf("rkt: Failed to remove directory %q: %v", dir, err)
				continue
			}
		} else if !os.IsNotExist(err) {
			glog.Errorf("rkt: Failed to lstat directory %q: %v", dir, err)
			continue
		}
		if err = os.Symlink(cfg.NetworkPluginDir, dir); err != nil {
			glog.Errorf("rkt: Failed to symlink: %v", err)
			continue
		}
		succeeded = true
	}

	if !succeeded {
		return fmt.Errorf("failed to prepare the net.d directory")
	}

	return nil
}

// networkName returns the network name that will be passed to
// rkt by '--net' if rkt runs in a private network.
// It will return 'kubenet' if the network plugin is kubenet.
// It will return the first network's name under '--network-plugin-dir' if
// the '--network-plugin' is cni.
func (cfg *NetworkPluginConfig) networkName() string {
	if len(cfg.NetworkPlugins) == 0 {
		return defaultNetworkName
	}

	switch cfg.NetworkPluginName {
	case kubenet.KubenetPluginName:
		return kubenet.KubenetPluginName
	case cni.CNIPluginName:
		return cfg.NetworkPlugins[0].(*cni.CNINetworkPlugin).DefaultNetworkName()
	}

	glog.Warningf("Network plugin not supported, using %q", defaultNetworkName)
	return defaultNetworkName
}
