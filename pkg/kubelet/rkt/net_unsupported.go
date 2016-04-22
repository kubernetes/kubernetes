// +build !linux

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

	"k8s.io/kubernetes/pkg/kubelet/network"
)

const defaultNetworkName = "default"

type NetworkPluginConfig struct {
	NetworkPluginName string
	NetworkPluginDir  string
	NetworkPlugins    []network.NetworkPlugin
}

func (cfg *NetworkPluginConfig) init(c *Config) error {
	return fmt.Errorf("rkt is not supported in this build")
}

func (cfg *NetworkPluginConfig) networkName() string {
	return defaultNetworkName
}
