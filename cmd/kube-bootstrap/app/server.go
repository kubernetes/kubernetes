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

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"k8s.io/kubernetes/pkg/bootstrap"

	"time"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// BootstrapServerConfig contains configures and runs a Kubernetes proxy server
type BootstrapServerConfig struct {
	ConfigFile string
}

type BootstrapServer struct {
	Config       *BootstrapServerConfig
	Bootstrapper *bootstrap.Bootstrapper
}

// AddFlags adds flags for a specific BootstrapServer to the specified FlagSet
func (s *BootstrapServerConfig) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.ConfigFile, "config", s.ConfigFile, "The path to the configuration file.")
}

func NewBootstrapServerConfig() *BootstrapServerConfig {
	return &BootstrapServerConfig{
		ConfigFile: "/etc/kubernetes/config",
	}
}

func NewBootstrapServer(config *BootstrapServerConfig, bootstrapper *bootstrap.Bootstrapper) (*BootstrapServer, error) {
	return &BootstrapServer{
		Config:       config,
		Bootstrapper: bootstrapper,
	}, nil
}

// NewBootstrapServerDefault creates a new BootstrapServer BootstrapServer with default parameters.
func NewBootstrapServerDefault(config *BootstrapServerConfig) (*BootstrapServer, error) {
	bootstrapper := bootstrap.NewBootstrapper(config.ConfigFile)
	return NewBootstrapServer(config, bootstrapper)
}

// Run runs the specified BootstrapServer, which bootstraps the machine.
// If it encounters an error, it will sleep and retry.
// Thus on exit the bootstrap has succeeded.
func (s *BootstrapServer) Run(_ []string) error {
	for {
		err := s.Bootstrapper.RunOnce()
		if err == nil {
			return nil
		}
		glog.Warningf("error during bootstrapping (will sleep and retry): %v", err)
		time.Sleep(10 * time.Second)
	}
}
