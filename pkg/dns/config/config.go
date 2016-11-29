/*
Copyright 2016 The Kubernetes Authors.

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

package config

import (
	types "k8s.io/kubernetes/pkg/api/unversioned"
	fed "k8s.io/kubernetes/pkg/dns/federation"
)

// Config populated either from the configuration source (command
// line flags or via the config map mechanism).
type Config struct {
	// The inclusion of TypeMeta is to ensure future compatibility if the
	// Config object was populated directly via a Kubernetes API mechanism.
	//
	// For example, instead of the custom implementation here, the
	// configuration could be obtained from an API that unifies
	// command-line flags, config-map, etc mechanisms.
	types.TypeMeta

	// Map of federation names that the cluster in which this kube-dns
	// is running belongs to, to the corresponding domain names.
	Federations map[string]string `json:"federations"`
}

func NewDefaultConfig() *Config {
	return &Config{
		Federations: make(map[string]string),
	}
}

// IsValid returns whether or not the configuration is valid.
func (config *Config) Validate() error {
	if err := config.validateFederations(); err != nil {
		return err
	}

	return nil
}

func (config *Config) validateFederations() error {
	for name, domain := range config.Federations {
		if err := fed.ValidateName(name); err != nil {
			return err
		}
		if err := fed.ValidateDomain(domain); err != nil {
			return err
		}
	}
	return nil
}
