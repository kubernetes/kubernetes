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

package options

import (
	"fmt"
	"strings"

	"github.com/spf13/pflag"

	"k8s.io/apiserver/pkg/server"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/server/resourceconfig"
	serverstore "k8s.io/apiserver/pkg/server/storage"
	utilflag "k8s.io/apiserver/pkg/util/flag"
	"k8s.io/kubernetes/pkg/kubeapiserver"
)

// APIEnablementOptions contains the options for enabling or disabling group/version
// and group/version/resource. The latter is added here compared to the generic API server
// APIEnablementOptions.
type APIEnablementOptions struct {
	RuntimeConfig utilflag.ConfigurationMap
}

func NewAPIEnablementOptions() *APIEnablementOptions {
	return &APIEnablementOptions{
		RuntimeConfig: make(utilflag.ConfigurationMap),
	}
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *APIEnablementOptions) AddFlags(fs *pflag.FlagSet) {
	fs.Var(&s.RuntimeConfig, "runtime-config", ""+
		"A set of key=value pairs that describe runtime configuration that may be passed "+
		"to apiserver. <group>/<version> (or <version> for the core group) key can be used to "+
		"turn on/off specific api versions. <group>/<version>/<resource> (or <version>/<resource> "+
		"for the core group) can be used to turn on/off specific resources. api/all is special key"+
		"to control all api versions, be careful setting it false, unless you know what you do. "+
		"api/legacy is deprecated, we will remove it in the future, so stop using it.")
}

// Validate validates RuntimeConfig with a list of APIServers' registries.
// Validate will filter out the known groups of each registry.
// If anything is left over after that, an error is returned.
func (s *APIEnablementOptions) Validate(registries ...genericoptions.GroupRegisty) []error {
	if s == nil {
		return nil
	}

	errors := []error{}
	legacyRuntimeConfig := make(utilflag.ConfigurationMap)
	genericRuntimeConfig := make(utilflag.ConfigurationMap)
	for key := range s.RuntimeConfig {
		tokens := strings.Split(key, "/")
		switch len(tokens) {
		case 2:
			genericRuntimeConfig[key] = s.RuntimeConfig[key]
		case 3:
			legacyRuntimeConfig[key] = s.RuntimeConfig[key]
		default:
			return append(errors, fmt.Errorf("invliad key %s", key))
		}
	}

	groups, err := resourceconfig.ParseGroups(legacyRuntimeConfig)
	if err != nil {
		return append(errors, err)
	}

	for _, registry := range registries {
		// filter out known groups
		groups = genericoptions.UnknownGroups(groups, registry)
	}
	if len(groups) != 0 {
		errors = append(errors, fmt.Errorf("unknown api groups %s", strings.Join(groups, ",")))
	}

	// make use of generic APIServer's APIEnablementOptions validate
	genericAPIEnablementOptions := genericoptions.APIEnablementOptions{RuntimeConfig: genericRuntimeConfig}
	errors = genericAPIEnablementOptions.Validate(registries...)

	return errors
}

// ApplyTo override MergedResourceConfig with defaults and registry
func (s *APIEnablementOptions) ApplyTo(c *server.Config, defaultResourceConfig *serverstore.ResourceConfig, registry resourceconfig.GroupVersionRegistry) error {
	if s == nil {
		return nil
	}

	mergedResourceConfig, err := kubeapiserver.MergeAPIResourceConfigs(defaultResourceConfig, s.RuntimeConfig, registry)
	c.MergedResourceConfig = mergedResourceConfig

	return err
}

// ConvertLegacyGroup remove useless "api/legacy" and convert the core group version "v1".
func (s *APIEnablementOptions) ConvertLegacyGroup() {
	if s == nil {
		return
	}

	overrides := s.RuntimeConfig
	// remove api/legacy, it is useless.
	delete(overrides, "api/legacy")

	for key := range overrides {
		// HACK: Hack for "v1" legacy group version.
		// Remove when we stop supporting the legacy group version.
		if key == "v1" || strings.HasPrefix(key, "v1/") {
			overrides["/"+key] = overrides[key]
			delete(overrides, key)
			continue
		}

		if key == "api/v1" {
			overrides["/v1"] = overrides[key]
			delete(overrides, key)
			continue
		}

		if strings.HasPrefix(key, "api/v1/") {
			tokens := strings.Split(key, "/")
			if len(tokens) == 3 {
				overrides["/v1"+"/"+tokens[2]] = overrides[key]
				delete(overrides, key)
			}
		}
	}
}
