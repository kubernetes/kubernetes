/*
Copyright 2017 The Kubernetes Authors.

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
	"k8s.io/apiserver/pkg/server/resourceconfig"
	serverstore "k8s.io/apiserver/pkg/server/storage"
	cliflag "k8s.io/component-base/cli/flag"
)

// APIEnablementOptions contains the options for which resources to turn on and off.
// Given small aggregated API servers, this option isn't required for "normal" API servers
type APIEnablementOptions struct {
	RuntimeConfig cliflag.ConfigurationMap
}

func NewAPIEnablementOptions() *APIEnablementOptions {
	return &APIEnablementOptions{
		RuntimeConfig: make(cliflag.ConfigurationMap),
	}
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *APIEnablementOptions) AddFlags(fs *pflag.FlagSet) {
	fs.Var(&s.RuntimeConfig, "runtime-config", ""+
		"A set of key=value pairs that describe runtime configuration that may be passed "+
		"to apiserver. <group>/<version> (or <version> for the core group) key can be used to "+
		"turn on/off specific api versions. api/all is special key to control all api versions, "+
		"be careful setting it false, unless you know what you do. api/legacy is deprecated, "+
		"we will remove it in the future, so stop using it.")
}

// Validate validates RuntimeConfig with a list of registries.
// Usually this list only has one element, the apiserver registry of the process.
// But in the advanced (and usually not recommended) case of delegated apiservers there can be more.
// Validate will filter out the known groups of each registry.
// If anything is left over after that, an error is returned.
func (s *APIEnablementOptions) Validate(registries ...GroupRegisty) []error {
	if s == nil {
		return nil
	}

	errors := []error{}
	if s.RuntimeConfig["api/all"] == "false" && len(s.RuntimeConfig) == 1 {
		// Do not allow only set api/all=false, in such case apiserver startup has no meaning.
		return append(errors, fmt.Errorf("invalid key with only api/all=false"))
	}

	groups, err := resourceconfig.ParseGroups(s.RuntimeConfig)
	if err != nil {
		return append(errors, err)
	}

	for _, registry := range registries {
		// filter out known groups
		groups = unknownGroups(groups, registry)
	}
	if len(groups) != 0 {
		errors = append(errors, fmt.Errorf("unknown api groups %s", strings.Join(groups, ",")))
	}

	return errors
}

// ApplyTo override MergedResourceConfig with defaults and registry
func (s *APIEnablementOptions) ApplyTo(c *server.Config, defaultResourceConfig *serverstore.ResourceConfig, registry resourceconfig.GroupVersionRegistry) error {

	if s == nil {
		return nil
	}

	mergedResourceConfig, err := resourceconfig.MergeAPIResourceConfigs(defaultResourceConfig, s.RuntimeConfig, registry)
	c.MergedResourceConfig = mergedResourceConfig

	return err
}

func unknownGroups(groups []string, registry GroupRegisty) []string {
	unknownGroups := []string{}
	for _, group := range groups {
		if !registry.IsGroupRegistered(group) {
			unknownGroups = append(unknownGroups, group)
		}
	}
	return unknownGroups
}

// GroupRegisty provides a method to check whether given group is registered.
type GroupRegisty interface {
	// IsRegistered returns true if given group is registered.
	IsGroupRegistered(group string) bool
}
