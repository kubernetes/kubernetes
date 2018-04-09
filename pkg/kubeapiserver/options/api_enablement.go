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
	"strings"

	"github.com/spf13/pflag"

	"k8s.io/apiserver/pkg/server"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/server/resourceconfig"
	serverstore "k8s.io/apiserver/pkg/server/storage"
)

// APIEnablementOptions holds the APIEnablement options.
// It is a wrap of generic APIEnablementOptions.
type APIEnablementOptions struct {
	GenericAPIEnablement *genericoptions.APIEnablementOptions
}

func NewAPIEnablementOptions() *APIEnablementOptions {
	return &APIEnablementOptions{
		GenericAPIEnablement: genericoptions.NewAPIEnablementOptions(),
	}
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *APIEnablementOptions) AddFlags(fs *pflag.FlagSet) {
	s.GenericAPIEnablement.AddFlags(fs)
}

// Validate verifies flags passed to kube-apiserver APIEnablementOptions.
// It calls GenericAPIEnablement.Validate.
func (s *APIEnablementOptions) Validate(registries ...genericoptions.GroupRegisty) []error {
	if s == nil {
		return nil
	}

	errors := s.GenericAPIEnablement.Validate(registries...)

	return errors
}

// ApplyTo override MergedResourceConfig with defaults and registry
func (s *APIEnablementOptions) ApplyTo(c *server.Config, defaultResourceConfig *serverstore.ResourceConfig, registry resourceconfig.GroupVersionRegistry) error {
	if s == nil {
		return nil
	}

	err := s.GenericAPIEnablement.ApplyTo(c, defaultResourceConfig, registry)

	return err
}

// ConvertLegacyGroup remove useless "api/legacy" and convert the legacy group "v1".
func (s *APIEnablementOptions) ConvertLegacyGroup() {
	if s == nil || s.GenericAPIEnablement == nil {
		return
	}
	overrides := s.GenericAPIEnablement.RuntimeConfig
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
