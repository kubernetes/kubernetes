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
	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

type FeatureConfig interface {
	ConfigurationMap() ConfigurationMap
	AddFlag(fs *pflag.FlagSet)
	// TODO: Define accessors for each non-API alpha feature.
}

type featureConfig struct {
	configMap *ConfigurationMap
}

func NewFeatureConfig() FeatureConfig {
	m := make(ConfigurationMap)
	return &featureConfig{
		configMap: &m,
	}
}

func (f *featureConfig) Set(value string) error {
	err := f.configMap.Set(value)
	if err == nil {
		glog.Infof("feature config: %s", f.String())
	}
	return err
}

func (f *featureConfig) String() string {
	return f.configMap.String()
}

func (f *featureConfig) Type() string {
	return f.configMap.Type()
}

// ConfigurationMap returns a copy of the feature ConfigurationMap.
func (f *featureConfig) ConfigurationMap() ConfigurationMap {
	output := make(ConfigurationMap)
	m := f.configMap
	if m == nil {
		return nil
	}
	for k, v := range *m {
		output[k] = v
	}
	return output
}

// AddFlag adds a flag for setting global feature config to the
// specified FlagSet.
func (f *featureConfig) AddFlag(fs *pflag.FlagSet) {
	// TODO: List keys in usage string for each feature that has an accessor defined
	fs.Var(f, "feature-config", ""+
		"A set of key=value pairs that describe feature configuration for alpha/experimental features.")
}
