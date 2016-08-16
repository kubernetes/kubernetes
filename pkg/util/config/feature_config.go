/*
Copyright 2014 The Kubernetes Authors.

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

var globalFeatureConfig featureConfig

type featureConfig struct {
	m *ConfigurationMap
}

func (f *featureConfig) Set(value string) error {
	*f.m = make(ConfigurationMap)
	err := f.m.Set(value)
	if err == nil {
		glog.Infof("feature config: %s", f.String())
	}
	return err
}

func (f *featureConfig) String() string {
	return f.m.String()
}

func (f *featureConfig) Type() string {
	return f.m.Type()
}

// FeatureConfig returns the global feature config map.
func FeatureConfig() ConfigurationMap {
	output := make(ConfigurationMap)
	m := globalFeatureConfig.m
	if m == nil {
		return nil
	}
	for k, v := range *m {
		output[k] = v
	}
	return output
}

// AddFeatureConfigFlag adds a flag for setting global feature config to the
// specified FlagSet.
func AddFeatureConfigFlag(fs *pflag.FlagSet) {
	fs.Var(&globalFeatureConfig, "feature-config", ""+
		"A set of key=value pairs that describe feature configuration for alpha/experimental features.")
}
