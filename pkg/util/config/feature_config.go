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
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

const (
	flagName = "feature-gate"
	// All known feature keys
	allAlpha = "enableAllAlpha"
)

// Default values for recorded features.
var featureDefaults = map[string]bool{
	allAlpha: false,
}

// FeatureGate parses and stores flag gates for known features from
// a ConfigurationMap.
type FeatureGate interface {
	AddFlag(fs *pflag.FlagSet)
	AlphaEnabled() bool
	// TODO: Define accessors for each non-API alpha feature.
}

// featureGate implements FeatureGate as well as pflag.Value
// for parsing from a flag using a ConfigurationMap.
type featureGate struct {
	configMap *ConfigurationMap
	features  map[string]bool
}

// NewFeatureGate creates a new FeatureGate. Caller should add flag
// to a flagset with AddFlag and parse flags before calling any of
// the accessors.
func NewFeatureGate() FeatureGate {
	m := make(ConfigurationMap)
	return &featureGate{
		configMap: &m,
	}
}

// Set, String, and Type implement pflag.Value

// Set Parses a string of the form // "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *featureGate) Set(value string) error {
	err := f.configMap.Set(value)
	if err != nil {
		return err
	}
	f.features = make(map[string]bool)
	for k := range *f.configMap {
		defaultValue, ok := featureDefaults[k]
		if !ok {
			return fmt.Errorf("unrecognized key: %s", k)
		}
		value, err := parseConfigMapEntry(*f.configMap, k, defaultValue)
		if err != nil {
			return err
		}
		f.features[k] = value
	}
	glog.Infof("feature config: %v", f.features)
	return nil
}

func (f *featureGate) String() string {
	return f.configMap.String()
}

func (f *featureGate) Type() string {
	return f.configMap.Type()
}

// AlphaEnabled returns value for allAlpha.
func (f *featureGate) AlphaEnabled() bool {
	defaultValue := featureDefaults[allAlpha]
	if f.features == nil {
		panic(fmt.Sprintf("--%s has not been parsed", flagName))
	}
	if v, ok := f.features[allAlpha]; ok {
		return v
	}
	return defaultValue
}

func parseConfigMapEntry(configMap ConfigurationMap, key string, defaultValue bool) (bool, error) {
	v, ok := configMap[key]
	if ok && v != "" {
		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return false, fmt.Errorf("invalid value of %s: %s, err: %v", key, v, err)
		}
		return boolValue, nil
	}
	return defaultValue, nil
}

// AddFlag adds a flag for setting global feature config to the
// specified FlagSet.
func (f *featureGate) AddFlag(fs *pflag.FlagSet) {
	var known []string
	for key, defaultValue := range featureDefaults {
		known = append(known, fmt.Sprintf("%s=true|false (default=%t)", key, defaultValue))
	}
	fs.Var(f, flagName, ""+
		"A set of key=value pairs that describe feature configuration for alpha/experimental features. "+
		"Keys include:\n"+strings.Join(known, "\n"))
}
