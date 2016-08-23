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
	"sort"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

const (
	flagName = "feature-gates"

	// All known feature keys
	// To add a new feature, define a key for it below and add
	// a featureInfo entry to knownFeatures.

	// allAlpha is a global toggle for alpha features. Per feature key
	// values override the default set by allAlpha.
	// e.g. allAlpha=false,newFeature=true will result in newFeature=true
	allAlpha = "allAlpha"
)

var (
	// DefaultFeatureGate is a shared global FeatureGate.
	DefaultFeatureGate = &featureGate{}

	// Default values for recorded features.
	knownFeatures = map[string]featureInfo{
		allAlpha: {false, alpha},
	}

	alpha = prerelease("ALPHA")
	beta  = prerelease("BETA")
	ga    = prerelease("")
)

type prerelease string

type featureInfo struct {
	enabled    bool
	prerelease prerelease
}

// FeatureGate parses and stores flag gates for known features from
// a string like feature1=true,feature2=false,...
type FeatureGate interface {
	AddFlag(fs *pflag.FlagSet)
	// owner: @jlowdermilk
	// alpha: v1.4
	AllAlpha() bool
	// TODO: Define accessors for each non-API alpha feature.
}

// featureGate implements FeatureGate as well as pflag.Value for flag parsing.
type featureGate struct {
	features map[string]bool
}

// Set, String, and Type implement pflag.Value

// Set Parses a string of the form // "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *featureGate) Set(value string) error {
	f.features = make(map[string]bool)
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		k := strings.TrimSpace(arr[0])
		_, ok := knownFeatures[k]
		if !ok {
			return fmt.Errorf("unrecognized key: %s", k)
		}
		if len(arr) != 2 {
			return fmt.Errorf("missing bool value for %s", k)
		}
		v := strings.TrimSpace(arr[1])
		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("invalid value of %s: %s, err: %v", k, v, err)
		}
		f.features[k] = boolValue
	}
	glog.Infof("feature gates: %v", f.features)
	return nil
}

func (f *featureGate) String() string {
	pairs := []string{}
	for k, v := range f.features {
		pairs = append(pairs, fmt.Sprintf("%s=%t", k, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

func (f *featureGate) Type() string {
	return "mapStringBool"
}

// AllAlpha returns value for allAlpha.
func (f *featureGate) AllAlpha() bool {
	return f.lookup(allAlpha)
}

func (f *featureGate) lookup(key string) bool {
	defaultValue := knownFeatures[key].enabled
	if f.features == nil {
		panic(fmt.Sprintf("--%s has not been parsed", flagName))
	}
	if v, ok := f.features[key]; ok {
		return v
	}
	return defaultValue

}

// AddFlag adds a flag for setting global feature gates to the specified FlagSet.
func (f *featureGate) AddFlag(fs *pflag.FlagSet) {
	var known []string
	for k, v := range knownFeatures {
		pre := ""
		if v.prerelease != ga {
			pre = fmt.Sprintf("%s - ", v.prerelease)
		}
		known = append(known, fmt.Sprintf("%s=true|false (%sdefault=%t)", k, pre, v.enabled))
	}
	fs.Var(f, flagName, ""+
		"A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(known, "\n"))
}
