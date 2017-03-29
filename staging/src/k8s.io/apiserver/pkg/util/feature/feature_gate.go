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

package feature

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

type Feature string

const (
	flagName = "feature-gates"

	// allAlphaGate is a global toggle for alpha features. Per-feature key
	// values override the default set by allAlphaGate. Examples:
	//   AllAlpha=false,NewFeature=true  will result in newFeature=true
	//   AllAlpha=true,NewFeature=false  will result in newFeature=false
	allAlphaGate Feature = "AllAlpha"
)

var (
	// The generic features.
	defaultFeatures = map[Feature]FeatureSpec{
		allAlphaGate: {Default: false, PreRelease: Alpha},
	}

	// Special handling for a few gates.
	specialFeatures = map[Feature]func(f *featureGate, val bool){
		allAlphaGate: setUnsetAlphaGates,
	}

	// DefaultFeatureGate is a shared global FeatureGate.
	DefaultFeatureGate FeatureGate = NewFeatureGate()
)

type FeatureSpec struct {
	Default    bool
	PreRelease prerelease
}

type prerelease string

const (
	// Values for PreRelease.
	Alpha = prerelease("ALPHA")
	Beta  = prerelease("BETA")
	GA    = prerelease("")
)

// FeatureGate parses and stores flag gates for known features from
// a string like feature1=true,feature2=false,...
type FeatureGate interface {
	AddFlag(fs *pflag.FlagSet)
	Set(value string) error
	Enabled(key Feature) bool
	Add(features map[Feature]FeatureSpec) error
	KnownFeatures() []string
}

// featureGate implements FeatureGate as well as pflag.Value for flag parsing.
type featureGate struct {
	known   map[Feature]FeatureSpec
	special map[Feature]func(*featureGate, bool)
	enabled map[Feature]bool

	// is set to true when AddFlag is called. Note: initialization is not go-routine safe, lookup is
	closed bool
}

func setUnsetAlphaGates(f *featureGate, val bool) {
	for k, v := range f.known {
		if v.PreRelease == Alpha {
			if _, found := f.enabled[k]; !found {
				f.enabled[k] = val
			}
		}
	}
}

// Set, String, and Type implement pflag.Value
var _ pflag.Value = &featureGate{}

func NewFeatureGate() *featureGate {
	f := &featureGate{
		known:   map[Feature]FeatureSpec{},
		special: specialFeatures,
		enabled: map[Feature]bool{},
	}
	for k, v := range defaultFeatures {
		f.known[k] = v
	}
	return f
}

// Set Parses a string of the form // "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *featureGate) Set(value string) error {
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		k := Feature(strings.TrimSpace(arr[0]))
		_, ok := f.known[Feature(k)]
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
		f.enabled[k] = boolValue

		// Handle "special" features like "all alpha gates"
		if fn, found := f.special[k]; found {
			fn(f, boolValue)
		}
	}

	glog.Infof("feature gates: %v", f.enabled)
	return nil
}

func (f *featureGate) String() string {
	pairs := []string{}
	for k, v := range f.enabled {
		pairs = append(pairs, fmt.Sprintf("%s=%t", k, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

func (f *featureGate) Type() string {
	return "mapStringBool"
}

func (f *featureGate) Add(features map[Feature]FeatureSpec) error {
	if f.closed {
		return fmt.Errorf("cannot add a feature gate after adding it to the flag set")
	}

	for name, spec := range features {
		if existingSpec, found := f.known[name]; found {
			if existingSpec == spec {
				continue
			}
			return fmt.Errorf("feature gate %q with different spec already exists: %v", name, existingSpec)
		}

		f.known[name] = spec
	}
	return nil
}

func (f *featureGate) Enabled(key Feature) bool {
	defaultValue := f.known[key].Default
	if f.enabled != nil {
		if v, ok := f.enabled[key]; ok {
			return v
		}
	}
	return defaultValue
}

// AddFlag adds a flag for setting global feature gates to the specified FlagSet.
func (f *featureGate) AddFlag(fs *pflag.FlagSet) {
	f.closed = true

	known := f.KnownFeatures()
	fs.Var(f, flagName, ""+
		"A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(known, "\n"))
}

// Returns a string describing the FeatureGate's known features.
func (f *featureGate) KnownFeatures() []string {
	var known []string
	for k, v := range f.known {
		pre := ""
		if v.PreRelease != GA {
			pre = fmt.Sprintf("%s - ", v.PreRelease)
		}
		known = append(known, fmt.Sprintf("%s=true|false (%sdefault=%t)", k, pre, v.Default))
	}
	sort.Strings(known)
	return known
}
