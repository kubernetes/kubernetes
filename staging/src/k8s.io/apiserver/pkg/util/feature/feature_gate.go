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
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

type Feature string
type Component string

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
	specialFeatures = map[Feature]func(known map[Feature]FeatureSpec, enabled map[Feature]bool, val bool){
		allAlphaGate: setUnsetAlphaGates,
	}

	// DefaultFeatureGate is a shared global FeatureGate.
	DefaultFeatureGate FeatureGate = NewFeatureGate()
)

// FeatureSpec describes a feature that can be used by many components
type FeatureSpec struct {
	// Default enables the feature by default
	Default bool
	// PreRelease set the release state of the feature
	PreRelease prerelease
	// Components list the components that is using the feature.
	// Feature with empty components will always be rendered by FeatureGate.
	Components []Component
}

// Equals checks if two FeatureSpec are equal
func (f FeatureSpec) Equals(other FeatureSpec) bool {
	sort.Slice(f.Components, func(i, j int) bool { return f.Components[i] < f.Components[j] })
	sort.Slice(other.Components, func(i, j int) bool { return other.Components[i] < other.Components[j] })
	return reflect.DeepEqual(f, other)
}

func (f FeatureSpec) hasComponent(component Component) bool {
	if len(f.Components) == 0 {
		return true
	}
	for _, fc := range f.Components {
		if fc == component {
			return true
		}
	}
	return false
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
	// AddFlag adds a flag for setting global feature gates of a component to the specified FlagSet.
	AddFlag(fs *pflag.FlagSet, component Component)
	// Set parses and stores flag gates for known features
	// from a string like feature1=true,feature2=false,...
	Set(value string) error
	// SetFromMap stores flag gates for known features from a map[string]bool or returns an error
	SetFromMap(m map[string]bool) error
	// Enabled returns true if the key is enabled.
	Enabled(key Feature) bool
	// Add adds features to the featureGate.
	Add(features map[Feature]FeatureSpec) error
	// KnownFeatures returns a slice of strings describing the FeatureGate's known features for a component.
	KnownFeatures(component Component) []string
}

// featureGate implements FeatureGate as well as pflag.Value for flag parsing.
type featureGate struct {
	special map[Feature]func(map[Feature]FeatureSpec, map[Feature]bool, bool)

	// lock guards writes to known, enabled, and reads/writes of closed
	lock sync.Mutex
	// known holds a map[Feature]FeatureSpec
	known *atomic.Value
	// enabled holds a map[Feature]bool
	enabled *atomic.Value
	// closed is set to true when AddFlag is called, and prevents subsequent calls to Add
	closed bool
}

func setUnsetAlphaGates(known map[Feature]FeatureSpec, enabled map[Feature]bool, val bool) {
	for k, v := range known {
		if v.PreRelease == Alpha {
			if _, found := enabled[k]; !found {
				enabled[k] = val
			}
		}
	}
}

// Set, String, and Type implement pflag.Value
var _ pflag.Value = &featureGate{}

func NewFeatureGate() *featureGate {
	known := map[Feature]FeatureSpec{}
	for k, v := range defaultFeatures {
		known[k] = v
	}

	knownValue := &atomic.Value{}
	knownValue.Store(known)

	enabled := map[Feature]bool{}
	enabledValue := &atomic.Value{}
	enabledValue.Store(enabled)

	f := &featureGate{
		known:   knownValue,
		special: specialFeatures,
		enabled: enabledValue,
	}
	return f
}

// Set parses a string of the form "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *featureGate) Set(value string) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	// Copy existing state
	known := map[Feature]FeatureSpec{}
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		known[k] = v
	}
	enabled := map[Feature]bool{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		enabled[k] = v
	}

	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		k := Feature(strings.TrimSpace(arr[0]))
		_, ok := known[k]
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
		enabled[k] = boolValue

		// Handle "special" features like "all alpha gates"
		if fn, found := f.special[k]; found {
			fn(known, enabled, boolValue)
		}
	}

	// Persist changes
	f.known.Store(known)
	f.enabled.Store(enabled)

	glog.Infof("feature gates: %v", enabled)
	return nil
}

// SetFromMap stores flag gates for known features from a map[string]bool or returns an error
func (f *featureGate) SetFromMap(m map[string]bool) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	// Copy existing state
	known := map[Feature]FeatureSpec{}
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		known[k] = v
	}
	enabled := map[Feature]bool{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		enabled[k] = v
	}

	for k, v := range m {
		k := Feature(k)
		_, ok := known[k]
		if !ok {
			return fmt.Errorf("unrecognized key: %s", k)
		}
		enabled[k] = v
		// Handle "special" features like "all alpha gates"
		if fn, found := f.special[k]; found {
			fn(known, enabled, v)
		}
	}

	// Persist changes
	f.known.Store(known)
	f.enabled.Store(enabled)

	glog.Infof("feature gates: %v", f.enabled)
	return nil
}

// String returns a string containing all enabled feature gates, formatted as "key1=value1,key2=value2,...".
func (f *featureGate) String() string {
	pairs := []string{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		pairs = append(pairs, fmt.Sprintf("%s=%t", k, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

func (f *featureGate) Type() string {
	return "mapStringBool"
}

// Add adds features to the featureGate.
func (f *featureGate) Add(features map[Feature]FeatureSpec) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		return fmt.Errorf("cannot add a feature gate after adding it to the flag set")
	}

	// Copy existing state
	known := map[Feature]FeatureSpec{}
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		known[k] = v
	}

	for name, spec := range features {
		if existingSpec, found := known[name]; found {
			if existingSpec.Equals(spec) {
				continue
			}
			return fmt.Errorf("feature gate %q with different spec already exists: %v", name, existingSpec)
		}

		known[name] = spec
	}

	// Persist updated state
	f.known.Store(known)

	return nil
}

// Enabled returns true if the key is enabled.
func (f *featureGate) Enabled(key Feature) bool {
	if v, ok := f.enabled.Load().(map[Feature]bool)[key]; ok {
		return v
	}
	return f.known.Load().(map[Feature]FeatureSpec)[key].Default
}

// AddFlag adds a flag for setting global feature gates to the specified FlagSet.
func (f *featureGate) AddFlag(fs *pflag.FlagSet, component Component) {
	f.lock.Lock()
	f.closed = true
	f.lock.Unlock()

	known := f.KnownFeatures(component)
	fs.Var(f, flagName, ""+
		"A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(known, "\n"))
}

// KnownFeatures returns a slice of strings describing the FeatureGate's known features.
// If component is empty, it should return all known features.
func (f *featureGate) KnownFeatures(component Component) []string {
	var known []string
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		if !v.hasComponent(component) {
			continue
		}
		pre := ""
		if v.PreRelease != GA {
			pre = fmt.Sprintf("%s - ", v.PreRelease)
		}
		known = append(known, fmt.Sprintf("%s=true|false (%sdefault=%t)", k, pre, v.Default))
	}
	sort.Strings(known)
	return known
}

// MustAddDefault adds the features to DefaultFeatureGate. Failing to add a
// feature will panic.
func MustAddDefault(features map[Feature]FeatureSpec) {
	if err := DefaultFeatureGate.Add(features); err != nil {
		panic(fmt.Sprintf("failed to add features: %v", err))
	}
}
