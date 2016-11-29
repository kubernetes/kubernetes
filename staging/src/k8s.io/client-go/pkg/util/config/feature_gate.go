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
	// a featureSpec entry to knownFeatures.

	// allAlphaGate is a global toggle for alpha features. Per-feature key
	// values override the default set by allAlphaGate. Examples:
	//   AllAlpha=false,NewFeature=true  will result in newFeature=true
	//   AllAlpha=true,NewFeature=false  will result in newFeature=false
	allAlphaGate              = "AllAlpha"
	externalTrafficLocalOnly  = "AllowExtTrafficLocalEndpoints"
	appArmor                  = "AppArmor"
	dynamicKubeletConfig      = "DynamicKubeletConfig"
	dynamicVolumeProvisioning = "DynamicVolumeProvisioning"
	streamingProxyRedirects   = "StreamingProxyRedirects"

	// experimentalHostUserNamespaceDefaulting Default userns=host for containers
	// that are using other host namespaces, host mounts, the pod contains a privileged container,
	// or specific non-namespaced capabilities
	// (MKNOD, SYS_MODULE, SYS_TIME). This should only be enabled if user namespace remapping is enabled
	// in the docker daemon.
	experimentalHostUserNamespaceDefaultingGate = "ExperimentalHostUserNamespaceDefaulting"
)

var (
	// Default values for recorded features.  Every new feature gate should be
	// represented here.
	knownFeatures = map[string]featureSpec{
		allAlphaGate:                                {false, alpha},
		externalTrafficLocalOnly:                    {true, beta},
		appArmor:                                    {true, beta},
		dynamicKubeletConfig:                        {false, alpha},
		dynamicVolumeProvisioning:                   {true, alpha},
		streamingProxyRedirects:                     {false, alpha},
		experimentalHostUserNamespaceDefaultingGate: {false, alpha},
	}

	// Special handling for a few gates.
	specialFeatures = map[string]func(f *featureGate, val bool){
		allAlphaGate: setUnsetAlphaGates,
	}

	// DefaultFeatureGate is a shared global FeatureGate.
	DefaultFeatureGate = &featureGate{
		known:   knownFeatures,
		special: specialFeatures,
	}
)

type featureSpec struct {
	enabled    bool
	prerelease prerelease
}

type prerelease string

const (
	// Values for prerelease.
	alpha = prerelease("ALPHA")
	beta  = prerelease("BETA")
	ga    = prerelease("")
)

// FeatureGate parses and stores flag gates for known features from
// a string like feature1=true,feature2=false,...
type FeatureGate interface {
	AddFlag(fs *pflag.FlagSet)
	Set(value string) error
	KnownFeatures() []string

	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.4
	// MyFeature() bool

	// owner: @timstclair
	// beta: v1.4
	AppArmor() bool

	// owner: @girishkalele
	// alpha: v1.4
	ExternalTrafficLocalOnly() bool

	// owner: @saad-ali
	// alpha: v1.3
	DynamicVolumeProvisioning() bool

	// owner: @mtaufen
	// alpha: v1.4
	DynamicKubeletConfig() bool

	// owner: timstclair
	// alpha: v1.5
	StreamingProxyRedirects() bool

	// owner: @pweil-
	// alpha: v1.5
	ExperimentalHostUserNamespaceDefaulting() bool
}

// featureGate implements FeatureGate as well as pflag.Value for flag parsing.
type featureGate struct {
	known   map[string]featureSpec
	special map[string]func(*featureGate, bool)
	enabled map[string]bool
}

func setUnsetAlphaGates(f *featureGate, val bool) {
	for k, v := range f.known {
		if v.prerelease == alpha {
			if _, found := f.enabled[k]; !found {
				f.enabled[k] = val
			}
		}
	}
}

// Set, String, and Type implement pflag.Value

// Set Parses a string of the form // "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *featureGate) Set(value string) error {
	f.enabled = make(map[string]bool)
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		k := strings.TrimSpace(arr[0])
		_, ok := f.known[k]
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

// ExternalTrafficLocalOnly returns value for AllowExtTrafficLocalEndpoints
func (f *featureGate) ExternalTrafficLocalOnly() bool {
	return f.lookup(externalTrafficLocalOnly)
}

// AppArmor returns the value for the AppArmor feature gate.
func (f *featureGate) AppArmor() bool {
	return f.lookup(appArmor)
}

// DynamicKubeletConfig returns value for dynamicKubeletConfig
func (f *featureGate) DynamicKubeletConfig() bool {
	return f.lookup(dynamicKubeletConfig)
}

// DynamicVolumeProvisioning returns value for dynamicVolumeProvisioning
func (f *featureGate) DynamicVolumeProvisioning() bool {
	return f.lookup(dynamicVolumeProvisioning)
}

// StreamingProxyRedirects controls whether the apiserver should intercept (and follow)
// redirects from the backend (Kubelet) for streaming requests (exec/attach/port-forward).
func (f *featureGate) StreamingProxyRedirects() bool {
	return f.lookup(streamingProxyRedirects)
}

// ExperimentalHostUserNamespaceDefaulting returns value for experimentalHostUserNamespaceDefaulting
func (f *featureGate) ExperimentalHostUserNamespaceDefaulting() bool {
	return f.lookup(experimentalHostUserNamespaceDefaultingGate)
}

func (f *featureGate) lookup(key string) bool {
	defaultValue := f.known[key].enabled
	if f.enabled != nil {
		if v, ok := f.enabled[key]; ok {
			return v
		}
	}
	return defaultValue

}

// AddFlag adds a flag for setting global feature gates to the specified FlagSet.
func (f *featureGate) AddFlag(fs *pflag.FlagSet) {
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
		if v.prerelease != ga {
			pre = fmt.Sprintf("%s - ", v.prerelease)
		}
		known = append(known, fmt.Sprintf("%s=true|false (%sdefault=%t)", k, pre, v.enabled))
	}
	sort.Strings(known)
	return known
}
