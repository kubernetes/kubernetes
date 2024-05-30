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

package featuregate

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/spf13/pflag"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/naming"
	"k8s.io/apimachinery/pkg/util/version"
	featuremetrics "k8s.io/component-base/metrics/prometheus/feature"
	baseversion "k8s.io/component-base/version"
	"k8s.io/klog/v2"
)

type Feature string

const (
	flagName = "feature-gates"

	// allAlphaGate is a global toggle for alpha features. Per-feature key
	// values override the default set by allAlphaGate. Examples:
	//   AllAlpha=false,NewFeature=true  will result in newFeature=true
	//   AllAlpha=true,NewFeature=false  will result in newFeature=false
	allAlphaGate Feature = "AllAlpha"

	// allBetaGate is a global toggle for beta features. Per-feature key
	// values override the default set by allBetaGate. Examples:
	//   AllBeta=false,NewFeature=true  will result in NewFeature=true
	//   AllBeta=true,NewFeature=false  will result in NewFeature=false
	allBetaGate Feature = "AllBeta"
)

var (
	// The generic features.
	defaultFeatures = map[Feature]VersionedSpecs{
		allAlphaGate: {{Default: false, PreRelease: Alpha, Version: version.MajorMinor(0, 0)}},
		allBetaGate:  {{Default: false, PreRelease: Beta, Version: version.MajorMinor(0, 0)}},
	}

	// Special handling for a few gates.
	specialFeatures = map[Feature]func(known map[Feature]VersionedSpecs, enabled map[Feature]bool, val bool, cVer *version.Version){
		allAlphaGate: setUnsetAlphaGates,
		allBetaGate:  setUnsetBetaGates,
	}
)

type FeatureSpec struct {
	// Default is the default enablement state for the feature
	Default bool
	// LockToDefault indicates that the feature is locked to its default and cannot be changed
	LockToDefault bool
	// PreRelease indicates the current maturity level of the feature
	PreRelease prerelease
	// Version indicates the earliest version from which this FeatureSpec is valid.
	// If multiple FeatureSpecs exist for a Feature, the one with the highest version that is less
	// than or equal to the effective version of the component is used.
	Version *version.Version
}

type VersionedSpecs []FeatureSpec

func (g VersionedSpecs) Len() int { return len(g) }
func (g VersionedSpecs) Less(i, j int) bool {
	return g[i].Version.LessThan(g[j].Version)
}
func (g VersionedSpecs) Swap(i, j int) { g[i], g[j] = g[j], g[i] }

type PromotionVersionMapping map[prerelease]string

type prerelease string

const (
	PreAlpha = prerelease("PRE-ALPHA")
	// Values for PreRelease.
	Alpha = prerelease("ALPHA")
	Beta  = prerelease("BETA")
	GA    = prerelease("")

	// Deprecated
	Deprecated = prerelease("DEPRECATED")
)

// FeatureGate indicates whether a given feature is enabled or not
type FeatureGate interface {
	// Enabled returns true if the key is enabled.
	Enabled(key Feature) bool
	// KnownFeatures returns a slice of strings describing the FeatureGate's known features.
	KnownFeatures() []string
	// DeepCopy returns a deep copy of the FeatureGate object, such that gates can be
	// set on the copy without mutating the original. This is useful for validating
	// config against potential feature gate changes before committing those changes.
	DeepCopy() MutableVersionedFeatureGate
	// Validate checks if the flag gates are valid at the emulated version.
	Validate() []error
}

// MutableFeatureGate parses and stores flag gates for known features from
// a string like feature1=true,feature2=false,...
type MutableFeatureGate interface {
	FeatureGate

	// AddFlag adds a flag for setting global feature gates to the specified FlagSet.
	AddFlag(fs *pflag.FlagSet)
	// Close sets closed to true, and prevents subsequent calls to Add
	Close()
	// Set parses and stores flag gates for known features
	// from a string like feature1=true,feature2=false,...
	Set(value string) error
	// SetFromMap stores flag gates for known features from a map[string]bool or returns an error
	SetFromMap(m map[string]bool) error
	// Add adds features to the featureGate.
	Add(features map[Feature]FeatureSpec) error
	// GetAll returns a copy of the map of known feature names to feature specs.
	GetAll() map[Feature]FeatureSpec
	// AddMetrics adds feature enablement metrics
	AddMetrics()
	// OverrideDefault sets a local override for the registered default value of a named
	// feature. If the feature has not been previously registered (e.g. by a call to Add), has a
	// locked default, or if the gate has already registered itself with a FlagSet, a non-nil
	// error is returned.
	//
	// When two or more components consume a common feature, one component can override its
	// default at runtime in order to adopt new defaults before or after the other
	// components. For example, a new feature can be evaluated with a limited blast radius by
	// overriding its default to true for a limited number of components without simultaneously
	// changing its default for all consuming components.
	OverrideDefault(name Feature, override bool) error
}

// MutableVersionedFeatureGate parses and stores flag gates for known features from
// a string like feature1=true,feature2=false,...
// MutableVersionedFeatureGate sets options based on the emulated version of the featured gate.
type MutableVersionedFeatureGate interface {
	MutableFeatureGate
	// EmulationVersion returns the version the feature gate is set to emulate.
	// If set, the feature gate would enable/disable features based on
	// feature availability and pre-release at the emulated version instead of the binary version.
	EmulationVersion() *version.Version
	// SetEmulationVersion overrides the emulationVersion of the feature gate.
	// Otherwise, the emulationVersion will be the same as the binary version.
	// If set, the feature defaults and availability will be as if the binary is at the emulated version.
	SetEmulationVersion(emulationVersion *version.Version) error
	// GetAll returns a copy of the map of known feature names to versioned feature specs.
	GetAllVersioned() map[Feature]VersionedSpecs
	// AddVersioned adds versioned feature specs to the featureGate.
	AddVersioned(features map[Feature]VersionedSpecs) error
}

// MutableVersionedFeatureGateForTests is a feature gate interface that should only be used in tests.
type MutableVersionedFeatureGateForTests interface {
	MutableVersionedFeatureGate
	// Reset sets the enabled and enabledRaw to the input map.
	Reset(m map[string]bool)
	// EnabledRawMap returns the raw enable map from the feature gate.
	EnabledRawMap() map[string]bool
}

// featureGate implements FeatureGate as well as pflag.Value for flag parsing.
type featureGate struct {
	featureGateName string

	special map[Feature]func(map[Feature]VersionedSpecs, map[Feature]bool, bool, *version.Version)

	// lock guards writes to all below fields.
	lock sync.Mutex
	// known holds a map[Feature]FeatureSpec
	known atomic.Value
	// enabled holds a map[Feature]bool
	enabled atomic.Value
	// enabledRaw holds a raw map[string]bool of the parsed flag.
	// It keeps the original values of "special" features like "all alpha gates",
	// while enabled keeps the values of all resolved features.
	enabledRaw atomic.Value
	// closed is set to true when AddFlag is called, and prevents subsequent calls to Add
	closed           bool
	emulationVersion atomic.Pointer[version.Version]
}

func setUnsetAlphaGates(known map[Feature]VersionedSpecs, enabled map[Feature]bool, val bool, cVer *version.Version) {
	for k, v := range known {
		if k == "AllAlpha" || k == "AllBeta" {
			continue
		}
		currentVersion := getCurrentVersion(v, cVer)
		if currentVersion.PreRelease == Alpha {
			if _, found := enabled[k]; !found {
				enabled[k] = val
			}
		}
	}
}

func setUnsetBetaGates(known map[Feature]VersionedSpecs, enabled map[Feature]bool, val bool, cVer *version.Version) {
	for k, v := range known {
		if k == "AllAlpha" || k == "AllBeta" {
			continue
		}
		currentVersion := getCurrentVersion(v, cVer)
		if currentVersion.PreRelease == Beta {
			if _, found := enabled[k]; !found {
				enabled[k] = val
			}
		}
	}
}

// Set, String, and Type implement pflag.Value
var _ pflag.Value = &featureGate{}

// internalPackages are packages that ignored when creating a name for featureGates. These packages are in the common
// call chains, so they'd be unhelpful as names.
var internalPackages = []string{"k8s.io/component-base/featuregate/feature_gate.go"}

// NewVersionedFeatureGate creates a feature gate with the emulation version set to the provided version.
// SetEmulationVersion can be called after to change emulation version to a desired value.
func NewVersionedFeatureGate(emulationVersion *version.Version) *featureGate {
	known := map[Feature]VersionedSpecs{}
	for k, v := range defaultFeatures {
		known[k] = v
	}

	f := &featureGate{
		featureGateName: naming.GetNameFromCallsite(internalPackages...),
		special:         specialFeatures,
	}
	f.known.Store(known)
	f.enabled.Store(map[Feature]bool{})
	f.enabledRaw.Store(map[string]bool{})
	f.emulationVersion.Store(emulationVersion)
	klog.V(1).Infof("new feature gate with emulationVersion=%s", f.emulationVersion.Load().String())
	return f
}

// NewFeatureGate creates a feature gate with the current binary version.
func NewFeatureGate() *featureGate {
	binaryVersison := version.MustParse(baseversion.Get().String())
	return NewVersionedFeatureGate(binaryVersison)
}

// Set parses a string of the form "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *featureGate) Set(value string) error {
	m := make(map[string]bool)
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		k := strings.TrimSpace(arr[0])
		if len(arr) != 2 {
			return fmt.Errorf("missing bool value for %s", k)
		}
		v := strings.TrimSpace(arr[1])
		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("invalid value of %s=%s, err: %v", k, v, err)
		}
		m[k] = boolValue
	}
	return f.SetFromMap(m)
}

// Validate checks if the flag gates are valid at the emulated version.
func (f *featureGate) Validate() []error {
	m, ok := f.enabledRaw.Load().(map[string]bool)
	if !ok {
		return []error{fmt.Errorf("cannot cast enabledRaw to map[string]bool")}
	}
	enabled := map[Feature]bool{}
	return f.unsafeSetFromMap(enabled, m)
}

// unsafeSetFromMap stores flag gates for known features from a map[string]bool into an enabled map.
func (f *featureGate) unsafeSetFromMap(enabled map[Feature]bool, m map[string]bool) []error {
	var errs []error
	// Copy existing state
	known := map[Feature]VersionedSpecs{}
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
		sort.Sort(v)
		known[k] = v
	}

	for k, v := range m {
		key := Feature(k)
		versionedSpecs, ok := known[key]
		if !ok {
			// early return if encounters an unknown feature.
			errs = append(errs, fmt.Errorf("unrecognized feature gate: %s", k))
			return errs
		}
		currentVersion := f.getCurrentVersion(versionedSpecs)
		if currentVersion.LockToDefault && currentVersion.Default != v {
			errs = append(errs, fmt.Errorf("cannot set feature gate %v to %v, feature is locked to %v", k, v, currentVersion.Default))
			continue
		}
		// Handle "special" features like "all alpha gates"
		if fn, found := f.special[key]; found {
			fn(known, enabled, v, f.emulationVersion.Load())
			enabled[key] = v
			continue
		}
		if currentVersion.PreRelease == PreAlpha {
			errs = append(errs, fmt.Errorf("cannot set feature gate %v to %v, feature is PreAlpha at emulated version %s", k, v, f.EmulationVersion().String()))
			continue
		}
		enabled[key] = v

		if currentVersion.PreRelease == Deprecated {
			klog.Warningf("Setting deprecated feature gate %s=%t. It will be removed in a future release.", k, v)
		} else if currentVersion.PreRelease == GA {
			klog.Warningf("Setting GA feature gate %s=%t. It will be removed in a future release.", k, v)
		}
	}
	return errs
}

// SetFromMap stores flag gates for known features from a map[string]bool or returns an error
func (f *featureGate) SetFromMap(m map[string]bool) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	// Copy existing state
	enabled := map[Feature]bool{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		enabled[k] = v
	}
	enabledRaw := map[string]bool{}
	for k, v := range f.enabledRaw.Load().(map[string]bool) {
		enabledRaw[k] = v
	}

	// Update enabledRaw first.
	// SetFromMap might be called when emulationVersion is not finalized yet, and we do not know the final state of enabled.
	// But the flags still need to be saved.
	for k, v := range m {
		enabledRaw[k] = v
	}
	f.enabledRaw.Store(enabledRaw)

	errs := f.unsafeSetFromMap(enabled, enabledRaw)
	if len(errs) == 0 {
		// Persist changes
		f.enabled.Store(enabled)
		klog.V(1).Infof("feature gates: %v", f.enabled)
	}
	return utilerrors.NewAggregate(errs)
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
	vs := map[Feature]VersionedSpecs{}
	for name, spec := range features {
		// if no version is provided for the FeatureSpec, it is defaulted to version 0.0 so that it can be enabled/disabled regardless of emulation version.
		spec.Version = version.MajorMinor(0, 0)
		vs[name] = VersionedSpecs{spec}
	}
	return f.AddVersioned(vs)
}

// AddVersioned adds versioned feature specs to the featureGate.
func (f *featureGate) AddVersioned(features map[Feature]VersionedSpecs) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		return fmt.Errorf("cannot add a feature gate after adding it to the flag set")
	}

	// Copy existing state
	known := map[Feature]VersionedSpecs{}
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
		known[k] = v
	}

	for name, specs := range features {
		sort.Sort(specs)
		if existingSpec, found := known[name]; found {
			sort.Sort(existingSpec)
			if reflect.DeepEqual(existingSpec, specs) {
				continue
			}
			return fmt.Errorf("feature gate %q with different spec already exists: %v", name, existingSpec)
		}
		known[name] = specs
	}

	// Persist updated state
	f.known.Store(known)

	return nil
}

func (f *featureGate) OverrideDefault(name Feature, override bool) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		return fmt.Errorf("cannot override default for feature %q: gates already added to a flag set", name)
	}

	known := map[Feature]VersionedSpecs{}
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
		sort.Sort(v)
		known[k] = v
	}

	specs, ok := known[name]
	if !ok {
		return fmt.Errorf("cannot override default: feature %q is not registered", name)
	}
	spec := f.getCurrentVersion(specs)
	switch {
	case spec.LockToDefault:
		return fmt.Errorf("cannot override default: feature %q default is locked to %t", name, spec.Default)
	case spec.PreRelease == PreAlpha:
		return fmt.Errorf("cannot override default: feature %q is not available before emulation version %s", name, f.EmulationVersion().String())
	case spec.PreRelease == Deprecated:
		klog.Warningf("Overriding default of deprecated feature gate %s=%t. It will be removed in a future release.", name, override)
	case spec.PreRelease == GA:
		klog.Warningf("Overriding default of GA feature gate %s=%t. It will be removed in a future release.", name, override)
	}

	spec.Default = override
	known[name] = specs
	f.known.Store(known)

	return nil
}

// GetAll returns a copy of the map of known feature names to feature specs for the current emulationVersion.
func (f *featureGate) GetAll() map[Feature]FeatureSpec {
	retval := map[Feature]FeatureSpec{}
	for k, v := range f.GetAllVersioned() {
		spec := f.getCurrentVersion(v)
		if spec.PreRelease == PreAlpha {
			// The feature is not available at the emulation version.
			continue
		}
		retval[k] = *f.getCurrentVersion(v)
	}
	return retval
}

// GetAllVersioned returns a copy of the map of known feature names to versioned feature specs.
func (f *featureGate) GetAllVersioned() map[Feature]VersionedSpecs {
	retval := map[Feature]VersionedSpecs{}
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
		retval[k] = v
	}
	return retval
}

func (f *featureGate) SetEmulationVersion(emulationVersion *version.Version) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	klog.V(1).Infof("set feature gate emulationVersion to %s", emulationVersion.String())
	f.emulationVersion.Store(emulationVersion)

	// Copy existing state
	enabledRaw := map[string]bool{}
	for k, v := range f.enabledRaw.Load().(map[string]bool) {
		enabledRaw[k] = v
	}
	// enabled map should be reset whenever emulationVersion is changed.
	enabled := map[Feature]bool{}
	errs := f.unsafeSetFromMap(enabled, enabledRaw)
	if len(errs) == 0 {
		// Persist changes
		f.enabled.Store(enabled)
	}
	return utilerrors.NewAggregate(errs)
}

func (f *featureGate) EmulationVersion() *version.Version {
	return f.emulationVersion.Load()
}

// FeatureSpec returns the FeatureSpec at the EmulationVersion if the key exists, an error otherwise.
func (f *featureGate) FeatureSpec(key Feature) (FeatureSpec, error) {
	if v, ok := f.known.Load().(map[Feature]VersionedSpecs)[key]; ok {
		currentVersion := f.getCurrentVersion(v)
		return *currentVersion, nil
	}
	return FeatureSpec{}, fmt.Errorf("feature %q is not registered in FeatureGate %q", key, f.featureGateName)
}

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *featureGate) Enabled(key Feature) bool {
	// fallback to default behavior, since we don't have emulation version set
	if v, ok := f.enabled.Load().(map[Feature]bool)[key]; ok {
		return v
	}
	if v, ok := f.known.Load().(map[Feature]VersionedSpecs)[key]; ok {
		return f.getCurrentVersion(v).Default
	}

	panic(fmt.Errorf("feature %q is not registered in FeatureGate %q", key, f.featureGateName))
}

func (f *featureGate) getCurrentVersion(v VersionedSpecs) *FeatureSpec {
	return getCurrentVersion(v, f.EmulationVersion())
}

func getCurrentVersion(v VersionedSpecs, emulationVersion *version.Version) *FeatureSpec {
	i := len(v) - 1
	for ; i >= 0; i-- {
		if v[i].Version.GreaterThan(emulationVersion) {
			continue
		}
		return &v[i]
	}
	return &FeatureSpec{
		Default:    false,
		PreRelease: PreAlpha,
		Version:    version.MajorMinor(0, 0),
	}
}

// Close sets closed to true, and prevents subsequent calls to Add
func (f *featureGate) Close() {
	f.lock.Lock()
	f.closed = true
	f.lock.Unlock()
}

// AddFlag adds a flag for setting global feature gates to the specified FlagSet.
func (f *featureGate) AddFlag(fs *pflag.FlagSet) {
	// TODO(mtaufen): Shouldn't we just close it on the first Set/SetFromMap instead?
	// Not all components expose a feature gates flag using this AddFlag method, and
	// in the future, all components will completely stop exposing a feature gates flag,
	// in favor of componentconfig.
	f.Close()

	known := f.KnownFeatures()
	fs.Var(f, flagName, ""+
		"A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(known, "\n"))
}

func (f *featureGate) AddMetrics() {
	for feature, featureSpec := range f.GetAll() {
		featuremetrics.RecordFeatureInfo(context.Background(), string(feature), string(featureSpec.PreRelease), f.Enabled(feature))
	}
}

// KnownFeatures returns a slice of strings describing the FeatureGate's known features.
// preAlpha, Deprecated and GA features are hidden from the list.
func (f *featureGate) KnownFeatures() []string {
	var known []string
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
		if k == "AllAlpha" || k == "AllBeta" {
			known = append(known, fmt.Sprintf("%s=true|false (%s - default=%t)", k, v[0].PreRelease, v[0].Default))
			continue
		}
		currentV := f.getCurrentVersion(v)
		if currentV.PreRelease == GA || currentV.PreRelease == Deprecated || currentV.PreRelease == PreAlpha {
			continue
		}
		known = append(known, fmt.Sprintf("%s=true|false (%s - default=%t)", k, currentV.PreRelease, currentV.Default))
	}
	sort.Strings(known)
	return known
}

// DeepCopy returns a deep copy of the FeatureGate object, such that gates can be
// set on the copy without mutating the original. This is useful for validating
// config against potential feature gate changes before committing those changes.
func (f *featureGate) DeepCopy() MutableVersionedFeatureGate {
	// Copy existing state.
	known := map[Feature]VersionedSpecs{}
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
		known[k] = v
	}
	enabled := map[Feature]bool{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		enabled[k] = v
	}
	enabledRaw := map[string]bool{}
	for k, v := range f.enabledRaw.Load().(map[string]bool) {
		enabledRaw[k] = v
	}

	// Construct a new featureGate around the copied state.
	// Note that specialFeatures is treated as immutable by convention,
	// and we maintain the value of f.closed across the copy.
	fg := &featureGate{
		special: specialFeatures,
		closed:  f.closed,
	}
	fg.emulationVersion.Store(f.EmulationVersion())
	fg.known.Store(known)
	fg.enabled.Store(enabled)
	fg.enabledRaw.Store(enabledRaw)
	return fg
}

// Reset sets the enabled and enabledRaw to the input map.
func (f *featureGate) Reset(m map[string]bool) {
	enabled := map[Feature]bool{}
	enabledRaw := map[string]bool{}
	f.enabled.Store(enabled)
	f.enabledRaw.Store(enabledRaw)
	_ = f.SetFromMap(m)
}

func (f *featureGate) EnabledRawMap() map[string]bool {
	return f.enabledRaw.Load().(map[string]bool)
}
