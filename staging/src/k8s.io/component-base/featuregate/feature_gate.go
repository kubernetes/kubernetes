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
	"maps"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/spf13/pflag"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/naming"
	"k8s.io/apimachinery/pkg/util/sets"
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
	specialFeatures = map[Feature]func(known map[Feature]VersionedSpecs, enabled map[Feature]bool, val bool, emuVer, minCompatVer *version.Version){
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
	// MinCompatibilityVersion indicates the lowest version that this feature spec is compatible with.
	// The component's minimum compatibility version must be greater than or equal to this version for this spec to be used.
	// This allows features with skew compatibility implications to be introduced as Beta,
	// but only on by default once the minimum compatibility version is high enough.
	// If unspecified, it is inherited from the previous feature stage.
	// If specified, it must be preceded by another FeatureSpec with the same version and different default but without the MinCompatibilityVersion.
	// i.e. MinCompatibilityVersion should only be used to specify a different default value at the same version.
	//
	// Version vs. MinCompatibilityVersion:
	//  - Version determines the stability of the feature based on the server's emulation version.
	//  - MinCompatibilityVersion adds a further check based on the version compatibility
	// with all servers in the control plane this server expects to communicate with.
	//
	// Example:
	// 	FeatureA: []FeatureSpec{
	// 		{Version: version.MustParse("1.35"), Default: false, PreRelease: Beta},
	// 		{Version: version.MustParse("1.35"), Default: true, PreRelease: Beta, MinCompatibilityVersion: version.MustParse("1.35")},
	// 	}
	// In a server running version 1.35:
	//  - With --min-compatibility-version=1.34 (default), FeatureA is Beta and disabled by default.
	//  - With --min-compatibility-version=1.35, FeatureA becomes enabled by default.
	MinCompatibilityVersion *version.Version
}

func (spec *FeatureSpec) HelpString(featureName Feature, includeCompatVer bool) string {
	if spec.PreRelease == GA || spec.PreRelease == Deprecated || spec.PreRelease == PreAlpha {
		return ""
	}
	s := fmt.Sprintf("%s=true|false (%s - default=%t)", featureName, spec.PreRelease, spec.Default)
	if includeCompatVer && spec.MinCompatibilityVersion != nil {
		return s + fmt.Sprintf(" if --min-compatibility-version>=%s", spec.MinCompatibilityVersion)
	}
	return s
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
	// Dependencies returns a copy of the known feature dependencies.
	Dependencies() map[Feature][]Feature
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
	// MinCompatibilityVersion returns the minimum version of all control plane components
	// that the current server must maintain compatibility with.
	// This is used to gate features that have cross-component compatibility concerns, for example, during cluster upgrades/rollbacks.
	// If not explicitly set, it defaults to one minor version prior to EmulationVersion().
	// A feature stage is disabled if its FeatureSpec.MinCompatibilityVersion is greater than this version.
	MinCompatibilityVersion() *version.Version
	// SetEmulationVersion overrides the emulationVersion of the feature gate, and
	// overrides the minCompatibilityVersion to 1 minor before emulationVersion.`
	// Otherwise, the emulationVersion will be the same as the binary version.
	// If set, the feature defaults and availability will be as if the binary is at the emulated version.
	SetEmulationVersion(emulationVersion *version.Version) error
	// SetEmulationVersion overrides the emulationVersion and minCompatibilityVersion of the feature gate.
	SetEmulationVersionAndMinCompatibilityVersion(emulationVersion *version.Version, minCompatibilityVersion *version.Version) error
	// GetAll returns a copy of the map of known feature names to versioned feature specs.
	GetAllVersioned() map[Feature]VersionedSpecs
	// AddVersioned adds versioned feature specs to the featureGate.
	AddVersioned(features map[Feature]VersionedSpecs) error
	// AddDependencies marks features that depend on other features. Must be called after all
	// referenced features have already been added (dependents & depnedencies). Cycles are forbidden.
	AddDependencies(features map[Feature][]Feature) error
	// OverrideDefaultAtVersion sets a local override for the registered default value of a named
	// feature for the prerelease lifecycle the given version is at.
	// If the feature has not been previously registered (e.g. by a call to Add),
	// has a locked default, or if the gate has already registered itself with a FlagSet, a non-nil
	// error is returned.
	//
	// When two or more components consume a common feature, one component can override its
	// default at runtime in order to adopt new defaults before or after the other
	// components. For example, a new feature can be evaluated with a limited blast radius by
	// overriding its default to true for a limited number of components without simultaneously
	// changing its default for all consuming components.
	OverrideDefaultAtVersion(name Feature, override bool, ver *version.Version) error
	// ExplicitlySet returns true if the feature value is explicitly set instead of
	// being derived from the default values or special features.
	ExplicitlySet(name Feature) bool
	// ResetFeatureValueToDefault resets the value of the feature back to the default value.
	ResetFeatureValueToDefault(name Feature) error
	// DeepCopyAndReset copies all the registered features of the FeatureGate object, with all the known features and overrides,
	// and resets all the enabled status of the new feature gate.
	// This is useful for creating a new instance of feature gate without inheriting all the enabled configurations of the base feature gate.
	DeepCopyAndReset() MutableVersionedFeatureGate
}

// featureGate implements FeatureGate as well as pflag.Value for flag parsing.
type featureGate struct {
	featureGateName string

	special map[Feature]func(map[Feature]VersionedSpecs, map[Feature]bool, bool, *version.Version, *version.Version)

	// lock guards writes to all below fields.
	lock sync.Mutex
	// known holds a map[Feature]FeatureSpec
	known atomic.Value
	// enabled holds a map[Feature]bool
	enabled      atomic.Value
	dependencies atomic.Pointer[map[Feature][]Feature]
	// enabledRaw holds a raw map[string]bool of the parsed flag.
	// It keeps the original values of "special" features like "all alpha gates",
	// while enabled keeps the values of all resolved features.
	enabledRaw atomic.Value
	// closed is set to true when AddFlag is called, and prevents subsequent calls to Add
	closed bool
	// queriedFeatures stores all the features that have been queried through the Enabled interface.
	// It is reset when SetEmulationVersion is called.
	queriedFeatures         atomic.Value
	emulationVersion        atomic.Pointer[version.Version]
	minCompatibilityVersion atomic.Pointer[version.Version]
}

func setUnsetAlphaGates(known map[Feature]VersionedSpecs, enabled map[Feature]bool, val bool, emuVer, minCompatVer *version.Version) {
	for k, v := range known {
		if k == "AllAlpha" || k == "AllBeta" {
			continue
		}
		featureSpec := featureSpecAtEmulationAndMinCompatVersion(v, emuVer, minCompatVer)
		if featureSpec.PreRelease == Alpha {
			if _, found := enabled[k]; !found {
				enabled[k] = val
			}
		}
	}
}

func setUnsetBetaGates(known map[Feature]VersionedSpecs, enabled map[Feature]bool, val bool, emuVer, minCompatVer *version.Version) {
	for k, v := range known {
		if k == "AllAlpha" || k == "AllBeta" {
			continue
		}
		featureSpec := featureSpecAtEmulationAndMinCompatVersion(v, emuVer, minCompatVer)
		if featureSpec.PreRelease == Beta {
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
// Equivalent to calling NewVersionedFeatureGateWithMinCompatibility with emulationVersion, emulationVersion-1.
// SetEmulationVersion can be called after to change emulation version to a desired value.
func NewVersionedFeatureGate(emulationVersion *version.Version) *featureGate {
	return NewVersionedFeatureGateWithMinCompatibility(emulationVersion, emulationVersion.SubtractMinor(1))
}

func NewVersionedFeatureGateWithMinCompatibility(emulationVersion, minCompatibilityVersion *version.Version) *featureGate {
	known := map[Feature]VersionedSpecs{}
	for k, v := range defaultFeatures {
		known[k] = v
	}

	f := &featureGate{
		featureGateName: naming.GetNameFromCallsite(internalPackages...),
		special:         specialFeatures,
	}
	f.known.Store(known)
	f.dependencies.Store(new(map[Feature][]Feature))
	f.enabled.Store(map[Feature]bool{})
	f.enabledRaw.Store(map[string]bool{})
	f.emulationVersion.Store(emulationVersion)
	f.minCompatibilityVersion.Store(minCompatibilityVersion)
	f.queriedFeatures.Store(sets.Set[Feature]{})
	return f
}

// NewFeatureGate creates a feature gate with the current binary version.
func NewFeatureGate() *featureGate {
	binaryVersison := version.MustParse(baseversion.DefaultKubeBinaryVersion)
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
	f.lock.Lock()
	defer f.lock.Unlock()
	m, ok := f.enabledRaw.Load().(map[string]bool)
	if !ok {
		return []error{fmt.Errorf("cannot cast enabledRaw to map[string]bool")}
	}
	enabled := map[Feature]bool{}
	return f.unsafeSetFromMap(enabled, m, f.EmulationVersion(), f.MinCompatibilityVersion())
}

// unsafeSetFromMap stores flag gates for known features from a map[string]bool into an enabled map.
func (f *featureGate) unsafeSetFromMap(enabled map[Feature]bool, m map[string]bool, emulationVersion, minCompatibilityVersion *version.Version) []error {
	var errs []error
	// Copy existing state
	known := map[Feature]VersionedSpecs{}
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
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
		featureSpec := featureSpecAtEmulationAndMinCompatVersion(versionedSpecs, emulationVersion, minCompatibilityVersion)
		if featureSpec.LockToDefault && featureSpec.Default != v {
			errs = append(errs, fmt.Errorf("cannot set feature gate %v to %v, feature is locked to %v", k, v, featureSpec.Default))
			continue
		}
		// Handle "special" features like "all alpha gates"
		if fn, found := f.special[key]; found {
			fn(known, enabled, v, emulationVersion, minCompatibilityVersion)
			enabled[key] = v
			continue
		}
		if featureSpec.PreRelease == PreAlpha {
			errs = append(errs, fmt.Errorf("cannot set feature gate %v to %v, feature is PreAlpha at emulated version %s, min compatibility version %s", k, v, emulationVersion, minCompatibilityVersion))
			continue
		}
		enabled[key] = v

		if featureSpec.PreRelease == Deprecated {
			klog.Warningf("Setting deprecated feature gate %s=%t. It will be removed in a future release.", k, v)
		} else if featureSpec.PreRelease == GA {
			klog.Warningf("Setting GA feature gate %s=%t. It will be removed in a future release.", k, v)
		}
	}

	if len(errs) > 0 {
		return errs
	}

	// If enabled features were set successfully, validate them against the dependencies.
	dependencies := *f.dependencies.Load()
	for feature, deps := range dependencies {
		if !featureEnabled(feature, enabled, known, emulationVersion, minCompatibilityVersion) {
			continue
		}

		var disabledDeps []Feature
		for _, dep := range deps {
			if !featureEnabled(dep, enabled, known, emulationVersion, minCompatibilityVersion) {
				disabledDeps = append(disabledDeps, dep)
			}
		}
		if len(disabledDeps) > 0 {
			errs = append(errs, fmt.Errorf("%s is enabled, but depends on features that are disabled: %v", feature, disabledDeps))
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

	errs := f.unsafeSetFromMap(enabled, enabledRaw, f.EmulationVersion(), f.MinCompatibilityVersion())
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
	known := f.GetAllVersioned()

	for name, specs := range features {
		var completedSpecs VersionedSpecs
		// Validate new specs are well-formed
		var lastSpec FeatureSpec
		var wasBeta, wasGA, wasDeprecated, wasLockedToDefault bool
		for i, spec := range specs {
			if spec.Version == nil {
				return fmt.Errorf("feature %q did not provide a version", name)
			}
			if len(spec.Version.Components()) != 2 {
				return fmt.Errorf("feature %q specified patch version: %s", name, spec.Version)
			}
			if spec.MinCompatibilityVersion != nil {
				if i == 0 || !spec.Version.EqualTo(lastSpec.Version) {
					return fmt.Errorf("feature %q specified MinCompatibilityVersion: %s without a preceding entry at the same Version without MinCompatibilityVersion set", name, spec.MinCompatibilityVersion)
				}
				if len(spec.MinCompatibilityVersion.Components()) != 2 {
					return fmt.Errorf("feature %q specified patch MinCompatibilityVersion: %s", name, spec.MinCompatibilityVersion)
				}
				if spec.MinCompatibilityVersion.GreaterThan(spec.Version) {
					return fmt.Errorf("feature %q MinCompatibilityVersion %s greater than Version %s", name, spec.MinCompatibilityVersion, spec.Version)
				}
			}
			// gates that begin as deprecated must indicate their prior state
			if i == 0 && spec.PreRelease == Deprecated && spec.Version.Minor() != 0 {
				return fmt.Errorf("feature %q introduced as deprecated must provide a 1.0 entry indicating initial state", name)
			}
			if i > 0 {
				// either versions or minCompatibilityVersions have to increase
				if spec.Version.LessThan(lastSpec.Version) {
					return fmt.Errorf("feature %q lists version transitions in decreasing order (%s < %s)", name, spec.Version, lastSpec.Version)
				}
				if spec.MinCompatibilityVersion != nil && lastSpec.MinCompatibilityVersion != nil && spec.MinCompatibilityVersion.LessThan(lastSpec.MinCompatibilityVersion) {
					return fmt.Errorf("feature %q lists min compatibility version transitions in decreasing order (%s < %s)", name, spec.MinCompatibilityVersion, lastSpec.MinCompatibilityVersion)
				}
				if spec.Version.EqualTo(lastSpec.Version) {
					if spec.MinCompatibilityVersion.EqualTo(lastSpec.MinCompatibilityVersion) {
						return fmt.Errorf("feature %q lists duplicate entries with the same versions (%s = %s)", name, spec.Version, lastSpec.Version)
					}
					if spec.PreRelease != lastSpec.PreRelease {
						return fmt.Errorf("feature %q lists stability changes with the same version (%s -> %s)", name, lastSpec.PreRelease, spec.PreRelease)
					}
					if spec.LockToDefault != lastSpec.LockToDefault {
						return fmt.Errorf("feature %q lists locked to default changes with the same version (%v -> %v)", name, lastSpec.LockToDefault, spec.LockToDefault)
					}
				}
				if spec.PreRelease == lastSpec.PreRelease && spec.Default == lastSpec.Default && spec.LockToDefault == lastSpec.LockToDefault {
					return fmt.Errorf("feature %q lists transition without stability change (%s -> %s)", name, lastSpec.Version, spec.Version)
				}
				if wasLockedToDefault && !spec.LockToDefault {
					return fmt.Errorf("feature %q must not unlock after locking to default", name)
				}
				// stability must not regress from ga --> {beta,alpha} or beta --> alpha, and
				// Deprecated state must be the terminal state
				switch {
				case spec.PreRelease != Deprecated && wasDeprecated:
					return fmt.Errorf("deprecated feature %q must not resurrect from its terminal state", name)
				case spec.PreRelease == Alpha && (wasBeta || wasGA):
					return fmt.Errorf("feature %q regresses stability from more stable level to %s in %s", name, spec.PreRelease, spec.Version)
				case spec.PreRelease == Beta && wasGA:
					return fmt.Errorf("feature %q regresses stability from more stable level to %s in %s", name, spec.PreRelease, spec.Version)
				}
			}
			completedSpec := spec
			// set MinCompatibilityVersion to the last MinCompatibilityVersion if unspecified.
			// this makes sure MinCompatibilityVersion does not decrease.
			// For example:
			// 	FeatureA: {
			// 		{Version: version.MustParse("1.35"), Default: false, PreRelease: Beta},
			// 		{Version: version.MustParse("1.35"), Default: true, PreRelease: Beta, MinCompatibilityVersion: 1.35},
			//      {Version: version.MustParse("1.37"), Default: true, PreRelease: GA},
			// 	},
			// FeatureA should have MinCompatibilityVersion at least 1.35 at GA.
			if completedSpec.MinCompatibilityVersion == nil && i > 0 {
				completedSpec.MinCompatibilityVersion = lastSpec.MinCompatibilityVersion
			}
			completedSpecs = append(completedSpecs, completedSpec)
			lastSpec = completedSpec
			wasBeta = wasBeta || spec.PreRelease == Beta
			wasGA = wasGA || spec.PreRelease == GA
			wasDeprecated = wasDeprecated || spec.PreRelease == Deprecated
			wasLockedToDefault = wasLockedToDefault || spec.LockToDefault
		}
		if existingSpec, found := known[name]; found {
			if reflect.DeepEqual(existingSpec, completedSpecs) {
				continue
			}
			return fmt.Errorf("feature gate %q with different spec already exists: %v", name, existingSpec)
		}
		known[name] = completedSpecs
	}
	// Persist updated state
	f.known.Store(known)

	return nil
}

// AddDependencies adds feature gate dependencies.
func (f *featureGate) AddDependencies(dependencies map[Feature][]Feature) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		return fmt.Errorf("cannot add a feature gate dependency after adding it to the flag set")
	}
	// Copy existing state
	known := f.GetAllVersioned()

	// Merge existing dependencies in.
	existing := *f.dependencies.Load()
	for k, v := range existing {
		dependencies[k] = append(dependencies[k], v...)
	}

	// Sort and compact all dependency lists.
	for k, v := range dependencies {
		slices.Sort(v)
		dependencies[k] = slices.Compact(v)
	}

	// Validate dependencies for each emulated version:
	// 1. Features & dependencies must be known
	// 2. Features cannot depend on features with a lower prerelease level
	// 3. Enabled features cannot depend on disabled features
	// 4. Locked-to-default featurs cannot depend on unlocked features
	for feature, deps := range dependencies {
		versionedFeature, ok := known[feature]
		if !ok {
			return fmt.Errorf("cannot add dependency for unknown feature %s", feature)
		}

		for _, dep := range deps {
			versionedDep, ok := known[dep]
			if !ok {
				return fmt.Errorf("cannot add dependency from %s to unknown feature %s", feature, dep)
			}

			// Check versions starting with the most recent for more intuitive error messages.
			for _, spec := range slices.Backward(versionedFeature) {
				// depSpec is the effective FeatureSpec for the dependency at the version declared by the dependent FeatureSpec.
				depSpec := featureSpecAtEmulationAndMinCompatVersion(versionedDep, spec.Version, spec.MinCompatibilityVersion)

				if stabilityOrder(spec.PreRelease) > stabilityOrder(depSpec.PreRelease) {
					return fmt.Errorf("%s feature %s cannot depend on %s feature %s at version %s", spec.PreRelease, feature, depSpec.PreRelease, dep, spec.Version)
				}
				if spec.Default && !depSpec.Default {
					return fmt.Errorf("default-enabled feature %s cannot depend on default-disabled feature %s at version %s", feature, dep, spec.Version)
				}
				if spec.LockToDefault && !depSpec.LockToDefault {
					return fmt.Errorf("locked-to-default feature %s cannot depend on unlocked feature %s at version %s", feature, dep, spec.Version)
				}
			}
		}
	}

	// Check for cycles
	visited := map[Feature]bool{}
	finished := map[Feature]bool{}
	var detectCycles func(Feature) error
	detectCycles = func(feature Feature) error {
		if finished[feature] {
			return nil
		}
		if visited[feature] {
			return fmt.Errorf("cycle detected with feature %s", feature)
		}
		visited[feature] = true
		for _, dep := range dependencies[feature] {
			if err := detectCycles(dep); err != nil {
				return err
			}
		}
		finished[feature] = true
		return nil
	}
	for feature := range dependencies {
		if err := detectCycles(feature); err != nil {
			return err
		}
	}

	// Persist updated state
	f.dependencies.Store(&dependencies)

	return nil
}

// stabilityOrder converts prereleases to a numerical value for dependency comparison.
// Features cannot depend on features with a lower prerelease level.
func stabilityOrder(p prerelease) int {
	switch p {
	case Deprecated:
		return 0 // Non-deprecated features cannot depend on deprecated features.
	case PreAlpha, Alpha: // Alpha features are allowed to depend on pre-alpha features.
		return 1
	case Beta:
		return 2
	case GA:
		return 3
	default:
		return -1 // Unknown prerelease
	}
}

func (f *featureGate) Dependencies() map[Feature][]Feature {
	return maps.Clone(*f.dependencies.Load())
}

func (f *featureGate) OverrideDefault(name Feature, override bool) error {
	return f.overrideDefaultAtEmulationAndMinCompatVersion(name, override, f.EmulationVersion(), f.MinCompatibilityVersion())
}

func (f *featureGate) OverrideDefaultAtVersion(name Feature, override bool, ver *version.Version) error {
	return f.overrideDefaultAtEmulationAndMinCompatVersion(name, override, ver, nil)
}

func (f *featureGate) overrideDefaultAtEmulationAndMinCompatVersion(name Feature, override bool, emuVer, minCompatVer *version.Version) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		return fmt.Errorf("cannot override default for feature %q: gates already added to a flag set", name)
	}

	// Copy existing state
	known := f.GetAllVersioned()

	specs, ok := known[name]
	if !ok {
		return fmt.Errorf("cannot override default: feature %q is not registered", name)
	}
	spec := featureSpecAtEmulationAndMinCompatVersion(specs, emuVer, minCompatVer)
	switch {
	case spec.LockToDefault:
		return fmt.Errorf("cannot override default: feature %q default is locked to %t", name, spec.Default)
	case spec.PreRelease == PreAlpha:
		return fmt.Errorf("cannot override default: feature %q is not available before version %s", name, emuVer)
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
	f.lock.Lock()
	versionedSpecs := f.GetAllVersioned()
	emuVer := f.EmulationVersion()
	minCompatVer := f.MinCompatibilityVersion()
	f.lock.Unlock()
	for k, v := range versionedSpecs {
		spec := featureSpecAtEmulationAndMinCompatVersion(v, emuVer, minCompatVer)
		if spec.PreRelease == PreAlpha {
			// The feature is not available at the emulation version.
			continue
		}
		retval[k] = *spec
	}
	return retval
}

// GetAllVersioned returns a copy of the map of known feature names to versioned feature specs.
func (f *featureGate) GetAllVersioned() map[Feature]VersionedSpecs {
	retval := map[Feature]VersionedSpecs{}
	for k, v := range f.known.Load().(map[Feature]VersionedSpecs) {
		vCopy := make([]FeatureSpec, len(v))
		_ = copy(vCopy, v)
		retval[k] = vCopy
	}
	return retval
}

func (f *featureGate) SetEmulationVersion(emulationVersion *version.Version) error {
	return f.SetEmulationVersionAndMinCompatibilityVersion(emulationVersion, emulationVersion.SubtractMinor(1))
}

func (f *featureGate) SetEmulationVersionAndMinCompatibilityVersion(emulationVersion *version.Version, minCompatibilityVersion *version.Version) error {
	if emulationVersion.EqualTo(f.EmulationVersion()) && minCompatibilityVersion.EqualTo(f.MinCompatibilityVersion()) {
		return nil
	}
	f.lock.Lock()
	defer f.lock.Unlock()
	klog.V(1).Infof("set feature gate emulationVersion to %s, minCompatibilityVersion to %s", emulationVersion, minCompatibilityVersion)

	// Copy existing state
	enabledRaw := map[string]bool{}
	for k, v := range f.enabledRaw.Load().(map[string]bool) {
		enabledRaw[k] = v
	}
	// enabled map should be reset whenever emulationVersion is changed.
	enabled := map[Feature]bool{}
	errs := f.unsafeSetFromMap(enabled, enabledRaw, emulationVersion, minCompatibilityVersion)

	queriedFeatures := f.queriedFeatures.Load().(sets.Set[Feature])
	known := f.known.Load().(map[Feature]VersionedSpecs)
	for feature := range queriedFeatures {
		newVal := featureEnabled(feature, enabled, known, emulationVersion, minCompatibilityVersion)
		oldVal := featureEnabled(feature, f.enabled.Load().(map[Feature]bool), known, f.EmulationVersion(), f.MinCompatibilityVersion())
		if newVal != oldVal {
			klog.Warningf("SetEmulationVersionAndMinCompatibilityVersion will change already queried feature:%s from %v to %v", feature, oldVal, newVal)
		}
	}

	if len(errs) == 0 {
		// Persist changes
		f.enabled.Store(enabled)
		f.emulationVersion.Store(emulationVersion)
		f.minCompatibilityVersion.Store(minCompatibilityVersion)
		f.queriedFeatures.Store(sets.Set[Feature]{})
	}
	return utilerrors.NewAggregate(errs)
}

func (f *featureGate) EmulationVersion() *version.Version {
	return f.emulationVersion.Load()
}

func (f *featureGate) MinCompatibilityVersion() *version.Version {
	return f.minCompatibilityVersion.Load()
}

// featureSpec returns the featureSpec at the EmulationVersion if the key exists, an error otherwise.
// This is useful to keep multiple implementations of a feature based on the PreRelease or Version info.
func (f *featureGate) featureSpec(key Feature) (FeatureSpec, error) {
	if v, ok := f.known.Load().(map[Feature]VersionedSpecs)[key]; ok {
		featureSpec := f.featureSpecAtEmulationAndMinCompatVersion(v)
		return *featureSpec, nil
	}
	return FeatureSpec{}, fmt.Errorf("feature %q is not registered in FeatureGate %q", key, f.featureGateName)
}

func (f *featureGate) unsafeRecordQueried(key Feature) {
	queriedFeatures := f.queriedFeatures.Load().(sets.Set[Feature])
	if _, ok := queriedFeatures[key]; ok {
		return
	}
	// Clone items from queriedFeatures before mutating it
	newQueriedFeatures := queriedFeatures.Clone()
	newQueriedFeatures.Insert(key)
	f.queriedFeatures.Store(newQueriedFeatures)
}

func featureEnabled(key Feature, enabled map[Feature]bool, known map[Feature]VersionedSpecs, emulationVersion, minCompatibilityVersion *version.Version) bool {
	// check explicitly set enabled list
	if v, ok := enabled[key]; ok {
		return v
	}
	if v, ok := known[key]; ok {
		featureSpec := featureSpecAtEmulationAndMinCompatVersion(v, emulationVersion, minCompatibilityVersion)
		return featureSpec.Default
	}

	panic(fmt.Errorf("feature %q is not registered in FeatureGate", key))
}

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *featureGate) Enabled(key Feature) bool {
	// TODO: ideally we should lock the feature gate in this call to be safe, need to evaluate how much performance impact locking would have.
	v := featureEnabled(key, f.enabled.Load().(map[Feature]bool), f.known.Load().(map[Feature]VersionedSpecs), f.EmulationVersion(), f.MinCompatibilityVersion())
	f.unsafeRecordQueried(key)
	return v
}

func (f *featureGate) featureSpecAtEmulationAndMinCompatVersion(v VersionedSpecs) *FeatureSpec {
	return featureSpecAtEmulationAndMinCompatVersion(v, f.EmulationVersion(), f.MinCompatibilityVersion())
}

func featureSpecAtEmulationAndMinCompatVersion(v VersionedSpecs, emulationVersion, minCompatibilityVersion *version.Version) *FeatureSpec {
	i := len(v) - 1
	if minCompatibilityVersion == nil {
		minCompatibilityVersion = emulationVersion.SubtractMinor(1)
	}
	for ; i >= 0; i-- {
		if v[i].Version.GreaterThan(emulationVersion) {
			continue
		}
		specMinCompatibilityVersion := v[i].MinCompatibilityVersion
		if specMinCompatibilityVersion != nil && !minCompatibilityVersion.AtLeast(specMinCompatibilityVersion) {
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
		featureSpec := f.featureSpecAtEmulationAndMinCompatVersion(v)
		featureSpecStr := featureSpec.HelpString(k, false)
		featureSpecMinCompatMode := featureSpecAtEmulationAndMinCompatVersion(v, f.EmulationVersion(), f.EmulationVersion())
		featureSpecMinCompatModeStr := featureSpecMinCompatMode.HelpString(k, true)
		if reflect.DeepEqual(featureSpec, featureSpecMinCompatMode) {
			if featureSpecStr != "" {
				known = append(known, featureSpecStr)
			}
			continue
		}
		components := []string{}
		if featureSpecStr != "" {
			components = append(components, featureSpecStr)
		}
		if featureSpecMinCompatModeStr != "" {
			components = append(components, featureSpecMinCompatModeStr)
		}
		if len(components) > 0 {
			known = append(known, strings.Join(components, ", or "))
		}
	}
	sort.Strings(known)
	return known
}

// DeepCopyAndReset copies all the registered features of the FeatureGate object, with all the known features and overrides,
// and resets all the enabled status of the new feature gate.
// This is useful for creating a new instance of feature gate without inheriting all the enabled configurations of the base feature gate.
func (f *featureGate) DeepCopyAndReset() MutableVersionedFeatureGate {
	fg := NewVersionedFeatureGateWithMinCompatibility(f.EmulationVersion(), f.MinCompatibilityVersion())
	known := f.GetAllVersioned()
	fg.known.Store(known)
	return fg
}

// DeepCopy returns a deep copy of the FeatureGate object, such that gates can be
// set on the copy without mutating the original. This is useful for validating
// config against potential feature gate changes before committing those changes.
func (f *featureGate) DeepCopy() MutableVersionedFeatureGate {
	f.lock.Lock()
	defer f.lock.Unlock()
	// Copy existing state.
	known := f.GetAllVersioned()
	enabled := map[Feature]bool{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		enabled[k] = v
	}
	enabledRaw := map[string]bool{}
	for k, v := range f.enabledRaw.Load().(map[string]bool) {
		enabledRaw[k] = v
	}
	dependencies := map[Feature][]Feature{}
	for k, v := range *f.dependencies.Load() {
		dependencies[k] = append([]Feature{}, v...)
	}

	// Construct a new featureGate around the copied state.
	// Note that specialFeatures is treated as immutable by convention,
	// and we maintain the value of f.closed across the copy.
	fg := &featureGate{
		special: specialFeatures,
		closed:  f.closed,
	}
	fg.emulationVersion.Store(f.EmulationVersion())
	fg.minCompatibilityVersion.Store(f.MinCompatibilityVersion())
	fg.known.Store(known)
	fg.enabled.Store(enabled)
	fg.dependencies.Store(&dependencies)
	fg.enabledRaw.Store(enabledRaw)
	fg.queriedFeatures.Store(sets.Set[Feature]{})
	return fg
}

// ExplicitlySet returns true if the feature value is explicitly set instead of
// being derived from the default values or special features.
func (f *featureGate) ExplicitlySet(name Feature) bool {
	enabledRaw := f.enabledRaw.Load().(map[string]bool)
	_, ok := enabledRaw[string(name)]
	return ok
}

// ResetFeatureValueToDefault resets the value of the feature back to the default value.
func (f *featureGate) ResetFeatureValueToDefault(name Feature) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	enabled := map[Feature]bool{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		enabled[k] = v
	}
	enabledRaw := map[string]bool{}
	for k, v := range f.enabledRaw.Load().(map[string]bool) {
		enabledRaw[k] = v
	}
	_, inEnabled := enabled[name]
	if inEnabled {
		delete(enabled, name)
	}
	_, inEnabledRaw := enabledRaw[string(name)]
	if inEnabledRaw {
		delete(enabledRaw, string(name))
	}
	// some features could be in enabled map but not enabledRaw map,
	// for example some Alpha feature when AllAlpha is set.
	if inEnabledRaw && !inEnabled {
		return fmt.Errorf("feature:%s was explicitly set, but not in enabled map", name)
	}
	f.enabled.Store(enabled)
	f.enabledRaw.Store(enabledRaw)
	return nil
}
