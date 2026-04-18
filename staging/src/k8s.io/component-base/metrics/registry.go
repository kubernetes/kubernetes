/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"slices"
	"strings"
	"sync"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/component-base/version"
)

var (
	registries     []*kubeRegistry // stores all registries created by NewKubeRegistry()
	registriesLock sync.RWMutex

	registeredMetricsTotal = NewCounterVec(
		&CounterOpts{
			Name:           "registered_metrics_total",
			Help:           "The count of registered metrics broken by stability level and deprecation version.",
			StabilityLevel: BETA,
		},
		[]string{"stability_level", "deprecated_version"},
	)
)

// shouldHide is used to check if a specific metric with deprecated version should be hidden
// according to metrics deprecation lifecycle.
func shouldHide(stabilityLevel StabilityLevel, currentVersion *semver.Version, deprecatedVersion *semver.Version) bool {
	hiddenMinor := deprecatedVersion.Minor + deprecationPeriodMinorVersions(stabilityLevel)

	switch {
	case deprecatedVersion.Major < currentVersion.Major:
		return true
	case deprecatedVersion.Major > currentVersion.Major:
		return false

	// deprecatedVersion.Major == currentVersion.Major
	case hiddenMinor < currentVersion.Minor:
		return true
	case hiddenMinor > currentVersion.Minor:
		return false

	// deprecatedVersion.Minor == currentVersion.Minor
	case strings.Contains(currentVersion.String(), "alpha.0"):
		// Wait until we're past the alpha.0 period of a minor development cycle to hide metrics whose deprecation period ends in that minor version.
		// See discussion in https://github.com/kubernetes/kubernetes/issues/133429#issuecomment-3165551443
		return false
	default:
		return true
	}
}

// getDeprecationReleaseWindow returns the number of minor releases a metric should be served
// after its deprecated version, based on its stability level.
func deprecationPeriodMinorVersions(stabilityLevel StabilityLevel) uint64 {
	switch stabilityLevel {
	case STABLE:
		return 3
	case BETA:
		return 1
	default: // ALPHA, INTERNAL
		return 0
	}
}

// isDeprecated returns true if the current version, ignoring pre-release tags,
// is greater than or equal to the deprecated version.
func isDeprecated(currentVersion, deprecatedVersion semver.Version) bool {
	switch {
	case currentVersion.Major < deprecatedVersion.Major:
		return false
	case currentVersion.Major > deprecatedVersion.Major:
		return true
	}

	// currentVersion.Major == deprecatedVersion.Major
	return currentVersion.Minor >= deprecatedVersion.Minor
}

// Registerable is an interface for a collector metric which we
// will register with KubeRegistry.
type Registerable interface {
	prometheus.Collector

	// Create will mark deprecated state for the collector
	Create(version *semver.Version) bool

	// ClearState will clear all the states marked by Create.
	ClearState()

	// FQName returns the fully-qualified metric name of the collector.
	FQName() string
}

type resettable interface {
	Reset()
}

// KubeRegistry is an interface which implements a subset of prometheus.Registerer and
// prometheus.Gatherer interfaces
type KubeRegistry interface {
	// Deprecated
	RawMustRegister(...prometheus.Collector)
	// CustomRegister is our internal variant of Prometheus registry.Register
	CustomRegister(c StableCollector) error
	// CustomMustRegister is our internal variant of Prometheus registry.MustRegister
	CustomMustRegister(cs ...StableCollector)
	// Register conforms to Prometheus registry.Register
	Register(Registerable) error
	// MustRegister conforms to Prometheus registry.MustRegister
	MustRegister(...Registerable)
	// Unregister conforms to Prometheus registry.Unregister
	Unregister(collector Collector) bool
	// Gather conforms to Prometheus gatherer.Gather
	Gather() ([]*dto.MetricFamily, error)
	// Reset invokes the Reset() function on all items in the registry
	// which are added as resettables.
	Reset()
	// RegisterMetaMetrics registers metrics about the number of registered metrics.
	RegisterMetaMetrics()
	// Registerer exposes the underlying prometheus registerer
	Registerer() prometheus.Registerer
	// Gatherer exposes the underlying prometheus gatherer
	Gatherer() prometheus.Gatherer
}

// KubeRegistryWithDeferred extends KubeRegistry to support deferred metric creation.
type KubeRegistryWithDeferred interface {
	KubeRegistry
	// EnableDeferredCreation puts the registry into deferred mode.
	// While deferred, MustRegister and Register store metrics without
	// creating the underlying prometheus objects. This allows
	// feature-gate-dependent options (e.g. native histograms) to be
	// configured before any metric is materialized.
	EnableDeferredCreation()
	// FinalizeDeferredMetrics creates all metrics that were deferred
	// during registration and registers them with the underlying
	// prometheus registry. After this call, subsequent MustRegister and
	// Register calls create metrics immediately. It is safe to call
	// multiple times; calls after the first are no-ops. Gather also
	// calls this automatically as a safety net.
	FinalizeDeferredMetrics()
}

// kubeRegistry is a wrapper around a prometheus registry-type object. Upon initialization
// the kubernetes binary version information is loaded into the registry object, so that
// automatic behavior can be configured for metric versioning.
type kubeRegistry struct {
	PromRegistry
	version              semver.Version
	hiddenCollectors     map[string]Registerable // stores all collectors that has been hidden
	stableCollectors     []StableCollector       // stores all stable collector
	hiddenCollectorsLock sync.RWMutex
	stableCollectorsLock sync.RWMutex
	resetLock            sync.RWMutex
	resettables          []resettable

	// deferLock guards deferredMode and the pending slices.
	deferLock         sync.Mutex
	deferredMode      bool
	pendingCollectors []Registerable    // metrics deferred from MustRegister/Register
	pendingCustom     []StableCollector // metrics deferred from CustomMustRegister/CustomRegister
}

// Register registers a new Collector to be included in metrics
// collection. It returns an error if the descriptors provided by the
// Collector are invalid or if they — in combination with descriptors of
// already registered Collectors — do not fulfill the consistency and
// uniqueness criteria described in the documentation of metric.Desc.
func (kr *kubeRegistry) Register(c Registerable) error {
	kr.deferLock.Lock()
	if kr.deferredMode {
		defer kr.deferLock.Unlock()
		if slices.Contains(kr.pendingCollectors, c) {
			return prometheus.AlreadyRegisteredError{ExistingCollector: c, NewCollector: c}
		}
		kr.pendingCollectors = append(kr.pendingCollectors, c)
		return nil
	}
	kr.deferLock.Unlock()

	if c.Create(&kr.version) {
		defer kr.addResettable(c)
		return kr.PromRegistry.Register(c)
	}

	kr.trackHiddenCollector(c)
	return nil
}

// Registerer exposes the underlying prometheus.Registerer
func (kr *kubeRegistry) Registerer() prometheus.Registerer {
	return kr.PromRegistry
}

// Gatherer exposes the underlying prometheus.Gatherer
func (kr *kubeRegistry) Gatherer() prometheus.Gatherer {
	return kr.PromRegistry
}

// MustRegister works like Register but registers any number of
// Collectors and panics upon the first registration that causes an
// error.
func (kr *kubeRegistry) MustRegister(cs ...Registerable) {
	kr.deferLock.Lock()
	if kr.deferredMode {
		defer kr.deferLock.Unlock()
		for _, c := range cs {
			if slices.Contains(kr.pendingCollectors, c) {
				panic(prometheus.AlreadyRegisteredError{ExistingCollector: c, NewCollector: c})
			}
			kr.pendingCollectors = append(kr.pendingCollectors, c)
		}
		return
	}
	kr.deferLock.Unlock()

	kr.mustRegisterImmediate(cs)
}

// mustRegisterImmediate performs the actual Create + PromRegistry.MustRegister
// work for a batch of collectors. It does NOT consult deferredMode; callers
// must have already handled the deferred-mode check (either by checking and
// returning, or by holding deferLock and ensuring deferredMode is false).
//
// The deferLock is intentionally not taken here so that this helper can be
// called from inside FinalizeDeferredMetrics, which holds deferLock across
// the work to prevent concurrent FinalizeDeferredRegistries hook callers
// from observing deferredMode=false while a metric's Create() is still in
// flight.
func (kr *kubeRegistry) mustRegisterImmediate(cs []Registerable) {
	toRegister := make([]prometheus.Collector, 0, len(cs))
	for _, c := range cs {
		if c.Create(&kr.version) {
			toRegister = append(toRegister, c)
			kr.addResettable(c)
		} else {
			kr.trackHiddenCollector(c)
		}
	}
	kr.PromRegistry.MustRegister(toRegister...)
}

// CustomRegister registers a new custom collector.
func (kr *kubeRegistry) CustomRegister(c StableCollector) error {
	kr.deferLock.Lock()
	if kr.deferredMode {
		defer kr.deferLock.Unlock()
		if slices.Contains(kr.pendingCustom, c) {
			return prometheus.AlreadyRegisteredError{ExistingCollector: c, NewCollector: c}
		}
		kr.pendingCustom = append(kr.pendingCustom, c)
		return nil
	}
	kr.deferLock.Unlock()

	kr.trackStableCollectors(c)
	defer kr.addResettable(c)
	if c.Create(&kr.version, c) {
		return kr.PromRegistry.Register(c)
	}
	return nil
}

// CustomMustRegister works like CustomRegister but registers any number of
// StableCollectors and panics upon the first registration that causes an
// error.
func (kr *kubeRegistry) CustomMustRegister(cs ...StableCollector) {
	kr.deferLock.Lock()
	if kr.deferredMode {
		defer kr.deferLock.Unlock()
		for _, c := range cs {
			if slices.Contains(kr.pendingCustom, c) {
				panic(prometheus.AlreadyRegisteredError{ExistingCollector: c, NewCollector: c})
			}
			kr.pendingCustom = append(kr.pendingCustom, c)
		}
		return
	}
	kr.deferLock.Unlock()

	kr.customMustRegisterImmediate(cs)
}

// customMustRegisterImmediate is the StableCollector counterpart of
// mustRegisterImmediate. Same locking contract: callers must have handled
// the deferred-mode check.
func (kr *kubeRegistry) customMustRegisterImmediate(cs []StableCollector) {
	kr.trackStableCollectors(cs...)
	toRegister := make([]prometheus.Collector, 0, len(cs))
	for _, c := range cs {
		if c.Create(&kr.version, c) {
			kr.addResettable(c)
			toRegister = append(toRegister, c)
		}
	}
	kr.PromRegistry.MustRegister(toRegister...)
}

// RawMustRegister takes a native prometheus.Collector and registers the collector
// to the registry. This bypasses metrics safety checks, so should only be used
// to register custom prometheus collectors.
//
// Deprecated
func (kr *kubeRegistry) RawMustRegister(cs ...prometheus.Collector) {
	kr.PromRegistry.MustRegister(cs...)
	for _, c := range cs {
		kr.addResettable(c)
	}
}

// addResettable will automatically add our metric to our reset
// list if it satisfies the interface
func (kr *kubeRegistry) addResettable(i interface{}) {
	kr.resetLock.Lock()
	defer kr.resetLock.Unlock()
	if resettable, ok := i.(resettable); ok {
		kr.resettables = append(kr.resettables, resettable)
	}
}

// Unregister unregisters the Collector that equals the Collector passed
// in as an argument.  (Two Collectors are considered equal if their
// Describe method yields the same set of descriptors.) The function
// returns whether a Collector was unregistered. Note that an unchecked
// Collector cannot be unregistered (as its Describe method does not
// yield any descriptor).
func (kr *kubeRegistry) Unregister(collector Collector) bool {
	return kr.PromRegistry.Unregister(collector)
}

// Gather calls the Collect method of the registered Collectors and then
// gathers the collected metrics into a lexicographically sorted slice
// of uniquely named MetricFamily protobufs. Gather ensures that the
// returned slice is valid and self-consistent so that it can be used
// for valid exposition. As an exception to the strict consistency
// requirements described for metric.Desc, Gather will tolerate
// different sets of label names for metrics of the same metric family.
//
// If the registry is still in deferred mode, Gather finalizes all
// pending metrics first so that they are included in the result.
func (kr *kubeRegistry) Gather() ([]*dto.MetricFamily, error) {
	kr.FinalizeDeferredMetrics()
	return kr.PromRegistry.Gather()
}

// EnableDeferredCreation switches the registry into deferred mode.
func (kr *kubeRegistry) EnableDeferredCreation() {
	kr.deferLock.Lock()
	defer kr.deferLock.Unlock()
	kr.deferredMode = true
}

// FinalizeDeferredMetrics processes all deferred metric registrations.
// It creates the underlying prometheus objects for every metric that was
// stored while the registry was in deferred mode, registers them, and
// disables deferred mode so future registrations take effect immediately.
// It is safe to call multiple times; calls after the first are no-ops.
//
// deferLock is held across the actual Create/Register work (not just the
// flag flip). This ensures that concurrent callers — in particular the
// FinalizeDeferredRegistries hook firing from multiple goroutines — block
// until finalization is complete, instead of observing deferredMode=false
// while a metric's Create() is still in flight (which would cause the
// caller to fall through to a noop and drop the write).
func (kr *kubeRegistry) FinalizeDeferredMetrics() {
	kr.deferLock.Lock()
	defer kr.deferLock.Unlock()
	if !kr.deferredMode {
		return
	}
	pending := kr.pendingCollectors
	pendingCustom := kr.pendingCustom
	kr.pendingCollectors = nil
	kr.pendingCustom = nil
	kr.deferredMode = false

	// Use the *Immediate helpers so we don't recurse back through
	// MustRegister/CustomMustRegister, which would deadlock on deferLock.
	// The lock stays held across the work, ensuring concurrent finalize
	// callers (e.g. parallel FinalizeDeferredRegistries hook firings)
	// block until everything is created instead of observing
	// deferredMode=false while Create() is still in flight.
	if len(pending) > 0 {
		kr.mustRegisterImmediate(pending)
	}
	if len(pendingCustom) > 0 {
		kr.customMustRegisterImmediate(pendingCustom)
	}
}

// trackHiddenCollector stores all hidden collectors.
func (kr *kubeRegistry) trackHiddenCollector(c Registerable) {
	kr.hiddenCollectorsLock.Lock()
	defer kr.hiddenCollectorsLock.Unlock()

	kr.hiddenCollectors[c.FQName()] = c
	hiddenMetricsTotal.Inc()
}

// trackStableCollectors stores all custom collectors.
func (kr *kubeRegistry) trackStableCollectors(cs ...StableCollector) {
	kr.stableCollectorsLock.Lock()
	defer kr.stableCollectorsLock.Unlock()

	kr.stableCollectors = append(kr.stableCollectors, cs...)
}

// enableHiddenCollectors will re-register all the hidden collectors.
func (kr *kubeRegistry) enableHiddenCollectors() {
	if len(kr.hiddenCollectors) == 0 {
		return
	}

	kr.hiddenCollectorsLock.Lock()
	cs := make([]Registerable, 0, len(kr.hiddenCollectors))

	for _, c := range kr.hiddenCollectors {
		c.ClearState()
		cs = append(cs, c)
	}

	kr.hiddenCollectors = make(map[string]Registerable)
	kr.hiddenCollectorsLock.Unlock()
	kr.MustRegister(cs...)
}

// enableHiddenStableCollectors will re-register the stable collectors if there is one or more hidden metrics in it.
// Since we can not register a metrics twice, so we have to unregister first then register again.
func (kr *kubeRegistry) enableHiddenStableCollectors() {
	if len(kr.stableCollectors) == 0 {
		return
	}

	kr.stableCollectorsLock.Lock()

	cs := make([]StableCollector, 0, len(kr.stableCollectors))
	for _, c := range kr.stableCollectors {
		if len(c.HiddenMetrics()) > 0 {
			kr.Unregister(c) // unregister must happens before clear state, otherwise no metrics would be unregister
			c.ClearState()
			cs = append(cs, c)
		}
	}

	kr.stableCollectors = nil
	kr.stableCollectorsLock.Unlock()
	kr.CustomMustRegister(cs...)
}

// Reset invokes Reset on all metrics that are resettable.
func (kr *kubeRegistry) Reset() {
	kr.resetLock.RLock()
	defer kr.resetLock.RUnlock()
	for _, r := range kr.resettables {
		r.Reset()
	}
}

// BuildVersion is a helper function that can be easily mocked.
var BuildVersion = version.Get

func newKubeRegistry(v apimachineryversion.Info) *kubeRegistry {
	r := &kubeRegistry{
		PromRegistry:     prometheus.NewRegistry(),
		version:          parseVersion(v),
		hiddenCollectors: make(map[string]Registerable),
		resettables:      make([]resettable, 0),
	}

	registriesLock.Lock()
	defer registriesLock.Unlock()
	registries = append(registries, r)

	return r
}

// NewKubeRegistry creates a new vanilla Registry
func NewKubeRegistry() KubeRegistry {
	r := newKubeRegistry(BuildVersion())
	return r
}

// NewKubeRegistryWithDeferred creates a new Registry with deferred metric creation support.
func NewKubeRegistryWithDeferred() KubeRegistryWithDeferred {
	r := newKubeRegistry(BuildVersion())
	return r
}

func (r *kubeRegistry) RegisterMetaMetrics() {
	r.MustRegister(registeredMetricsTotal)
	r.MustRegister(disabledMetricsTotal)
	r.MustRegister(hiddenMetricsTotal)
	r.MustRegister(cardinalityEnforcementUnexpectedCategorizationsTotal)
}
