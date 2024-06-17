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
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/component-base/version"
)

var (
	showHiddenOnce      sync.Once
	disabledMetricsLock sync.RWMutex
	showHidden          atomic.Bool
	registries          []*kubeRegistry // stores all registries created by NewKubeRegistry()
	registriesLock      sync.RWMutex
	disabledMetrics     = map[string]struct{}{}

	registeredMetricsTotal = NewCounterVec(
		&CounterOpts{
			Name:           "registered_metrics_total",
			Help:           "The count of registered metrics broken by stability level and deprecation version.",
			StabilityLevel: BETA,
		},
		[]string{"stability_level", "deprecated_version"},
	)

	disabledMetricsTotal = NewCounter(
		&CounterOpts{
			Name:           "disabled_metrics_total",
			Help:           "The count of disabled metrics.",
			StabilityLevel: BETA,
		},
	)

	hiddenMetricsTotal = NewCounter(
		&CounterOpts{
			Name:           "hidden_metrics_total",
			Help:           "The count of hidden metrics.",
			StabilityLevel: BETA,
		},
	)

	cardinalityEnforcementUnexpectedCategorizationsTotal = NewCounter(
		&CounterOpts{
			Name:           "cardinality_enforcement_unexpected_categorizations_total",
			Help:           "The count of unexpected categorizations during cardinality enforcement.",
			StabilityLevel: ALPHA,
		},
	)
)

// shouldHide be used to check if a specific metric with deprecated version should be hidden
// according to metrics deprecation lifecycle.
func shouldHide(currentVersion *semver.Version, deprecatedVersion *semver.Version) bool {
	guardVersion, err := semver.Make(fmt.Sprintf("%d.%d.0", currentVersion.Major, currentVersion.Minor))
	if err != nil {
		panic("failed to make version from current version")
	}

	if deprecatedVersion.LT(guardVersion) {
		return true
	}

	return false
}

// ValidateShowHiddenMetricsVersion checks invalid version for which show hidden metrics.
func ValidateShowHiddenMetricsVersion(v string) []error {
	err := validateShowHiddenMetricsVersion(parseVersion(version.Get()), v)
	if err != nil {
		return []error{err}
	}

	return nil
}

func SetDisabledMetric(name string) {
	disabledMetricsLock.Lock()
	defer disabledMetricsLock.Unlock()
	disabledMetrics[name] = struct{}{}
	disabledMetricsTotal.Inc()
}

// SetShowHidden will enable showing hidden metrics. This will no-opt
// after the initial call
func SetShowHidden() {
	showHiddenOnce.Do(func() {
		showHidden.Store(true)

		// re-register collectors that has been hidden in phase of last registry.
		for _, r := range registries {
			r.enableHiddenCollectors()
			r.enableHiddenStableCollectors()
		}
	})
}

// ShouldShowHidden returns whether showing hidden deprecated metrics
// is enabled. While the primary usecase for this is internal (to determine
// registration behavior) this can also be used to introspect
func ShouldShowHidden() bool {
	return showHidden.Load()
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
}

// Register registers a new Collector to be included in metrics
// collection. It returns an error if the descriptors provided by the
// Collector are invalid or if they — in combination with descriptors of
// already registered Collectors — do not fulfill the consistency and
// uniqueness criteria described in the documentation of metric.Desc.
func (kr *kubeRegistry) Register(c Registerable) error {
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
	metrics := make([]prometheus.Collector, 0, len(cs))
	for _, c := range cs {
		if c.Create(&kr.version) {
			metrics = append(metrics, c)
			kr.addResettable(c)
		} else {
			kr.trackHiddenCollector(c)
		}
	}
	kr.PromRegistry.MustRegister(metrics...)
}

// CustomRegister registers a new custom collector.
func (kr *kubeRegistry) CustomRegister(c StableCollector) error {
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
	kr.trackStableCollectors(cs...)
	collectors := make([]prometheus.Collector, 0, len(cs))
	for _, c := range cs {
		if c.Create(&kr.version, c) {
			kr.addResettable(c)
			collectors = append(collectors, c)
		}
	}
	kr.PromRegistry.MustRegister(collectors...)
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
func (kr *kubeRegistry) Gather() ([]*dto.MetricFamily, error) {
	return kr.PromRegistry.Gather()
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

// enableHiddenCollectors will re-register all of the hidden collectors.
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

func (r *kubeRegistry) RegisterMetaMetrics() {
	r.MustRegister(registeredMetricsTotal)
	r.MustRegister(disabledMetricsTotal)
	r.MustRegister(hiddenMetricsTotal)
	r.MustRegister(cardinalityEnforcementUnexpectedCategorizationsTotal)
}
