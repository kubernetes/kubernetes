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

	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/component-base/version"
)

var (
	showHiddenOnce sync.Once
	showHidden     atomic.Value
	registries     []*kubeRegistry // stores all registries created by NewKubeRegistry()
	registriesLock sync.RWMutex
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

func validateShowHiddenMetricsVersion(currentVersion semver.Version, targetVersionStr string) error {
	if targetVersionStr == "" {
		return nil
	}

	validVersionStr := fmt.Sprintf("%d.%d", currentVersion.Major, currentVersion.Minor-1)
	if targetVersionStr != validVersionStr {
		return fmt.Errorf("--show-hidden-metrics-for-version must be omitted or have the value '%v'. Only the previous minor version is allowed", validVersionStr)
	}

	return nil
}

// ValidateShowHiddenMetricsVersion checks invalid version for which show hidden metrics.
func ValidateShowHiddenMetricsVersion(v string) []error {
	err := validateShowHiddenMetricsVersion(parseVersion(version.Get()), v)
	if err != nil {
		return []error{err}
	}

	return nil
}

// SetShowHidden will enable showing hidden metrics. This will no-opt
// after the initial call
func SetShowHidden() {
	showHiddenOnce.Do(func() {
		showHidden.Store(true)

		// re-register collectors that has been hidden in phase of last registry.
		for _, r := range registries {
			r.enableHiddenCollectors()
		}
	})
}

// ShouldShowHidden returns whether showing hidden deprecated metrics
// is enabled. While the primary usecase for this is internal (to determine
// registration behavior) this can also be used to introspect
func ShouldShowHidden() bool {
	return showHidden.Load() != nil && showHidden.Load().(bool)
}

// Registerable is an interface for a collector metric which we
// will register with KubeRegistry.
type Registerable interface {
	prometheus.Collector

	// Create will mark deprecated state for the collector
	Create(version *semver.Version) bool

	// ClearState will clear all the states marked by Create.
	ClearState()
}

// KubeRegistry is an interface which implements a subset of prometheus.Registerer and
// prometheus.Gatherer interfaces
type KubeRegistry interface {
	// Deprecated
	RawRegister(prometheus.Collector) error
	// Deprecated
	RawMustRegister(...prometheus.Collector)
	CustomRegister(c StableCollector) error
	CustomMustRegister(cs ...StableCollector)
	Register(Registerable) error
	MustRegister(...Registerable)
	Unregister(Registerable) bool
	Gather() ([]*dto.MetricFamily, error)
}

// kubeRegistry is a wrapper around a prometheus registry-type object. Upon initialization
// the kubernetes binary version information is loaded into the registry object, so that
// automatic behavior can be configured for metric versioning.
type kubeRegistry struct {
	PromRegistry
	version              semver.Version
	hiddenCollectors     []Registerable // stores all collectors that has been hidden
	hiddenCollectorsLock sync.RWMutex
}

// Register registers a new Collector to be included in metrics
// collection. It returns an error if the descriptors provided by the
// Collector are invalid or if they — in combination with descriptors of
// already registered Collectors — do not fulfill the consistency and
// uniqueness criteria described in the documentation of metric.Desc.
func (kr *kubeRegistry) Register(c Registerable) error {
	if c.Create(&kr.version) {
		return kr.PromRegistry.Register(c)
	}

	kr.trackHiddenCollector(c)

	return nil
}

// MustRegister works like Register but registers any number of
// Collectors and panics upon the first registration that causes an
// error.
func (kr *kubeRegistry) MustRegister(cs ...Registerable) {
	metrics := make([]prometheus.Collector, 0, len(cs))
	for _, c := range cs {
		if c.Create(&kr.version) {
			metrics = append(metrics, c)
		} else {
			kr.trackHiddenCollector(c)
		}
	}
	kr.PromRegistry.MustRegister(metrics...)
}

// CustomRegister registers a new custom collector.
func (kr *kubeRegistry) CustomRegister(c StableCollector) error {
	if c.Create(&kr.version, c) {
		return kr.PromRegistry.Register(c)
	}

	return nil
}

// CustomMustRegister works like CustomRegister but registers any number of
// StableCollectors and panics upon the first registration that causes an
// error.
func (kr *kubeRegistry) CustomMustRegister(cs ...StableCollector) {
	collectors := make([]prometheus.Collector, 0, len(cs))
	for _, c := range cs {
		if c.Create(&kr.version, c) {
			collectors = append(collectors, c)
		}
	}

	kr.PromRegistry.MustRegister(collectors...)
}

// RawRegister takes a native prometheus.Collector and registers the collector
// to the registry. This bypasses metrics safety checks, so should only be used
// to register custom prometheus collectors.
//
// Deprecated
func (kr *kubeRegistry) RawRegister(c prometheus.Collector) error {
	return kr.PromRegistry.Register(c)
}

// RawMustRegister takes a native prometheus.Collector and registers the collector
// to the registry. This bypasses metrics safety checks, so should only be used
// to register custom prometheus collectors.
//
// Deprecated
func (kr *kubeRegistry) RawMustRegister(cs ...prometheus.Collector) {
	kr.PromRegistry.MustRegister(cs...)
}

// Unregister unregisters the Collector that equals the Collector passed
// in as an argument.  (Two Collectors are considered equal if their
// Describe method yields the same set of descriptors.) The function
// returns whether a Collector was unregistered. Note that an unchecked
// Collector cannot be unregistered (as its Describe method does not
// yield any descriptor).
func (kr *kubeRegistry) Unregister(collector Registerable) bool {
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

	kr.hiddenCollectors = append(kr.hiddenCollectors, c)
}

// enableHiddenCollectors will re-register all of the hidden collectors.
func (kr *kubeRegistry) enableHiddenCollectors() {
	kr.hiddenCollectorsLock.Lock()
	defer kr.hiddenCollectorsLock.Unlock()

	for _, c := range kr.hiddenCollectors {
		c.ClearState()
		kr.MustRegister(c)
	}
	kr.hiddenCollectors = nil
}

func newKubeRegistry(v apimachineryversion.Info) *kubeRegistry {
	r := &kubeRegistry{
		PromRegistry: prometheus.NewRegistry(),
		version:      parseVersion(v),
	}

	registriesLock.Lock()
	defer registriesLock.Unlock()
	registries = append(registries, r)

	return r
}

// NewKubeRegistry creates a new vanilla Registry without any Collectors
// pre-registered.
func NewKubeRegistry() KubeRegistry {
	r := newKubeRegistry(version.Get())

	return r
}
