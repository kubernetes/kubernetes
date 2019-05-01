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
	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/version"
)

var DefaultGlobalRegistry = NewKubeRegistry()

type KubeRegistry struct {
	PromRegistry
	version semver.Version
}

// Register registers a collectable metric, but it uses a global registry.
func Register(c KubeCollector) error {
	return DefaultGlobalRegistry.Register(c)
}

// MustRegister works like Register but registers any number of
// Collectors and panics upon the first registration that causes an
// error.
func MustRegister(cs ...KubeCollector) {
	DefaultGlobalRegistry.MustRegister(cs...)
}

func (kr *KubeRegistry) Register(c KubeCollector) error {
	if c.Create(&kr.version) {
		return kr.PromRegistry.Register(c)
	}
	return nil
}

func (kr *KubeRegistry) MustRegister(cs ...KubeCollector) {
	metrics := make([]prometheus.Collector, 0, len(cs))
	for _, c := range cs {
		if c.Create(&kr.version) {
			metrics = append(metrics, c)
		}
	}
	kr.PromRegistry.MustRegister(metrics...)
}

func (kr *KubeRegistry) Unregister(collector KubeCollector) bool {
	return kr.PromRegistry.Unregister(collector)
}

func (kr *KubeRegistry) Gather() ([]*dto.MetricFamily, error) {
	return kr.PromRegistry.Gather()
}

func NewKubeRegistry() *KubeRegistry {
	v, err := parseVersion(version.Get())
	if err != nil {
		klog.Fatalf("Can't initialize a registry without a valid version %v", err)
	}
	if v == nil {
		klog.Fatalf("No valid version loaded for metrics registry")
	}
	return newKubeRegistry(semver.MustParse(*v))
}

// newKubeRegistry creates a new vanilla Registry without any Collectors
// pre-registered.
func newKubeRegistry(version semver.Version) *KubeRegistry {
	return &KubeRegistry{
		PromRegistry: prometheus.NewRegistry(),
		version:      version,
	}
}
