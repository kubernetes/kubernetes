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

package legacyregistry

import (
	"fmt"
	"net/http"
	"reflect"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	dto "github.com/prometheus/client_model/go"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/component-base/metrics"
)

var (
	globalRegistryFactory = metricsRegistryFactory{
		registerQueue:     make([]metrics.Registerable, 0),
		mustRegisterQueue: make([]metrics.Registerable, 0),
		globalRegistry:    noopRegistry{},
	}
)

type noopRegistry struct{}

func (noopRegistry) Register(metrics.Registerable) error     { return nil }
func (noopRegistry) MustRegister(...metrics.Registerable)    {}
func (noopRegistry) RawRegister(prometheus.Collector) error  { return nil }
func (noopRegistry) RawMustRegister(...prometheus.Collector) {}
func (noopRegistry) Unregister(metrics.Registerable) bool    { return true }
func (noopRegistry) Gather() ([]*dto.MetricFamily, error)    { return nil, nil }

type metricsRegistryFactory struct {
	globalRegistry    metrics.KubeRegistry
	kubeVersion       *apimachineryversion.Info
	registrationLock  sync.Mutex
	registerQueue     []metrics.Registerable
	mustRegisterQueue []metrics.Registerable
}

// HandlerForGlobalRegistry returns a http handler for the global registry. This
// allows us to return a handler for the global registry without having to expose
// the global registry itself directly.
func HandlerForGlobalRegistry(opts promhttp.HandlerOpts) http.Handler {
	return promhttp.HandlerFor(globalRegistryFactory.globalRegistry, opts)
}

// SetRegistryFactoryVersion sets the kubernetes version information for all
// subsequent metrics registry initializations. Only the first call has an effect.
// If a version is not set, then metrics registry creation will no-opt
func SetRegistryFactoryVersion(ver apimachineryversion.Info) []error {
	globalRegistryFactory.registrationLock.Lock()
	defer globalRegistryFactory.registrationLock.Unlock()
	if globalRegistryFactory.kubeVersion != nil {
		if globalRegistryFactory.kubeVersion.String() != ver.String() {
			panic(fmt.Sprintf("Cannot load a global registry more than once, had %s tried to load %s",
				globalRegistryFactory.kubeVersion.String(),
				ver.String()))
		}
		return nil
	}
	registrationErrs := make([]error, 0)
	preloadedMetrics := []prometheus.Collector{
		prometheus.NewGoCollector(),
		prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}),
	}
	globalRegistryFactory.globalRegistry = metrics.NewKubeRegistry(ver)
	globalRegistryFactory.globalRegistry.RawMustRegister(preloadedMetrics...)
	globalRegistryFactory.kubeVersion = &ver
	for _, c := range globalRegistryFactory.registerQueue {
		err := globalRegistryFactory.globalRegistry.Register(c)
		if err != nil {
			registrationErrs = append(registrationErrs, err)
		}
	}
	for _, c := range globalRegistryFactory.mustRegisterQueue {
		globalRegistryFactory.globalRegistry.MustRegister(c)
	}
	return registrationErrs
}

// Register registers a collectable metric, but it uses a global registry. Registration is deferred
// until the global registry has a version to use.
func Register(c metrics.Registerable) error {
	globalRegistryFactory.registrationLock.Lock()
	defer globalRegistryFactory.registrationLock.Unlock()

	if reflect.DeepEqual(globalRegistryFactory.globalRegistry, noopRegistry{}) {
		globalRegistryFactory.registerQueue = append(globalRegistryFactory.registerQueue, c)
		return nil
	}

	return globalRegistryFactory.globalRegistry.Register(c)
}

// MustRegister works like Register but registers any number of
// Collectors and panics upon the first registration that causes an
// error. Registration is deferred  until the global registry has a version to use.
func MustRegister(cs ...metrics.Registerable) {
	globalRegistryFactory.registrationLock.Lock()
	defer globalRegistryFactory.registrationLock.Unlock()

	if reflect.DeepEqual(globalRegistryFactory.globalRegistry, noopRegistry{}) {
		for _, c := range cs {
			globalRegistryFactory.mustRegisterQueue = append(globalRegistryFactory.mustRegisterQueue, c)
		}
		return
	}
	globalRegistryFactory.globalRegistry.MustRegister(cs...)
	return
}
