//go:build !providerless
// +build !providerless

/*
Copyright 2020 The Kubernetes Authors.

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

package gce

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

const (
	label = "feature"
)

var (
	metricsInterval = 10 * time.Minute
	l4ILBCount      = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name:           "number_of_l4_ilbs",
			Help:           "Number of L4 ILBs",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{label},
	)
)

// init registers L4 internal loadbalancer usage metrics.
func init() {
	klog.V(3).Infof("Registering Service Controller loadbalancer usage metrics %v", l4ILBCount)
	legacyregistry.MustRegister(l4ILBCount)
}

// LoadBalancerMetrics is a cache that contains loadbalancer service resource
// states for computing usage metrics.
type LoadBalancerMetrics struct {
	// l4ILBServiceMap is a map of service key and L4 ILB service state.
	l4ILBServiceMap map[string]L4ILBServiceState

	sync.Mutex
}

type feature string

func (f feature) String() string {
	return string(f)
}

const (
	l4ILBService      = feature("L4ILBService")
	l4ILBGlobalAccess = feature("L4ILBGlobalAccess")
	l4ILBCustomSubnet = feature("L4ILBCustomSubnet")
	// l4ILBInSuccess feature specifies that ILB VIP is configured.
	l4ILBInSuccess = feature("L4ILBInSuccess")
	// l4ILBInInError feature specifies that an error had occurred for this service
	// in ensureInternalLoadbalancer method.
	l4ILBInError = feature("L4ILBInError")
)

// L4ILBServiceState contains Internal Loadbalancer feature states as specified
// in k8s Service.
type L4ILBServiceState struct {
	// EnabledGlobalAccess specifies if Global Access is enabled.
	EnabledGlobalAccess bool
	// EnabledCustomSubNet specifies if Custom Subnet is enabled.
	EnabledCustomSubnet bool
	// InSuccess specifies if the ILB service VIP is configured.
	InSuccess bool
}

// loadbalancerMetricsCollector is an interface to update/delete L4 loadbalancer
// states in the cache that is used for computing L4 Loadbalancer usage metrics.
type loadbalancerMetricsCollector interface {
	// Run starts a goroutine to compute and export metrics a periodic interval.
	Run(stopCh <-chan struct{})
	// SetL4ILBService adds/updates L4 ILB service state for given service key.
	SetL4ILBService(svcKey string, state L4ILBServiceState)
	// DeleteL4ILBService removes the given L4 ILB service key.
	DeleteL4ILBService(svcKey string)
}

// newLoadBalancerMetrics initializes LoadBalancerMetrics and starts a goroutine
// to compute and export metrics periodically.
func newLoadBalancerMetrics() loadbalancerMetricsCollector {
	return &LoadBalancerMetrics{
		l4ILBServiceMap: make(map[string]L4ILBServiceState),
	}
}

// Run implements loadbalancerMetricsCollector.
func (lm *LoadBalancerMetrics) Run(stopCh <-chan struct{}) {
	klog.V(3).Infof("Loadbalancer Metrics initialized. Metrics will be exported at an interval of %v", metricsInterval)
	// Compute and export metrics periodically.
	select {
	case <-stopCh:
		return
	case <-time.After(metricsInterval): // Wait for service states to be populated in the cache before computing metrics.
		wait.Until(lm.export, metricsInterval, stopCh)
	}
}

// SetL4ILBService implements loadbalancerMetricsCollector.
func (lm *LoadBalancerMetrics) SetL4ILBService(svcKey string, state L4ILBServiceState) {
	lm.Lock()
	defer lm.Unlock()

	if lm.l4ILBServiceMap == nil {
		klog.Fatalf("Loadbalancer Metrics failed to initialize correctly.")
	}
	lm.l4ILBServiceMap[svcKey] = state
}

// DeleteL4ILBService implements loadbalancerMetricsCollector.
func (lm *LoadBalancerMetrics) DeleteL4ILBService(svcKey string) {
	lm.Lock()
	defer lm.Unlock()

	delete(lm.l4ILBServiceMap, svcKey)
}

// export computes and exports loadbalancer usage metrics.
func (lm *LoadBalancerMetrics) export() {
	ilbCount := lm.computeL4ILBMetrics()
	klog.V(5).Infof("Exporting L4 ILB usage metrics: %#v", ilbCount)
	for feature, count := range ilbCount {
		l4ILBCount.With(map[string]string{label: feature.String()}).Set(float64(count))
	}
	klog.V(5).Infof("L4 ILB usage metrics exported.")
}

// computeL4ILBMetrics aggregates L4 ILB metrics in the cache.
func (lm *LoadBalancerMetrics) computeL4ILBMetrics() map[feature]int {
	lm.Lock()
	defer lm.Unlock()
	klog.V(4).Infof("Computing L4 ILB usage metrics from service state map: %#v", lm.l4ILBServiceMap)
	counts := map[feature]int{
		l4ILBService:      0,
		l4ILBGlobalAccess: 0,
		l4ILBCustomSubnet: 0,
		l4ILBInSuccess:    0,
		l4ILBInError:      0,
	}

	for key, state := range lm.l4ILBServiceMap {
		klog.V(6).Infof("ILB Service %s has EnabledGlobalAccess: %t, EnabledCustomSubnet: %t, InSuccess: %t", key, state.EnabledGlobalAccess, state.EnabledCustomSubnet, state.InSuccess)
		counts[l4ILBService]++
		if !state.InSuccess {
			counts[l4ILBInError]++
			// Skip counting other features if the service is in error state.
			continue
		}
		counts[l4ILBInSuccess]++
		if state.EnabledGlobalAccess {
			counts[l4ILBGlobalAccess]++
		}
		if state.EnabledCustomSubnet {
			counts[l4ILBCustomSubnet]++
		}
	}
	klog.V(4).Info("L4 ILB usage metrics computed.")
	return counts
}
