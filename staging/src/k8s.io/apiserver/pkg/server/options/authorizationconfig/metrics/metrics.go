/*
Copyright 2024 The Kubernetes Authors.

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
	"crypto/sha256"
	"fmt"
	"hash"
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "authorization_config_controller"
)

var (
	authorizationConfigAutomaticReloadsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reloads_total",
			Help:           "Total number of automatic reloads of authorization configuration split by status and apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status", "apiserver_id_hash"},
	)

	authorizationConfigAutomaticReloadLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_last_timestamp_seconds",
			Help:           "Timestamp of the last automatic reload of authorization configuration split by status and apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status", "apiserver_id_hash"},
	)
)

var registerMetrics sync.Once
var hashPool *sync.Pool

func RegisterMetrics() {
	registerMetrics.Do(func() {
		hashPool = &sync.Pool{
			New: func() interface{} {
				return sha256.New()
			},
		}
		legacyregistry.MustRegister(authorizationConfigAutomaticReloadsTotal)
		legacyregistry.MustRegister(authorizationConfigAutomaticReloadLastTimestampSeconds)
	})
}

func ResetMetricsForTest() {
	authorizationConfigAutomaticReloadsTotal.Reset()
	authorizationConfigAutomaticReloadLastTimestampSeconds.Reset()
	legacyregistry.Reset()
}

func RecordAuthorizationConfigAutomaticReloadFailure(apiServerID string) {
	apiServerIDHash := getHash(apiServerID)
	authorizationConfigAutomaticReloadsTotal.WithLabelValues("failure", apiServerIDHash).Inc()
	authorizationConfigAutomaticReloadLastTimestampSeconds.WithLabelValues("failure", apiServerIDHash).SetToCurrentTime()
}

func RecordAuthorizationConfigAutomaticReloadSuccess(apiServerID string) {
	apiServerIDHash := getHash(apiServerID)
	authorizationConfigAutomaticReloadsTotal.WithLabelValues("success", apiServerIDHash).Inc()
	authorizationConfigAutomaticReloadLastTimestampSeconds.WithLabelValues("success", apiServerIDHash).SetToCurrentTime()
}

func getHash(data string) string {
	if len(data) == 0 {
		return ""
	}
	h := hashPool.Get().(hash.Hash)
	h.Reset()
	h.Write([]byte(data))
	dataHash := fmt.Sprintf("sha256:%x", h.Sum(nil))
	hashPool.Put(h)
	return dataHash
}
