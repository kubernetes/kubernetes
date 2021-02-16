/*
Copyright 2021 The Kubernetes Authors.

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
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// EphemeralVolumeSubsystem - subsystem name used for Endpoint Slices.
const EphemeralVolumeSubsystem = "ephemeral_volume_controller"

var (
	// EphemeralVolumeCreate tracks the number of
	// PersistentVolumeClaims().Create calls for each failure
	// reason
	// (https://pkg.go.dev/k8s.io/apimachinery@v0.20.2/pkg/apis/meta/v1#StatusReason),
	// with empty for successful calls.
	EphemeralVolumeCreate = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      EphemeralVolumeSubsystem,
			Name:           "create",
			Help:           "Number of PersistenVolumeClaims creation requests",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"reason"},
	)
)

var registerMetrics sync.Once

// RegisterMetrics registers EphemeralVolume metrics.
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(EphemeralVolumeCreate)
	})
}
