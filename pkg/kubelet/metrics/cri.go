/*
Copyright The Kubernetes Authors.

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
	"context"
	"sync"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	crimetrics "k8s.io/cri-client/pkg/metrics"
	"k8s.io/kubernetes/pkg/features"
)

var registerCRIStreamingMetricsOnce sync.Once

// RegisterCRIStreamingMetrics must be called before CRI clients are used so
// startup failures and cached fallbacks are recorded before Register runs.
func RegisterCRIStreamingMetrics() {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CRIListStreaming) {
		return
	}
	registerCRIStreamingMetricsOnce.Do(func() {
		legacyregistry.MustRegister(CRIListStreamingTotal)
		crimetrics.Register(crimetrics.RegisterOpts{
			ListStreamingTotal: &criListStreamingAdapter{CRIListStreamingTotal},
		})
	})
}

type criListStreamingAdapter struct {
	counter *metrics.CounterVec
}

func (a *criListStreamingAdapter) Increment(ctx context.Context, operation, result string) {
	a.counter.WithContext(ctx).WithLabelValues(operation, result).Inc()
}
