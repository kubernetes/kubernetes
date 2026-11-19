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

package prober

import (
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
)

type probeResultState struct {
	lastResult results.Result
	resultRun  int

	// proberResultsMetricLabels holds the labels attached to this worker
	// for the ProberResults metric by result.
	proberResultsSuccessfulMetricLabels metrics.Labels
	proberResultsFailedMetricLabels     metrics.Labels
	proberResultsUnknownMetricLabels    metrics.Labels
	// proberDurationMetricLabels holds the labels attached to this worker
	// for the ProberDuration metric by result.
	proberDurationMetricLabels metrics.Labels
}

func newProbeResultState(probeType probeType, pod *v1.Pod, container v1.Container) probeResultState {
	w := probeResultState{}
	basicMetricLabels := metrics.Labels{
		"probe_type": probeType.String(),
		"container":  container.Name,
		"pod":        pod.Name,
		"namespace":  pod.Namespace,
		"pod_uid":    string(pod.UID),
	}

	proberDurationLabels := metrics.Labels{
		"probe_type": probeType.String(),
		"container":  container.Name,
		"pod":        pod.Name,
		"namespace":  pod.Namespace,
	}

	w.proberResultsSuccessfulMetricLabels = deepCopyPrometheusLabels(basicMetricLabels)
	w.proberResultsSuccessfulMetricLabels["result"] = probeResultSuccessful

	w.proberResultsFailedMetricLabels = deepCopyPrometheusLabels(basicMetricLabels)
	w.proberResultsFailedMetricLabels["result"] = probeResultFailed

	w.proberResultsUnknownMetricLabels = deepCopyPrometheusLabels(basicMetricLabels)
	w.proberResultsUnknownMetricLabels["result"] = probeResultUnknown

	w.proberDurationMetricLabels = deepCopyPrometheusLabels(proberDurationLabels)

	return w
}

// recordResult records each attempt and reports whether its threshold is met.
func (w *probeResultState) recordResult(result results.Result, spec *v1.Probe, startTime time.Time) bool {
	switch result {
	case results.Success:
		ProberResults.With(w.proberResultsSuccessfulMetricLabels).Inc()
		ProberDuration.With(w.proberDurationMetricLabels).Observe(time.Since(startTime).Seconds())
	case results.Failure:
		ProberResults.With(w.proberResultsFailedMetricLabels).Inc()
	default:
		ProberResults.With(w.proberResultsUnknownMetricLabels).Inc()
		ProberDuration.With(w.proberDurationMetricLabels).Observe(time.Since(startTime).Seconds())
	}

	if w.lastResult == result {
		w.resultRun++
	} else {
		w.lastResult = result
		w.resultRun = 1
	}

	if (result == results.Failure && w.resultRun < int(spec.FailureThreshold)) ||
		(result == results.Success && w.resultRun < int(spec.SuccessThreshold)) {
		// Success or failure is below threshold - leave the probe state unchanged.
		return false
	}

	return true
}
