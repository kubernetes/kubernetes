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

package monitor

import (
	"strings"
	"testing"
	"time"

	"k8s.io/component-base/metrics/testutil"

	v2 "k8s.io/api/autoscaling/v2"
)

func TestReconciliationsTotalMetric(t *testing.T) {
	Register()
	reconciliationsTotal.Reset()

	m := New()
	m.ObserveReconciliationResult(ActionLabelScaleUp, ErrorLabelNone, 100*time.Millisecond)

	expected := `
        # HELP horizontal_pod_autoscaler_controller_reconciliations_total [ALPHA] Number of reconciliations of HPA controller. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. Also, the label 'error' should be either 'spec', 'internal', or 'none'. Note that if both spec and internal errors happen during a reconciliation, the first one to occur is reported in ` + "`error`" + ` label.
        # TYPE horizontal_pod_autoscaler_controller_reconciliations_total counter
        horizontal_pod_autoscaler_controller_reconciliations_total{action="scale_up",error="none"} 1
    `
	if err := testutil.CollectAndCompare(reconciliationsTotal, strings.NewReader(expected)); err != nil {
		t.Errorf("unexpected metric: %v", err)
	}
}

func TestReconciliationsDurationMetric(t *testing.T) {
	Register()
	reconciliationsDuration.Reset()

	m := New()
	m.ObserveReconciliationResult(ActionLabelScaleDown, ErrorLabelInternal, 150*time.Millisecond)

	expected := `
        # HELP horizontal_pod_autoscaler_controller_reconciliation_duration_seconds [ALPHA] The time(seconds) that the HPA controller takes to reconcile once. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. Also, the label 'error' should be either 'spec', 'internal', or 'none'. Note that if both spec and internal errors happen during a reconciliation, the first one to occur is reported in ` + "`error`" + ` label.
        # TYPE horizontal_pod_autoscaler_controller_reconciliation_duration_seconds histogram
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.001"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.002"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.004"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.008"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.016"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.032"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.064"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.128"} 0
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.256"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="0.512"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="1.024"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="2.048"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="4.096"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="8.192"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="16.384"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_bucket{action="scale_down",error="internal",le="+Inf"} 1
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_sum{action="scale_down",error="internal"} 0.15
        horizontal_pod_autoscaler_controller_reconciliation_duration_seconds_count{action="scale_down",error="internal"} 1
    `
	if err := testutil.CollectAndCompare(reconciliationsDuration, strings.NewReader(expected)); err != nil {
		t.Errorf("unexpected metric: %v", err)
	}
}

func TestMetricComputationTotalMetric(t *testing.T) {
	Register()
	metricComputationTotal.Reset()

	m := New()
	m.ObserveMetricComputationResult(ActionLabelNone, ErrorLabelSpec, 50*time.Millisecond, v2.ResourceMetricSourceType)

	expected := `
        # HELP horizontal_pod_autoscaler_controller_metric_computation_total [ALPHA] Number of metric computations. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. Also, the label 'error' should be either 'spec', 'internal', or 'none'. The label 'metric_type' corresponds to HPA.spec.metrics[*].type
        # TYPE horizontal_pod_autoscaler_controller_metric_computation_total counter
        horizontal_pod_autoscaler_controller_metric_computation_total{action="none",error="spec",metric_type="Resource"} 1
    `
	if err := testutil.CollectAndCompare(metricComputationTotal, strings.NewReader(expected)); err != nil {
		t.Errorf("unexpected metric: %v", err)
	}
}

func TestMetricComputationDurationMetric(t *testing.T) {
	Register()
	metricComputationDuration.Reset()

	m := New()
	m.ObserveMetricComputationResult(ActionLabelScaleUp, ErrorLabelNone, 25*time.Millisecond, v2.PodsMetricSourceType)

	expected := `
        # HELP horizontal_pod_autoscaler_controller_metric_computation_duration_seconds [ALPHA] The time(seconds) that the HPA controller takes to calculate one metric. The label 'action' should be either 'scale_down', 'scale_up', or 'none'. The label 'error' should be either 'spec', 'internal', or 'none'. The label 'metric_type' corresponds to HPA.spec.metrics[*].type
        # TYPE horizontal_pod_autoscaler_controller_metric_computation_duration_seconds histogram
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.001"} 0
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.002"} 0
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.004"} 0
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.008"} 0
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.016"} 0
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.032"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.064"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.128"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.256"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="0.512"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="1.024"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="2.048"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="4.096"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="8.192"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="16.384"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_bucket{action="scale_up",error="none",metric_type="Pods",le="+Inf"} 1
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_sum{action="scale_up",error="none",metric_type="Pods"} 0.025
        horizontal_pod_autoscaler_controller_metric_computation_duration_seconds_count{action="scale_up",error="none",metric_type="Pods"} 1
    `
	if err := testutil.CollectAndCompare(metricComputationDuration, strings.NewReader(expected)); err != nil {
		t.Errorf("unexpected metric: %v", err)
	}
}

func TestHPAAdditionAndDeletionMetrics(t *testing.T) {
	Register()
	numHorizontalPodAutoscalers.Set(0)

	m := New()

	m.ObserveHPAAddition()
	m.ObserveHPAAddition()

	expected := `
        # HELP horizontal_pod_autoscaler_controller_num_horizontal_pod_autoscalers [ALPHA] Current number of controlled HPA objects.
        # TYPE horizontal_pod_autoscaler_controller_num_horizontal_pod_autoscalers gauge
        horizontal_pod_autoscaler_controller_num_horizontal_pod_autoscalers 2
    `
	if err := testutil.CollectAndCompare(numHorizontalPodAutoscalers, strings.NewReader(expected)); err != nil {
		t.Errorf("unexpected metric after additions: %v", err)
	}

	m.ObserveHPADeletion()

	expected = `
        # HELP horizontal_pod_autoscaler_controller_num_horizontal_pod_autoscalers [ALPHA] Current number of controlled HPA objects.
        # TYPE horizontal_pod_autoscaler_controller_num_horizontal_pod_autoscalers gauge
        horizontal_pod_autoscaler_controller_num_horizontal_pod_autoscalers 1
    `
	if err := testutil.CollectAndCompare(numHorizontalPodAutoscalers, strings.NewReader(expected)); err != nil {
		t.Errorf("unexpected metric after deletion: %v", err)
	}
}

func TestDesiredReplicasMetric(t *testing.T) {
	Register()
	desiredReplicasCount.Reset()

	m := New()
	m.ObserveDesiredReplicas("default", "test-hpa", 5)
	m.ObserveDesiredReplicas("kube-system", "system-hpa", 10)

	expected := `
        # HELP horizontal_pod_autoscaler_controller_desired_replicas [ALPHA] Current desired replica count for HPA objects.
        # TYPE horizontal_pod_autoscaler_controller_desired_replicas gauge
        horizontal_pod_autoscaler_controller_desired_replicas{hpa_name="system-hpa",namespace="kube-system"} 10
        horizontal_pod_autoscaler_controller_desired_replicas{hpa_name="test-hpa",namespace="default"} 5
    `
	if err := testutil.CollectAndCompare(desiredReplicasCount, strings.NewReader(expected)); err != nil {
		t.Errorf("unexpected metric: %v", err)
	}
}
