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

package podautoscaler

import (
	"testing"
	"time"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/ktesting"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

// syncPeriodTestWindow is how long the tests let the HPA controller run before
// they count the reconciliations of each HPA. With the global period of 1s, an
// HPA that uses it is reconciled about 15 times in this window.
const syncPeriodTestWindow = 15 * time.Second

// TestHPASyncPeriod verifies that an HPA that sets spec.syncPeriodSeconds is
// reconciled at its own period, while an HPA without it keeps using the global
// period of 1s.
func TestHPASyncPeriod(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	tCtx := ktesting.Init(t).WithRESTConfig(server.ClientConfig)
	clients, metricValue := createClients(tCtx)
	metricValue.Store(resource.MustParse("100"))
	startHPAControllerAndWaitForCaches(tCtx, clients)
	specSamplesBefore := countDelayRatioSamples(tCtx, "spec")

	globalNS := createTestNamespaceWithName(tCtx, "sync-period-global")
	createHPA(tCtx, createDeployment(tCtx, globalNS.Name, 1), newExternalMetricSpec())
	specNS := createTestNamespaceWithName(tCtx, "sync-period-spec")
	createHPA(tCtx, createDeployment(tCtx, specNS.Name, 1), newExternalMetricSpec(), withSyncPeriodSeconds(5))

	time.Sleep(syncPeriodTestWindow)

	globalQueries := countExternalMetricQueries(clients, globalNS.Name)
	specQueries := countExternalMetricQueries(clients, specNS.Name)
	tCtx.Logf("External metric queries in %v: %d with the global period, %d with spec.syncPeriodSeconds=5", syncPeriodTestWindow, globalQueries, specQueries)
	if globalQueries < 8 {
		tCtx.Errorf("The HPA without spec.syncPeriodSeconds queried its metric %d times, want at least 8", globalQueries)
	}
	if specQueries < 2 || specQueries > 6 {
		tCtx.Errorf("The HPA with spec.syncPeriodSeconds=5 queried its metric %d times, want between 2 and 6", specQueries)
	}
	if countDelayRatioSamples(tCtx, "spec") == specSamplesBefore {
		tCtx.Error("No reconciliation delay ratio was recorded for the HPA with spec.syncPeriodSeconds")
	}
}

// TestHPASyncPeriodFeatureGateDisabled verifies that the HPA controller ignores
// a stored spec.syncPeriodSeconds while the feature gate is disabled, and that
// the API server keeps the stored value.
func TestHPASyncPeriodFeatureGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	tCtx := ktesting.Init(t).WithRESTConfig(server.ClientConfig)
	clients, metricValue := createClients(tCtx)
	metricValue.Store(resource.MustParse("100"))

	ns := createTestNamespace(tCtx)
	createHPA(tCtx, createDeployment(tCtx, ns.Name, 1), newExternalMetricSpec(), withSyncPeriodSeconds(30))

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, false)
	startHPAControllerAndWaitForCaches(tCtx, clients)

	time.Sleep(syncPeriodTestWindow)

	queries := countExternalMetricQueries(clients, ns.Name)
	tCtx.Logf("External metric queries in %v: %d", syncPeriodTestWindow, queries)
	if queries < 8 {
		tCtx.Errorf("The HPA queried its metric %d times, want at least 8 at the global period", queries)
	}
	hpa, err := tCtx.Client().AutoscalingV2().HorizontalPodAutoscalers(ns.Name).Get(tCtx, hpaName, metav1.GetOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to get HPA: %v", err)
	}
	expectSyncPeriodSeconds(tCtx, "autoscaling/v2", hpa.Spec.SyncPeriodSeconds, 30)
}

// newExternalMetricSpec returns a metric spec with a target of 100 per replica,
// so that a metric value of 100 keeps a single replica.
func newExternalMetricSpec() autoscalingv2.MetricSpec {
	return autoscalingv2.MetricSpec{
		Type: autoscalingv2.ExternalMetricSourceType,
		External: &autoscalingv2.ExternalMetricSource{
			Metric: autoscalingv2.MetricIdentifier{
				Name: externalMetricName,
			},
			Target: autoscalingv2.MetricTarget{
				Type:         autoscalingv2.AverageValueMetricType,
				AverageValue: new(resource.MustParse("100")),
			},
		},
	}
}

func expectSyncPeriodSeconds(tCtx ktesting.TContext, apiVersion string, got *int32, want int32) {
	tCtx.Helper()
	if got == nil {
		tCtx.Errorf("spec.syncPeriodSeconds read through %s is unset, want %d", apiVersion, want)
		return
	}
	if *got != want {
		tCtx.Errorf("spec.syncPeriodSeconds read through %s = %d, want %d", apiVersion, *got, want)
	}
}

// countDelayRatioSamples returns how many reconciliation delay ratios with the
// period source label the HPA controllers of this process have recorded.
func countDelayRatioSamples(tCtx ktesting.TContext, periodSource string) uint64 {
	tCtx.Helper()
	families, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		tCtx.Fatalf("Failed to gather metrics: %v", err)
	}
	var count uint64
	for _, family := range families {
		if family.GetName() != "horizontal_pod_autoscaler_controller_reconciliation_delay_ratio" {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() == "period_source" && label.GetValue() == periodSource {
					count += metric.GetHistogram().GetSampleCount()
				}
			}
		}
	}
	return count
}
