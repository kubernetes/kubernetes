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
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/ktesting"
	clienttesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

// syncPeriodTestWindow is how long the tests let the HPA controller run before
// they count the reconciliations of each HPA. With the global period of 1s, an
// HPA that uses it is reconciled about 15 times in this window.
const syncPeriodTestWindow = 15 * time.Second

// TestHPASyncPeriodRoundTrip verifies that spec.syncPeriodSeconds is preserved
// when an HPA is written and read through both autoscaling/v1 and
// autoscaling/v2, and that the API server rejects out-of-range values.
func TestHPASyncPeriodRoundTrip(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	tCtx := ktesting.Init(t).WithRESTConfig(server.ClientConfig)

	ns := createTestNamespace(tCtx)
	deployment := createDeployment(tCtx, ns.Name, 1)
	v1HPAs := tCtx.Client().AutoscalingV1().HorizontalPodAutoscalers(ns.Name)
	v2HPAs := tCtx.Client().AutoscalingV2().HorizontalPodAutoscalers(ns.Name)

	tCtx.Run("created with autoscaling/v2", func(tCtx ktesting.TContext) {
		hpa := newHPA(deployment, newExternalMetricSpec(), withSyncPeriodSeconds(30))
		hpa.Name = "created-with-v2"
		if _, err := v2HPAs.Create(tCtx, hpa, metav1.CreateOptions{}); err != nil {
			tCtx.Fatalf("Failed to create HPA: %v", err)
		}

		v1HPA, err := v1HPAs.Get(tCtx, hpa.Name, metav1.GetOptions{})
		if err != nil {
			tCtx.Fatalf("Failed to get HPA through autoscaling/v1: %v", err)
		}
		expectSyncPeriodSeconds(tCtx, "autoscaling/v1", v1HPA.Spec.SyncPeriodSeconds, 30)

		v1HPA.Spec.SyncPeriodSeconds = ptr.To[int32](60)
		if _, err := v1HPAs.Update(tCtx, v1HPA, metav1.UpdateOptions{}); err != nil {
			tCtx.Fatalf("Failed to update HPA through autoscaling/v1: %v", err)
		}

		v2HPA, err := v2HPAs.Get(tCtx, hpa.Name, metav1.GetOptions{})
		if err != nil {
			tCtx.Fatalf("Failed to get HPA through autoscaling/v2: %v", err)
		}
		expectSyncPeriodSeconds(tCtx, "autoscaling/v2", v2HPA.Spec.SyncPeriodSeconds, 60)
	})

	tCtx.Run("created with autoscaling/v1", func(tCtx ktesting.TContext) {
		hpa := &autoscalingv1.HorizontalPodAutoscaler{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "created-with-v1",
				Namespace: ns.Name,
			},
			Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
				ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       deployment.Name,
				},
				MaxReplicas:                    10,
				TargetCPUUtilizationPercentage: ptr.To[int32](50),
				SyncPeriodSeconds:              ptr.To[int32](45),
			},
		}
		if _, err := v1HPAs.Create(tCtx, hpa, metav1.CreateOptions{}); err != nil {
			tCtx.Fatalf("Failed to create HPA: %v", err)
		}

		v2HPA, err := v2HPAs.Get(tCtx, hpa.Name, metav1.GetOptions{})
		if err != nil {
			tCtx.Fatalf("Failed to get HPA through autoscaling/v2: %v", err)
		}
		expectSyncPeriodSeconds(tCtx, "autoscaling/v2", v2HPA.Spec.SyncPeriodSeconds, 45)

		v2HPA.Spec.SyncPeriodSeconds = ptr.To[int32](90)
		if _, err := v2HPAs.Update(tCtx, v2HPA, metav1.UpdateOptions{}); err != nil {
			tCtx.Fatalf("Failed to update HPA through autoscaling/v2: %v", err)
		}

		v1HPA, err := v1HPAs.Get(tCtx, hpa.Name, metav1.GetOptions{})
		if err != nil {
			tCtx.Fatalf("Failed to get HPA through autoscaling/v1: %v", err)
		}
		expectSyncPeriodSeconds(tCtx, "autoscaling/v1", v1HPA.Spec.SyncPeriodSeconds, 90)
	})

	tCtx.Run("out-of-range values are rejected", func(tCtx ktesting.TContext) {
		for _, seconds := range []int32{2, 3601} {
			hpa := newHPA(deployment, newExternalMetricSpec(), withSyncPeriodSeconds(seconds))
			hpa.Name = fmt.Sprintf("sync-period-%d", seconds)
			_, err := v2HPAs.Create(tCtx, hpa, metav1.CreateOptions{})
			if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "spec.syncPeriodSeconds") {
				tCtx.Errorf("Expected an invalid error for spec.syncPeriodSeconds=%d, got: %v", seconds, err)
			}
		}
	})
}

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

func TestHPASyncPeriodUpdates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	tCtx := ktesting.Init(t).WithRESTConfig(server.ClientConfig)
	clients, metricValue := createClients(tCtx)
	metricValue.Store(resource.MustParse("100"))

	queryTimes := make(chan time.Time, 100)
	clients.externalMetrics.PrependReactor("list", "*", func(action clienttesting.Action) (bool, runtime.Object, error) {
		select {
		case queryTimes <- time.Now():
		default:
		}
		return false, nil, nil
	})
	startHPAControllerAndWaitForCaches(tCtx, clients)

	ns := createTestNamespace(tCtx)
	hpa := createHPA(tCtx, createDeployment(tCtx, ns.Name, 1), newExternalMetricSpec())
	hpas := tCtx.Client().AutoscalingV2().HorizontalPodAutoscalers(ns.Name)

	for _, step := range []struct {
		name       string
		patch      string
		wantPeriod time.Duration
	}{
		{name: "initial global period", wantPeriod: hpaControllerResyncPeriod},
		{name: "set short period", patch: `{"spec":{"syncPeriodSeconds":3}}`, wantPeriod: 3 * time.Second},
		{name: "increase period", patch: `{"spec":{"syncPeriodSeconds":8}}`, wantPeriod: 8 * time.Second},
		{name: "restore global period", patch: `{"spec":{"syncPeriodSeconds":null}}`, wantPeriod: hpaControllerResyncPeriod},
	} {
		tCtx.Run(step.name, func(tCtx ktesting.TContext) {
			if step.patch != "" {
				var err error
				hpa, err = hpas.Patch(tCtx, hpa.Name, types.MergePatchType, []byte(step.patch), metav1.PatchOptions{})
				if err != nil {
					tCtx.Fatalf("Failed to update spec.syncPeriodSeconds: %v", err)
				}
			}
			if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 20*time.Second, true, func(ctx context.Context) (bool, error) {
				observed, err := hpas.Get(ctx, hpa.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return observed.Status.ObservedGeneration != nil && *observed.Status.ObservedGeneration == hpa.Generation, nil
			}); err != nil {
				tCtx.Fatalf("Controller did not observe HPA generation %d: %v", hpa.Generation, err)
			}

			// Spec changes reconcile immediately, and increasing the period can
			// leave one earlier timer queued. Measure intervals after the next
			// reconciliation to exclude both from the periodic cadence.
			for len(queryTimes) > 0 {
				<-queryTimes
			}
			timer := time.NewTimer(3*step.wantPeriod + 5*time.Second)
			defer timer.Stop()
			var previous time.Time
			for sample := range 3 {
				select {
				case observed := <-queryTimes:
					if !previous.IsZero() {
						interval := observed.Sub(previous)
						tCtx.Logf("External metric queries were %v apart with period %v", interval, step.wantPeriod)
						if interval < step.wantPeriod*4/5 {
							tCtx.Errorf("External metric queries were %v apart, want approximately %v", interval, step.wantPeriod)
						}
					}
					previous = observed
				case <-timer.C:
					tCtx.Fatalf("Observed %d of 3 external metric queries within %v after setting period %v", sample, 3*step.wantPeriod+5*time.Second, step.wantPeriod)
				case <-tCtx.Done():
					tCtx.Fatalf("Waiting for external metric queries: %v", tCtx.Err())
				}
			}
		})
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
