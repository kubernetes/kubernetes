/*
Copyright 2022 The Kubernetes Authors.

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

package statefulset

import (
	"sync"

	apps "k8s.io/api/apps/v1"
	"k8s.io/component-base/metrics"
)

// StatefulSet is the subsystem name used by this package.
const StatefulSet = "statefulset"

var (
	policyCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      StatefulSet,
			Name:           "statefulset_reconcile_policy_total",
			Help:           "Count of PVC retention policies seen during reconcile of StatefulSets.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"deleted_policy", "scaled_policy"},
	)
	reconcileSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      StatefulSet,
			Name:           "statefulset_reconcile_seconds",
			Help:           "StatefulSet reconcile duration in seconds.",
			Buckets:        metrics.ExponentialBuckets(1, 1.5, 10),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)
	unhealthyPodsCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      StatefulSet,
			Name:           "statefulset_unhealthy_pods_total",
			Help:           "Count of StatefulSets reconciles that saw an unhealthy pod.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)
)

func recordRetentionPolicyMetrics(set *apps.StatefulSet) {
	if set == nil {
		return
	}
	policy := set.Spec.PersistentVolumeClaimRetentionPolicy
	if policy == nil {
		policy = &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{}
	}
	deletedPolicy := policy.WhenDeleted
	if deletedPolicy == "" {
		deletedPolicy = apps.RetainPersistentVolumeClaimRetentionPolicyType
	}
	scaledPolicy := policy.WhenScaled
	if scaledPolicy == "" {
		scaledPolicy = apps.RetainPersistentVolumeClaimRetentionPolicyType
	}
	policyCounter.WithLabelValues(string(deletedPolicy), string(scaledPolicy)).Inc()
}

var once sync.Once

func registerMetrics() {
	once.Do(func() {
		r := metrics.NewKubeRegistry()
		r.MustRegister(policyCounter)
		r.MustRegister(reconcileSeconds)
		r.MustRegister(unhealthyPodsCounter)
	})
}
