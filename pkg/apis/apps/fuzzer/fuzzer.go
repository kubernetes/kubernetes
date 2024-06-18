/*
Copyright 2017 The Kubernetes Authors.

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

package fuzzer

import (
	"fmt"

	fuzz "github.com/google/gofuzz"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/apps"
)

// Funcs returns the fuzzer functions for the apps api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(r *apps.ControllerRevision, c fuzz.Continue) {
			c.FuzzNoCustom(r)
			// match the fuzzer default content for runtime.Object
			r.Data = runtime.RawExtension{Raw: []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`)}
		},
		func(s *apps.StatefulSet, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again

			// match defaulter
			if len(s.Spec.PodManagementPolicy) == 0 {
				s.Spec.PodManagementPolicy = apps.OrderedReadyPodManagement
			}
			if len(s.Spec.UpdateStrategy.Type) == 0 {
				s.Spec.UpdateStrategy.Type = apps.RollingUpdateStatefulSetStrategyType
			}
			if s.Spec.PersistentVolumeClaimRetentionPolicy == nil {
				s.Spec.PersistentVolumeClaimRetentionPolicy = &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{}
			}
			if len(s.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted) == 0 {
				s.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted = apps.RetainPersistentVolumeClaimRetentionPolicyType
			}
			if len(s.Spec.PersistentVolumeClaimRetentionPolicy.WhenScaled) == 0 {
				s.Spec.PersistentVolumeClaimRetentionPolicy.WhenScaled = apps.RetainPersistentVolumeClaimRetentionPolicyType
			}
			if s.Spec.RevisionHistoryLimit == nil {
				s.Spec.RevisionHistoryLimit = new(int32)
				*s.Spec.RevisionHistoryLimit = 10
			}
			if s.Status.ObservedGeneration == nil {
				s.Status.ObservedGeneration = new(int64)
			}
			if s.Status.CollisionCount == nil {
				s.Status.CollisionCount = new(int32)
			}
			if s.Spec.Selector == nil {
				s.Spec.Selector = &metav1.LabelSelector{MatchLabels: s.Spec.Template.Labels}
			}
			if len(s.Labels) == 0 {
				s.Labels = s.Spec.Template.Labels
			}
		},
		func(j *apps.Deployment, c fuzz.Continue) {
			c.FuzzNoCustom(j)

			// match defaulting
			if j.Spec.Selector == nil {
				j.Spec.Selector = &metav1.LabelSelector{MatchLabels: j.Spec.Template.Labels}
			}
			if len(j.Labels) == 0 {
				j.Labels = j.Spec.Template.Labels
			}
		},
		func(j *apps.DeploymentSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			rhl := int32(c.Rand.Int31())
			pds := int32(c.Rand.Int31())
			j.RevisionHistoryLimit = &rhl
			j.ProgressDeadlineSeconds = &pds
		},
		func(j *apps.DeploymentStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []apps.DeploymentStrategyType{apps.RecreateDeploymentStrategyType, apps.RollingUpdateDeploymentStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != apps.RollingUpdateDeploymentStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := apps.RollingUpdateDeployment{}
				if c.RandBool() {
					rollingUpdate.MaxUnavailable = intstr.FromInt32(c.Rand.Int31())
					rollingUpdate.MaxSurge = intstr.FromInt32(c.Rand.Int31())
				} else {
					rollingUpdate.MaxSurge = intstr.FromString(fmt.Sprintf("%d%%", c.Rand.Int31()))
				}
				j.RollingUpdate = &rollingUpdate
			}
		},
		func(j *apps.DaemonSet, c fuzz.Continue) {
			c.FuzzNoCustom(j)

			// match defaulter
			j.Spec.Template.Generation = 0
			if len(j.ObjectMeta.Labels) == 0 {
				j.ObjectMeta.Labels = j.Spec.Template.ObjectMeta.Labels
			}
		},
		func(j *apps.DaemonSetSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			rhl := int32(c.Rand.Int31())
			j.RevisionHistoryLimit = &rhl
		},
		func(j *apps.DaemonSetUpdateStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []apps.DaemonSetUpdateStrategyType{apps.RollingUpdateDaemonSetStrategyType, apps.OnDeleteDaemonSetStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != apps.RollingUpdateDaemonSetStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := apps.RollingUpdateDaemonSet{}
				if c.RandBool() {
					if c.RandBool() {
						rollingUpdate.MaxUnavailable = intstr.FromInt32(c.Rand.Int31())
						rollingUpdate.MaxSurge = intstr.FromInt32(c.Rand.Int31())
					} else {
						rollingUpdate.MaxSurge = intstr.FromString(fmt.Sprintf("%d%%", c.Rand.Int31()))
					}
				}

				j.RollingUpdate = &rollingUpdate
			}
		},
		func(j *apps.ReplicaSet, c fuzz.Continue) {
			c.FuzzNoCustom(j)

			// match defaulter
			if j.Spec.Selector == nil {
				j.Spec.Selector = &metav1.LabelSelector{MatchLabels: j.Spec.Template.Labels}
			}
			if len(j.Labels) == 0 {
				j.Labels = j.Spec.Template.Labels
			}
		},
	}
}
