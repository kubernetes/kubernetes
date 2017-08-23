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
	fuzz "github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/apps"
)

// Funcs returns the fuzzer functions for the apps api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(s *apps.StatefulSet, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again

			// match defaulter
			if len(s.Spec.PodManagementPolicy) == 0 {
				s.Spec.PodManagementPolicy = apps.OrderedReadyPodManagement
			}
			if len(s.Spec.UpdateStrategy.Type) == 0 {
				s.Spec.UpdateStrategy.Type = apps.RollingUpdateStatefulSetStrategyType
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
		},
	}
}
