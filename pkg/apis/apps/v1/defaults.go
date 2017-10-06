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

package v1

import (
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_DaemonSet(obj *appsv1.DaemonSet) {
	updateStrategy := &obj.Spec.UpdateStrategy
	if updateStrategy.Type == "" {
		updateStrategy.Type = appsv1.RollingUpdateDaemonSetStrategyType
	}
	if updateStrategy.Type == appsv1.RollingUpdateDaemonSetStrategyType {
		if updateStrategy.RollingUpdate == nil {
			rollingUpdate := appsv1.RollingUpdateDaemonSet{}
			updateStrategy.RollingUpdate = &rollingUpdate
		}
		if updateStrategy.RollingUpdate.MaxUnavailable == nil {
			// Set default MaxUnavailable as 1 by default.
			maxUnavailable := intstr.FromInt(1)
			updateStrategy.RollingUpdate.MaxUnavailable = &maxUnavailable
		}
	}
	if obj.Spec.RevisionHistoryLimit == nil {
		obj.Spec.RevisionHistoryLimit = new(int32)
		*obj.Spec.RevisionHistoryLimit = 10
	}
}
