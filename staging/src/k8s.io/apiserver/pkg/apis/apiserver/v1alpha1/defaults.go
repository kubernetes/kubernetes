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

package v1alpha1

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

var (
	defaultEventAdditionalDuration        = 5 * time.Millisecond
	defaultMaximumSeats            uint64 = 10
	defaultObjectsPerSeat                 = 100.0
	defaultWatchesPerSeat                 = 10.0
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_WorkEstimatorConfiguration(obj *WorkEstimatorConfiguration) {
	if obj.MaximumSeats == nil {
		obj.MaximumSeats = &defaultMaximumSeats
	}
}

func SetDefaults_ListWorkEstimatorConfiguration(obj *ListWorkEstimatorConfiguration) {
	if obj.ObjectsPerSeat == nil {
		obj.ObjectsPerSeat = &defaultObjectsPerSeat
	}
}

func SetDefaults_MutatingWorkEstimatorConfiguration(obj *MutatingWorkEstimatorConfiguration) {
	if obj.EventAdditionalDuration == nil {
		obj.EventAdditionalDuration = &metav1.Duration{Duration: defaultEventAdditionalDuration}
	}
	if obj.WatchesPerSeat == nil {
		obj.WatchesPerSeat = &defaultWatchesPerSeat
	}
}
