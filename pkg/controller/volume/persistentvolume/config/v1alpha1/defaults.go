/*
Copyright 2019 The Kubernetes Authors.

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
	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
	"k8s.io/utils/pointer"
)

// RecommendedDefaultPersistentVolumeBinderControllerConfiguration defaults a pointer to a
// PersistentVolumeBinderControllerConfiguration struct. This will set the recommended default
// values, but they may be subject to change between API versions. This function
// is intentionally not registered in the scheme as a "normal" `SetDefaults_Foo`
// function to allow consumers of this type to set whatever defaults for their
// embedded configs. Forcing consumers to use these defaults would be problematic
// as defaulting in the scheme is done as part of the conversion, and there would
// be no easy way to opt-out. Instead, if you want to use this defaulting method
// run it in your wrapper struct of this type in its `SetDefaults_` method.
func RecommendedDefaultPersistentVolumeBinderControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.PersistentVolumeBinderControllerConfiguration) {
	zero := metav1.Duration{}
	if obj.PVClaimBinderSyncPeriod == zero {
		obj.PVClaimBinderSyncPeriod = metav1.Duration{Duration: 15 * time.Second}
	}

	if obj.VolumeHostAllowLocalLoopback == nil {
		trueValue := true
		obj.VolumeHostAllowLocalLoopback = &trueValue
	}

	// Use the default VolumeConfiguration options.
	RecommendedDefaultVolumeConfiguration(&obj.VolumeConfiguration)
}

// RecommendedDefaultVolumeConfiguration defaults a pointer to a VolumeConfiguration
// struct. This will set the recommended default values, but they may be subject to
// change between API versions. This function is intentionally not registered in the
// scheme as a "normal" `SetDefaults_Foo` function to allow consumers of this type to
// set whatever defaults for their embedded configs. Forcing consumers to use these
// defaults would be problematic as defaulting in the scheme is done as part of the
// conversion, and there would be no easy way to opt-out. Instead, if you want to use
// this defaulting method run it in your wrapper struct of this type in its `SetDefaults_` method.
func RecommendedDefaultVolumeConfiguration(obj *kubectrlmgrconfigv1alpha1.VolumeConfiguration) {
	if obj.EnableHostPathProvisioning == nil {
		obj.EnableHostPathProvisioning = pointer.Bool(false)
	}
	if obj.EnableDynamicProvisioning == nil {
		obj.EnableDynamicProvisioning = pointer.Bool(true)
	}
	if obj.FlexVolumePluginDir == "" {
		obj.FlexVolumePluginDir = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"
	}
	// Use the default PersistentVolumeRecyclerConfiguration options.
	RecommendedDefaultPersistentVolumeRecyclerConfiguration(&obj.PersistentVolumeRecyclerConfiguration)
}

// RecommendedDefaultPersistentVolumeRecyclerConfiguration defaults a pointer to a
// PersistentVolumeRecyclerConfiguration struct. This will set the recommended default
// values, but they may be subject to change between API versions. This function
// is intentionally not registered in the scheme as a "normal" `SetDefaults_Foo`
// function to allow consumers of this type to set whatever defaults for their
// embedded configs. Forcing consumers to use these defaults would be problematic
// as defaulting in the scheme is done as part of the conversion, and there would
// be no easy way to opt-out. Instead, if you want to use this defaulting method
// run it in your wrapper struct of this type in its `SetDefaults_` method.
func RecommendedDefaultPersistentVolumeRecyclerConfiguration(obj *kubectrlmgrconfigv1alpha1.PersistentVolumeRecyclerConfiguration) {
	if obj.MaximumRetry == 0 {
		obj.MaximumRetry = 3
	}
	if obj.MinimumTimeoutNFS == 0 {
		obj.MinimumTimeoutNFS = 300
	}
	if obj.IncrementTimeoutNFS == 0 {
		obj.IncrementTimeoutNFS = 30
	}
	if obj.MinimumTimeoutHostPath == 0 {
		obj.MinimumTimeoutHostPath = 60
	}
	if obj.IncrementTimeoutHostPath == 0 {
		obj.IncrementTimeoutHostPath = 30
	}
}
