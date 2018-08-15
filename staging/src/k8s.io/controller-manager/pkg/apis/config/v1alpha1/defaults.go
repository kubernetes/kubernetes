/*
Copyright 2018 The Kubernetes Authors.

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

	apimachineryconfigv1alpha1 "k8s.io/apimachinery/pkg/apis/config/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserverconfigv1alpha1 "k8s.io/apiserver/pkg/apis/config/v1alpha1"
)

// RecommendedDefaultGenericControllerManagerConfiguration defaults a pointer to a
// GenericControllerManagerConfiguration struct. This will set the recommended default
// values, but they may be subject to change between API versions. This function
// is intentionally not registered in the scheme as a "normal" `SetDefaults_Foo`
// function to allow consumers of this type to set whatever defaults for their
// embedded configs. Forcing consumers to use these defaults would be problematic
// as defaulting in the scheme is done as part of the conversion, and there would
// be no easy way to opt-out. Instead, if you want to use this defaulting method
// run it in your wrapper struct of this type in its `SetDefaults_` method.
func RecommendedDefaultGenericControllerManagerConfiguration(obj *GenericControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if obj.Address == "" {
		obj.Address = "0.0.0.0"
	}
	if obj.MinResyncPeriod == zero {
		obj.MinResyncPeriod = metav1.Duration{Duration: 12 * time.Hour}
	}
	if obj.ControllerStartInterval == zero {
		obj.ControllerStartInterval = metav1.Duration{Duration: 0 * time.Second}
	}
	if len(obj.Controllers) == 0 {
		obj.Controllers = []string{"*"}
	}

	// Use the default ClientConnectionConfiguration and LeaderElectionConfiguration options
	apimachineryconfigv1alpha1.RecommendedDefaultClientConnectionConfiguration(&obj.ClientConnection)
	apiserverconfigv1alpha1.RecommendedDefaultLeaderElectionConfiguration(&obj.LeaderElection)

}
