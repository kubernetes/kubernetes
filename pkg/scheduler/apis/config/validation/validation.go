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

package validation

import (
	apimachinery "k8s.io/apimachinery/pkg/apis/config/validation"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apiserver "k8s.io/apiserver/pkg/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// ValidateKubeSchedulerConfiguration ensures validation of the KubeSchedulerConfiguration struct
func ValidateKubeSchedulerConfiguration(cc *config.KubeSchedulerConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apimachinery.ValidateClientConnectionConfiguration(&cc.ClientConnection, field.NewPath("clientConnection"))...)
	allErrs = append(allErrs, ValidateKubeSchedulerLeaderElectionConfiguration(&cc.LeaderElection, field.NewPath("leaderElection"))...)
	if len(cc.SchedulerName) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("schedulerName"), ""))
	}
	for _, msg := range validation.IsValidSocketAddr(cc.HealthzBindAddress) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("healthzBindAddress"), cc.HealthzBindAddress, msg))
	}
	for _, msg := range validation.IsValidSocketAddr(cc.MetricsBindAddress) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("metricsBindAddress"), cc.MetricsBindAddress, msg))
	}
	if cc.HardPodAffinitySymmetricWeight < 0 || cc.HardPodAffinitySymmetricWeight > 100 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("hardPodAffinitySymmetricWeight"), cc.HardPodAffinitySymmetricWeight, "not in valid range 0-100"))
	}
	if cc.BindTimeoutSeconds == nil {
		allErrs = append(allErrs, field.Required(field.NewPath("bindTimeoutSeconds"), ""))
	}
	if cc.PercentageOfNodesToScore < 0 || cc.PercentageOfNodesToScore > 100 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("percentageOfNodesToScore"),
			cc.PercentageOfNodesToScore, "not in valid range 0-100"))
	}
	return allErrs
}

// ValidateKubeSchedulerLeaderElectionConfiguration ensures validation of the KubeSchedulerLeaderElectionConfiguration struct
func ValidateKubeSchedulerLeaderElectionConfiguration(cc *config.KubeSchedulerLeaderElectionConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !cc.LeaderElectionConfiguration.LeaderElect {
		return allErrs
	}
	allErrs = append(allErrs, apiserver.ValidateLeaderElectionConfiguration(&cc.LeaderElectionConfiguration, field.NewPath("leaderElectionConfiguration"))...)
	if len(cc.LockObjectNamespace) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("lockObjectNamespace"), ""))
	}
	if len(cc.LockObjectName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("lockObjectName"), ""))
	}
	return allErrs
}
