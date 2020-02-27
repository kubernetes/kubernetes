/*
Copyright 2020 The Kubernetes Authors.

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

package v1alpha2

import (
	conversion "k8s.io/apimachinery/pkg/conversion"
	v1alpha2 "k8s.io/kube-scheduler/config/v1alpha2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"
)

func Convert_v1alpha2_KubeSchedulerConfiguration_To_config_KubeSchedulerConfiguration(in *v1alpha2.KubeSchedulerConfiguration, out *config.KubeSchedulerConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha2_KubeSchedulerConfiguration_To_config_KubeSchedulerConfiguration(in, out, s); err != nil {
		return err
	}
	out.AlgorithmSource.Provider = pointer.StringPtr(v1alpha2.SchedulerDefaultProviderName)
	return nil
}

func Convert_config_KubeSchedulerConfiguration_To_v1alpha2_KubeSchedulerConfiguration(in *config.KubeSchedulerConfiguration, out *v1alpha2.KubeSchedulerConfiguration, s conversion.Scope) error {
	return autoConvert_config_KubeSchedulerConfiguration_To_v1alpha2_KubeSchedulerConfiguration(in, out, s)
}
