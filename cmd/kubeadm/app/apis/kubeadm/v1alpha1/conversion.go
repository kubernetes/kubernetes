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
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_kubeadm_MasterConfiguration_To_v1alpha1_MasterConfiguration,
		Convert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration,
	)
	if err != nil {
		return err
	}

	return nil
}

func Convert_kubeadm_MasterConfiguration_To_v1alpha1_MasterConfiguration(in *kubeadm.MasterConfiguration, out *MasterConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_MasterConfiguration_To_v1alpha1_MasterConfiguration(in, out, s); err != nil {
		return err
	}

	// Setting .CloudProvider is not supported from internal API not supported
	return nil
}

func Convert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration(in *MasterConfiguration, out *kubeadm.MasterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration(in, out, s); err != nil {
		return err
	}

	UpgradeCloudProvider(in, out)

	return nil
}

// UpgradeCloudProvider handles the removal of .CloudProvider as smoothly as possible
func UpgradeCloudProvider(in *MasterConfiguration, out *kubeadm.MasterConfiguration) {
	if len(in.CloudProvider) != 0 {
		if out.APIServerExtraArgs == nil {
			out.APIServerExtraArgs = map[string]string{}
		}
		if out.ControllerManagerExtraArgs == nil {
			out.ControllerManagerExtraArgs = map[string]string{}
		}

		out.APIServerExtraArgs["cloud-provider"] = in.CloudProvider
		out.ControllerManagerExtraArgs["cloud-provider"] = in.CloudProvider
	}
}
