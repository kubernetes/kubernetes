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

package v1alpha1

import (
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
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

	out.APIServerExtraArgs = in.APIServer.ExtraArgs
	out.AuthorizationModes = in.APIServer.AuthorizationModes
	out.APIServerCertSANs = in.APIServer.ServingCertExtraSANs
	out.ControllerManagerExtraArgs = in.ControllerManager.ExtraArgs
	out.SchedulerExtraArgs = in.Scheduler.ExtraArgs
	out.CertificatesDir = in.Paths.CertificatesDir

	return nil
}

func Convert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration(in *MasterConfiguration, out *kubeadm.MasterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration(in, out, s); err != nil {
		return err
	}

	out.APIServer.ExtraArgs = in.APIServerExtraArgs
	out.APIServer.AuthorizationModes = in.AuthorizationModes
	out.APIServer.ServingCertExtraSANs = in.APIServerCertSANs
	out.ControllerManager.ExtraArgs = in.ControllerManagerExtraArgs
	out.Scheduler.ExtraArgs = in.SchedulerExtraArgs
	out.Paths.CertificatesDir = in.CertificatesDir

	return nil
}
