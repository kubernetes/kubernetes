/*
Copyright 2021 The Kubernetes Authors.

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
	conversion "k8s.io/apimachinery/pkg/conversion"

	kubeadmv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	output "k8s.io/kubernetes/cmd/kubeadm/app/apis/output"
)

// The conversion functions here are required since output.v1alpha2.BootstrapToken embeds
// bootstraptoken.v1.BootstrapToken, instead of kubeadm.v1beta2.BootstrapToken.

func Convert_v1alpha1_BootstrapToken_To_output_BootstrapToken(in *BootstrapToken, out *output.BootstrapToken, s conversion.Scope) error {
	kubeadmv1beta2.Convert_v1beta2_BootstrapToken_To_v1_BootstrapToken(&in.BootstrapToken, &out.BootstrapToken, s)
	return nil
}

func Convert_output_BootstrapToken_To_v1alpha1_BootstrapToken(in *output.BootstrapToken, out *BootstrapToken, s conversion.Scope) error {
	kubeadmv1beta2.Convert_v1_BootstrapToken_To_v1beta2_BootstrapToken(&in.BootstrapToken, &out.BootstrapToken, s)
	return nil
}
