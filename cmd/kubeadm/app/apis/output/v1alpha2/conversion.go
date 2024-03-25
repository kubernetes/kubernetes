/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/conversion"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/output"
)

// Convert_output_UpgradePlan_To_v1alpha2_UpgradePlan converts a private UpgradePlan to public UpgradePlan.
func Convert_output_UpgradePlan_To_v1alpha2_UpgradePlan(in *output.UpgradePlan, out *UpgradePlan, s conversion.Scope) error {
	return autoConvert_output_UpgradePlan_To_v1alpha2_UpgradePlan(in, out, s)
}

// Convert_output_ComponentUpgradePlan_To_v1alpha2_ComponentUpgradePlan converts a private ComponentUpgradePlan to public ComponentUpgradePlan.
func Convert_output_ComponentUpgradePlan_To_v1alpha2_ComponentUpgradePlan(in *output.ComponentUpgradePlan, out *ComponentUpgradePlan, s conversion.Scope) error {
	return autoConvert_output_ComponentUpgradePlan_To_v1alpha2_ComponentUpgradePlan(in, out, s)
}
