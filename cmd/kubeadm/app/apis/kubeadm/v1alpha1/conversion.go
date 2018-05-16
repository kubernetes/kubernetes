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
		Convert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration,
		Convert_kubeadm_MasterConfiguration_To_v1alpha1_MasterConfiguration,
		Convert_v1alpha1_NodeConfiguration_To_kubeadm_NodeConfiguration,
		Convert_kubeadm_NodeConfiguration_To_v1alpha1_NodeConfiguration,
	)
	if err != nil {
		return err
	}

	return nil
}

// no-ops, as we don't support rollbacks
func Convert_kubeadm_MasterConfiguration_To_v1alpha1_MasterConfiguration(in *MasterConfiguration, out *kubeadm.MasterConfiguration, s conversion.Scope) error {
	return nil
}
func Convert_kubeadm_NodeConfiguration_To_v1alpha1_NodeConfiguration(in *NodeConfiguration, out *kubeadm.NodeConfiguration, s conversion.Scope) error {
	return nil
}

func Convert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration(in *MasterConfiguration, out *kubeadm.MasterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_MasterConfiguration_To_kubeadm_MasterConfiguration(in, out, s); err != nil {
		return err
	}

	UpgradeMasterClusterName(in, out)

	return nil
}

func Convert_v1alpha1_NodeConfiguration_To_kubeadm_NodeConfiguration(in *NodeConfiguration, out *kubeadm.NodeConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_NodeConfiguration_To_kubeadm_NodeConfiguration(in, out, s); err != nil {
		return err
	}

	UpgradeNodeClusterName(in, out)

	return nil
}

// UpgradeMasterClusterName handles the rename of .ClusterName between v1alpha1 and v1alpha2
func UpgradeMasterClusterName(in *MasterConfiguration, out *kubeadm.MasterConfiguration) {
	if len(in.ClusterName) != 0 {
		out.Metadata.ClusterName = in.ClusterName
	}
}

// UpgradeNodeClusterName handles the rename of .ClusterName between v1alpha1 and v1alpha2
func UpgradeNodeClusterName(in *NodeConfiguration, out *kubeadm.NodeConfiguration) {
	if len(in.ClusterName) != 0 {
		out.Metadata.ClusterName = in.ClusterName
	}
}
