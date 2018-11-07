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

package v1alpha3

import (
	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func Convert_v1alpha3_JoinConfiguration_To_kubeadm_JoinConfiguration(in *JoinConfiguration, out *kubeadm.JoinConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_JoinConfiguration_To_kubeadm_JoinConfiguration(in, out, s); err != nil {
		return err
	}

	if len(in.ClusterName) != 0 {
		return errors.New("clusterName has been removed from JoinConfiguration and clusterName from ClusterConfiguration will be used instead. Please cleanup JoinConfiguration.ClusterName fields")
	}

	if len(in.FeatureGates) != 0 {
		return errors.New("featureGates has been removed from JoinConfiguration and featureGates from ClusterConfiguration will be used instead. Please cleanup JoinConfiguration.FeatureGates fields")
	}

	out.Discovery.Timeout = in.DiscoveryTimeout

	if len(in.TLSBootstrapToken) != 0 {
		out.Discovery.TLSBootstrapToken = in.TLSBootstrapToken
	} else {
		out.Discovery.TLSBootstrapToken = in.Token
	}

	if len(in.DiscoveryFile) != 0 {
		out.Discovery.File = &kubeadm.FileDiscovery{
			KubeConfigPath: in.DiscoveryFile,
		}
	} else {
		out.Discovery.BootstrapToken = &kubeadm.BootstrapTokenDiscovery{
			CACertHashes:             in.DiscoveryTokenCACertHashes,
			UnsafeSkipCAVerification: in.DiscoveryTokenUnsafeSkipCAVerification,
		}
		if len(in.DiscoveryTokenAPIServers) != 0 {
			out.Discovery.BootstrapToken.APIServerEndpoint = in.DiscoveryTokenAPIServers[0]
		}
		if len(in.DiscoveryToken) != 0 {
			out.Discovery.BootstrapToken.Token = in.DiscoveryToken
		} else {
			out.Discovery.BootstrapToken.Token = in.Token
		}
	}

	return nil
}

func Convert_kubeadm_JoinConfiguration_To_v1alpha3_JoinConfiguration(in *kubeadm.JoinConfiguration, out *JoinConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_JoinConfiguration_To_v1alpha3_JoinConfiguration(in, out, s); err != nil {
		return err
	}

	out.DiscoveryTimeout = in.Discovery.Timeout
	out.TLSBootstrapToken = in.Discovery.TLSBootstrapToken

	if in.Discovery.BootstrapToken != nil {
		out.DiscoveryToken = in.Discovery.BootstrapToken.Token
		out.DiscoveryTokenAPIServers = []string{in.Discovery.BootstrapToken.APIServerEndpoint}
		out.DiscoveryTokenCACertHashes = in.Discovery.BootstrapToken.CACertHashes
		out.DiscoveryTokenUnsafeSkipCAVerification = in.Discovery.BootstrapToken.UnsafeSkipCAVerification

	} else if in.Discovery.File != nil {
		out.DiscoveryFile = in.Discovery.File.KubeConfigPath
	}

	return nil
}

func Convert_v1alpha3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	out.APIServer.ExtraArgs = in.APIServerExtraArgs
	out.APIServer.CertSANs = in.APIServerCertSANs
	out.APIServer.TimeoutForControlPlane = &metav1.Duration{
		Duration: constants.DefaultControlPlaneTimeout,
	}
	if err := convertSlice_v1alpha3_HostPathMount_To_kubeadm_HostPathMount(&in.APIServerExtraVolumes, &out.APIServer.ExtraVolumes, s); err != nil {
		return err
	}

	out.ControllerManager.ExtraArgs = in.ControllerManagerExtraArgs
	if err := convertSlice_v1alpha3_HostPathMount_To_kubeadm_HostPathMount(&in.ControllerManagerExtraVolumes, &out.ControllerManager.ExtraVolumes, s); err != nil {
		return err
	}

	out.Scheduler.ExtraArgs = in.SchedulerExtraArgs
	if err := convertSlice_v1alpha3_HostPathMount_To_kubeadm_HostPathMount(&in.SchedulerExtraVolumes, &out.Scheduler.ExtraVolumes, s); err != nil {
		return err
	}

	return nil
}

func Convert_kubeadm_ClusterConfiguration_To_v1alpha3_ClusterConfiguration(in *kubeadm.ClusterConfiguration, out *ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_ClusterConfiguration_To_v1alpha3_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	out.APIServerExtraArgs = in.APIServer.ExtraArgs
	out.APIServerCertSANs = in.APIServer.CertSANs
	if err := convertSlice_kubeadm_HostPathMount_To_v1alpha3_HostPathMount(&in.APIServer.ExtraVolumes, &out.APIServerExtraVolumes, s); err != nil {
		return err
	}

	out.ControllerManagerExtraArgs = in.ControllerManager.ExtraArgs
	if err := convertSlice_kubeadm_HostPathMount_To_v1alpha3_HostPathMount(&in.ControllerManager.ExtraVolumes, &out.ControllerManagerExtraVolumes, s); err != nil {
		return err
	}

	out.SchedulerExtraArgs = in.Scheduler.ExtraArgs
	if err := convertSlice_kubeadm_HostPathMount_To_v1alpha3_HostPathMount(&in.Scheduler.ExtraVolumes, &out.SchedulerExtraVolumes, s); err != nil {
		return err
	}

	return nil
}

func Convert_v1alpha3_HostPathMount_To_kubeadm_HostPathMount(in *HostPathMount, out *kubeadm.HostPathMount, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_HostPathMount_To_kubeadm_HostPathMount(in, out, s); err != nil {
		return err
	}

	out.ReadOnly = !in.Writable
	return nil
}

func Convert_kubeadm_HostPathMount_To_v1alpha3_HostPathMount(in *kubeadm.HostPathMount, out *HostPathMount, s conversion.Scope) error {
	if err := autoConvert_kubeadm_HostPathMount_To_v1alpha3_HostPathMount(in, out, s); err != nil {
		return err
	}

	out.Writable = !in.ReadOnly
	return nil
}

func convertSlice_v1alpha3_HostPathMount_To_kubeadm_HostPathMount(in *[]HostPathMount, out *[]kubeadm.HostPathMount, s conversion.Scope) error {
	if *in != nil {
		*out = make([]kubeadm.HostPathMount, len(*in))
		for i := range *in {
			if err := Convert_v1alpha3_HostPathMount_To_kubeadm_HostPathMount(&(*in)[i], &(*out)[i], s); err != nil {
				return err
			}
		}
	} else {
		*out = nil
	}
	return nil
}

func convertSlice_kubeadm_HostPathMount_To_v1alpha3_HostPathMount(in *[]kubeadm.HostPathMount, out *[]HostPathMount, s conversion.Scope) error {
	if *in != nil {
		*out = make([]HostPathMount, len(*in))
		for i := range *in {
			if err := Convert_kubeadm_HostPathMount_To_v1alpha3_HostPathMount(&(*in)[i], &(*out)[i], s); err != nil {
				return err
			}
		}
	} else {
		*out = nil
	}
	return nil
}
