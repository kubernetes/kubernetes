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
	"unsafe"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func Convert_v1alpha3_JoinConfiguration_To_kubeadm_JoinConfiguration(in *JoinConfiguration, out *kubeadm.JoinConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_JoinConfiguration_To_kubeadm_JoinConfiguration(in, out, s); err != nil {
		return err
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
	out.APIServer.ExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.APIServerExtraVolumes))
	out.APIServer.CertSANs = in.APIServerCertSANs

	out.ControllerManager.ExtraArgs = in.ControllerManagerExtraArgs
	out.ControllerManager.ExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.ControllerManagerExtraVolumes))

	out.Scheduler.ExtraArgs = in.SchedulerExtraArgs
	out.Scheduler.ExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.SchedulerExtraVolumes))

	return nil
}

func Convert_kubeadm_ClusterConfiguration_To_v1alpha3_ClusterConfiguration(in *kubeadm.ClusterConfiguration, out *ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_ClusterConfiguration_To_v1alpha3_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	out.APIServerExtraArgs = in.APIServer.ExtraArgs
	out.APIServerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.APIServer.ExtraVolumes))
	out.APIServerCertSANs = in.APIServer.CertSANs

	out.ControllerManagerExtraArgs = in.ControllerManager.ExtraArgs
	out.ControllerManagerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.ControllerManager.ExtraVolumes))

	out.SchedulerExtraArgs = in.Scheduler.ExtraArgs
	out.SchedulerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.Scheduler.ExtraVolumes))

	return nil
}
