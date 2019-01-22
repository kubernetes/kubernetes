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
	"regexp"
	"strings"

	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

var imageRegEx = regexp.MustCompile(`(?P<repository>.+/)(?P<image>[^:]+)(?P<tag>:.+)`)

func Convert_v1alpha3_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s); err != nil {
		return err
	}
	return Convert_v1alpha3_APIEndpoint_To_kubeadm_APIEndpoint(&in.APIEndpoint, &out.LocalAPIEndpoint, s)
}

func Convert_kubeadm_InitConfiguration_To_v1alpha3_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_InitConfiguration_To_v1alpha3_InitConfiguration(in, out, s); err != nil {
		return err
	}
	return Convert_kubeadm_APIEndpoint_To_v1alpha3_APIEndpoint(&in.LocalAPIEndpoint, &out.APIEndpoint, s)
}

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

	if in.ControlPlane == true {
		out.ControlPlane = &kubeadm.JoinControlPlane{}
		if err := autoConvert_v1alpha3_APIEndpoint_To_kubeadm_APIEndpoint(&in.APIEndpoint, &out.ControlPlane.LocalAPIEndpoint, s); err != nil {
			return err
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

	if in.ControlPlane != nil {
		out.ControlPlane = true
		if err := autoConvert_kubeadm_APIEndpoint_To_v1alpha3_APIEndpoint(&in.ControlPlane.LocalAPIEndpoint, &out.APIEndpoint, s); err != nil {
			return err
		}
	}

	return nil
}

func Convert_v1alpha3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	if len(in.AuditPolicyConfiguration.Path) > 0 {
		return errors.New("AuditPolicyConfiguration has been removed from ClusterConfiguration. Please cleanup ClusterConfiguration.AuditPolicyConfiguration fields")
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

	if err := Convert_v1alpha3_UnifiedControlPlaneImage_To_kubeadm_UseHyperKubeImage(in, out); err != nil {
		return err
	}

	// converting v1alpha3 featureGate CoreDNS to internal DNS.Type
	if features.Enabled(in.FeatureGates, features.CoreDNS) {
		out.DNS.Type = kubeadm.CoreDNS
	} else {
		out.DNS.Type = kubeadm.KubeDNS
	}
	delete(out.FeatureGates, features.CoreDNS)

	return nil
}

func Convert_v1alpha3_UnifiedControlPlaneImage_To_kubeadm_UseHyperKubeImage(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration) error {
	if len(in.UnifiedControlPlaneImage) == 0 {
		out.UseHyperKubeImage = false
		return nil
	}

	k8sImageTag := kubeadmutil.KubernetesVersionToImageTag(in.KubernetesVersion)
	expectedImage := images.GetGenericImage(in.ImageRepository, constants.HyperKube, k8sImageTag)
	if expectedImage == in.UnifiedControlPlaneImage {
		out.UseHyperKubeImage = true
		return nil
	}

	return errors.Errorf("cannot convert unifiedControlPlaneImage=%q to useHyperKubeImage", in.UnifiedControlPlaneImage)
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

	if in.UseHyperKubeImage {
		out.UnifiedControlPlaneImage = images.GetKubernetesImage("", in)
	} else {
		out.UnifiedControlPlaneImage = ""
	}

	// converting internal DNS.Type to v1alpha3 featureGate CoreDNS (this is only for getting roundtrip passing, but it is never used in reality)
	if out.FeatureGates == nil {
		out.FeatureGates = map[string]bool{}
	}
	if in.DNS.Type == kubeadm.KubeDNS {
		out.FeatureGates[features.CoreDNS] = false
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

func Convert_v1alpha3_LocalEtcd_To_kubeadm_LocalEtcd(in *LocalEtcd, out *kubeadm.LocalEtcd, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_LocalEtcd_To_kubeadm_LocalEtcd(in, out, s); err != nil {
		return err
	}

	var err error
	out.ImageMeta, err = etcdImageToImageMeta(in.Image)
	return err
}

func etcdImageToImageMeta(image string) (kubeadm.ImageMeta, error) {
	// empty image -> empty image meta
	if image == "" {
		return kubeadm.ImageMeta{}, nil
	}

	matches := imageRegEx.FindStringSubmatch(image)
	if len(matches) != 4 {
		return kubeadm.ImageMeta{}, errors.New("Conversion Error: kubeadm does not support converting v1alpha3 configurations with etcd image without explicit repository or tag definition. Please fix the image name")
	}

	imageRepository := strings.TrimSuffix(matches[1], "/")
	imageName := matches[2]
	imageTag := strings.TrimPrefix(matches[3], ":")

	if imageName != constants.Etcd {
		return kubeadm.ImageMeta{}, errors.New("Conversion Error: kubeadm does not support converting v1alpha3 configurations with etcd imageName different than etcd. Please fix the image name")
	}

	return kubeadm.ImageMeta{
		ImageRepository: imageRepository,
		ImageTag:        imageTag,
	}, nil
}

func Convert_kubeadm_LocalEtcd_To_v1alpha3_LocalEtcd(in *kubeadm.LocalEtcd, out *LocalEtcd, s conversion.Scope) error {
	if err := autoConvert_kubeadm_LocalEtcd_To_v1alpha3_LocalEtcd(in, out, s); err != nil {
		return err
	}

	// converting internal LocalEtcd.ImageMeta to v1alpha3 LocalEtcd.Image (this is only for getting roundtrip passing, but it is
	// never used in reality)

	return nil
}
