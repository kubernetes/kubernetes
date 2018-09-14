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

package fuzzer

import (
	"time"

	fuzz "github.com/google/gofuzz"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	utilpointer "k8s.io/utils/pointer"
)

// NOTE: Right now this code is unused, as the test utilizing this is disabled.

// Funcs returns the fuzzer functions for the kubeadm apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *kubeadm.ClusterConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			fuzzClusterConfig(obj)
		},
		func(obj *kubeadm.InitConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			fuzzClusterConfig(&obj.ClusterConfiguration)
			fuzzBootstrapTokens(&obj.BootstrapTokens)
			fuzzNodeRegistration(&obj.NodeRegistration)
			fuzzAPIEndpoint(&obj.APIEndpoint)
		},
		func(obj *kubeadm.JoinConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			fuzzNodeRegistration(&obj.NodeRegistration)
			fuzzAPIEndpoint(&obj.APIEndpoint)
			obj.CACertPath = "foo"
			obj.DiscoveryFile = "foo"
			obj.DiscoveryToken = "foo"
			obj.DiscoveryTokenAPIServers = []string{"foo"}
			obj.DiscoveryTimeout = &metav1.Duration{Duration: 1}
			obj.TLSBootstrapToken = "foo"
			obj.Token = "foo"
			obj.ClusterName = "foo"
		},
	}
}

func fuzzBootstrapTokens(obj *[]kubeadm.BootstrapToken) {
	obj = &[]kubeadm.BootstrapToken{
		{
			Token: &kubeadm.BootstrapTokenString{
				ID:     "abcdef",
				Secret: "abcdef0123456789",
			},
			TTL:    &metav1.Duration{Duration: 1 * time.Hour},
			Usages: []string{"foo"},
			Groups: []string{"foo"},
		},
	}
}

func fuzzNodeRegistration(obj *kubeadm.NodeRegistrationOptions) {
	obj.CRISocket = "foo"
	obj.Name = "foo"
	obj.Taints = []v1.Taint{}
}

func fuzzAPIEndpoint(obj *kubeadm.APIEndpoint) {
	obj.BindPort = 20
	obj.AdvertiseAddress = "foo"
}

func fuzzClusterConfig(obj *kubeadm.ClusterConfiguration) {
	obj.KubernetesVersion = "v10"
	obj.Networking.ServiceSubnet = "10.96.0.0/12"
	obj.Networking.DNSDomain = "cluster.local"
	obj.CertificatesDir = "foo"
	obj.APIServerCertSANs = []string{"foo"}
	obj.ImageRepository = "foo"
	obj.CIImageRepository = ""
	obj.UnifiedControlPlaneImage = "foo"
	obj.FeatureGates = map[string]bool{"foo": true}
	obj.ClusterName = "foo"
	obj.APIServerExtraArgs = map[string]string{"foo": "foo"}
	obj.APIServerExtraVolumes = []kubeadm.HostPathMount{{
		Name:      "foo",
		HostPath:  "foo",
		MountPath: "foo",
		Writable:  false,
	}}
	obj.Etcd.Local = &kubeadm.LocalEtcd{
		Image:          "foo",
		DataDir:        "foo",
		ServerCertSANs: []string{"foo"},
		PeerCertSANs:   []string{"foo"},
		ExtraArgs:      map[string]string{"foo": "foo"},
	}
	obj.AuditPolicyConfiguration = kubeadm.AuditPolicyConfiguration{
		Path:      "foo",
		LogDir:    "/foo",
		LogMaxAge: utilpointer.Int32Ptr(0),
	}
	// Set the Kubelet ComponentConfig to an empty, defaulted struct
	extkubeletconfig := &kubeletconfigv1beta1.KubeletConfiguration{}
	obj.ComponentConfigs.Kubelet = &kubeletconfig.KubeletConfiguration{}
	componentconfigs.Scheme.Default(extkubeletconfig)
	componentconfigs.Scheme.Convert(extkubeletconfig, obj.ComponentConfigs.Kubelet, nil)
	componentconfigs.DefaultKubeletConfiguration(obj)
	// Set the KubeProxy ComponentConfig to an empty, defaulted struct
	extkubeproxyconfig := &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
	obj.ComponentConfigs.KubeProxy = &kubeproxyconfig.KubeProxyConfiguration{}
	componentconfigs.Scheme.Default(extkubeproxyconfig)
	componentconfigs.Scheme.Convert(extkubeproxyconfig, obj.ComponentConfigs.KubeProxy, nil)
	componentconfigs.DefaultKubeProxyConfiguration(obj)
}
