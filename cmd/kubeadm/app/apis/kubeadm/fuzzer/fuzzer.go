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
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

// Funcs returns the fuzzer functions for the kubeadm apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *kubeadm.MasterConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			obj.KubernetesVersion = "v10"
			obj.API.BindPort = 20
			obj.API.AdvertiseAddress = "foo"
			obj.Networking.ServiceSubnet = "foo"
			obj.Networking.DNSDomain = "foo"
			obj.CertificatesDir = "foo"
			obj.APIServerCertSANs = []string{"foo"}

			obj.BootstrapTokens = []kubeadm.BootstrapToken{
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
			// Note: We don't set values here for obj.Etcd.External, as these are mutually exlusive.
			// And to make sure the fuzzer doesn't set a random value for obj.Etcd.External, we let
			// kubeadmapi.Etcd implement fuzz.Interface (we handle that ourselves)
			obj.Etcd.Local = &kubeadm.LocalEtcd{
				Image:          "foo",
				DataDir:        "foo",
				ServerCertSANs: []string{"foo"},
				PeerCertSANs:   []string{"foo"},
				ExtraArgs:      map[string]string{"foo": "foo"},
			}
			obj.NodeRegistration = kubeadm.NodeRegistrationOptions{
				CRISocket: "foo",
				Name:      "foo",
				Taints:    []v1.Taint{},
			}
			obj.KubeletConfiguration = kubeadm.KubeletConfiguration{
				BaseConfig: &kubeletconfigv1beta1.KubeletConfiguration{
					StaticPodPath: "foo",
					ClusterDNS:    []string{"foo"},
					ClusterDomain: "foo",
					Authorization: kubeletconfigv1beta1.KubeletAuthorization{
						Mode: "Webhook",
					},
					Authentication: kubeletconfigv1beta1.KubeletAuthentication{
						X509: kubeletconfigv1beta1.KubeletX509Authentication{
							ClientCAFile: "/etc/kubernetes/pki/ca.crt",
						},
						Anonymous: kubeletconfigv1beta1.KubeletAnonymousAuthentication{
							Enabled: utilpointer.BoolPtr(false),
						},
					},
					RotateCertificates: true,
				},
			}
			kubeletconfigv1beta1.SetDefaults_KubeletConfiguration(obj.KubeletConfiguration.BaseConfig)
			obj.KubeProxy = kubeadm.KubeProxy{
				Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
					FeatureGates:       map[string]bool{"foo": true},
					BindAddress:        "foo",
					HealthzBindAddress: "foo:10256",
					MetricsBindAddress: "foo:",
					EnableProfiling:    bool(true),
					ClusterCIDR:        "foo",
					HostnameOverride:   "foo",
					ClientConnection: kubeproxyconfigv1alpha1.ClientConnectionConfiguration{
						KubeConfigFile:     "foo",
						AcceptContentTypes: "foo",
						ContentType:        "foo",
						QPS:                float32(5),
						Burst:              10,
					},
					IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
						SyncPeriod: metav1.Duration{Duration: 1},
					},
					IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
						MasqueradeBit: utilpointer.Int32Ptr(0),
						SyncPeriod:    metav1.Duration{Duration: 1},
					},
					OOMScoreAdj:       utilpointer.Int32Ptr(0),
					ResourceContainer: "foo",
					UDPIdleTimeout:    metav1.Duration{Duration: 1},
					Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
						MaxPerCore: utilpointer.Int32Ptr(2),
						Min:        utilpointer.Int32Ptr(1),
						TCPEstablishedTimeout: &metav1.Duration{Duration: 5},
						TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5},
					},
					ConfigSyncPeriod: metav1.Duration{Duration: 1},
				},
			}
			obj.AuditPolicyConfiguration = kubeadm.AuditPolicyConfiguration{
				Path:      "foo",
				LogDir:    "/foo",
				LogMaxAge: utilpointer.Int32Ptr(0),
			}
		},
		func(obj *kubeadm.NodeConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			obj.CACertPath = "foo"
			obj.DiscoveryFile = "foo"
			obj.DiscoveryToken = "foo"
			obj.DiscoveryTokenAPIServers = []string{"foo"}
			obj.DiscoveryTimeout = &metav1.Duration{Duration: 1}
			obj.TLSBootstrapToken = "foo"
			obj.Token = "foo"
			obj.ClusterName = "foo"
			obj.NodeRegistration = kubeadm.NodeRegistrationOptions{
				CRISocket: "foo",
				Name:      "foo",
			}
		},
	}
}
