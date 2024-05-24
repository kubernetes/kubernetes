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
	fuzz "github.com/google/gofuzz"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/utils/ptr"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// Funcs returns the fuzzer functions for the kubeadm apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		fuzzInitConfiguration,
		fuzzClusterConfiguration,
		fuzzComponentConfigMap,
		fuzzDNS,
		fuzzNodeRegistration,
		fuzzLocalEtcd,
		fuzzNetworking,
		fuzzJoinConfiguration,
		fuzzJoinControlPlane,
		fuzzResetConfiguration,
		fuzzUpgradeConfiguration,
	}
}

func fuzzInitConfiguration(obj *kubeadm.InitConfiguration, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)

	// Avoid round tripping the ClusterConfiguration embedded in the InitConfiguration, since it is
	// only present in the internal version and not in public versions
	obj.ClusterConfiguration = kubeadm.ClusterConfiguration{}

	// Adds the default bootstrap token to get the round trip working
	obj.BootstrapTokens = []bootstraptokenv1.BootstrapToken{
		{
			Groups: []string{"foo"},
			Usages: []string{"foo"},
			TTL:    &metav1.Duration{Duration: 1234},
		},
	}
	obj.SkipPhases = nil
	obj.NodeRegistration.ImagePullPolicy = corev1.PullIfNotPresent
	obj.NodeRegistration.ImagePullSerial = ptr.To(true)
	obj.Patches = nil
	obj.DryRun = false
	kubeadm.SetDefaultTimeouts(&obj.Timeouts)
}

func fuzzNodeRegistration(obj *kubeadm.NodeRegistrationOptions, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)
	obj.IgnorePreflightErrors = nil
	obj.ImagePullSerial = ptr.To(true)
}

func fuzzClusterConfiguration(obj *kubeadm.ClusterConfiguration, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)
	obj.CertificatesDir = "foo"
	obj.ClusterName = "bar"
	obj.ImageRepository = "baz"
	obj.CIImageRepository = "" // This fields doesn't exists in public API >> using default to get the roundtrip test pass
	obj.KubernetesVersion = "qux"
	obj.CIKubernetesVersion = "" // This fields doesn't exists in public API >> using default to get the roundtrip test pass
	obj.APIServer.TimeoutForControlPlane = &metav1.Duration{
		Duration: 0,
	}
	obj.ControllerManager.ExtraEnvs = []kubeadm.EnvVar{}
	obj.APIServer.ExtraEnvs = []kubeadm.EnvVar{}
	obj.Scheduler.ExtraEnvs = []kubeadm.EnvVar{}
	obj.Etcd.Local.ExtraEnvs = []kubeadm.EnvVar{}
	obj.EncryptionAlgorithm = kubeadm.EncryptionAlgorithmRSA2048
	obj.Proxy.Disabled = false
}

func fuzzDNS(obj *kubeadm.DNS, c fuzz.Continue) {
	c.FuzzNoCustom(obj)
	obj.Disabled = false
}

func fuzzComponentConfigMap(obj *kubeadm.ComponentConfigMap, c fuzz.Continue) {
	// This is intentionally empty because component config does not exists in the public api
	// (empty mean all ComponentConfigs fields nil, and this is necessary for getting roundtrip passing)
}

func fuzzLocalEtcd(obj *kubeadm.LocalEtcd, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)
	obj.DataDir = "foo"
}

func fuzzNetworking(obj *kubeadm.Networking, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)
	obj.DNSDomain = "foo"
	obj.ServiceSubnet = "bar"
}

func fuzzJoinConfiguration(obj *kubeadm.JoinConfiguration, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)
	obj.CACertPath = "foo"
	obj.Discovery = kubeadm.Discovery{
		BootstrapToken:    &kubeadm.BootstrapTokenDiscovery{Token: "baz"},
		TLSBootstrapToken: "qux",
		Timeout:           &metav1.Duration{},
	}
	obj.SkipPhases = nil
	obj.NodeRegistration.ImagePullPolicy = corev1.PullIfNotPresent
	obj.NodeRegistration.ImagePullSerial = ptr.To(true)
	obj.Patches = nil
	obj.DryRun = false
	kubeadm.SetDefaultTimeouts(&obj.Timeouts)
}

func fuzzJoinControlPlane(obj *kubeadm.JoinControlPlane, c fuzz.Continue) {
	c.FuzzNoCustom(obj)
}

func fuzzResetConfiguration(obj *kubeadm.ResetConfiguration, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)
	obj.CertificatesDir = "/tmp"
	kubeadm.SetDefaultTimeouts(&obj.Timeouts)
}

func fuzzUpgradeConfiguration(obj *kubeadm.UpgradeConfiguration, c fuzz.Continue) {
	c.FuzzNoCustom(obj)

	// Pinning values for fields that get defaults if fuzz value is empty string or nil (thus making the round trip test fail)
	obj.Node.EtcdUpgrade = ptr.To(true)
	obj.Apply.EtcdUpgrade = ptr.To(true)
	obj.Apply.CertificateRenewal = ptr.To(false)
	obj.Node.CertificateRenewal = ptr.To(false)
	kubeadm.SetDefaultTimeouts(&obj.Timeouts)
}
