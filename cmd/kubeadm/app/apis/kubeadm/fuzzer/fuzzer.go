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
	"github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
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
			obj.AuthorizationModes = []string{"foo"}
			obj.CertificatesDir = "foo"
			obj.APIServerCertSANs = []string{}
			obj.Token = "foo"
			obj.Etcd.Image = "foo"
			obj.Etcd.DataDir = "foo"
			obj.ImageRepository = "foo"
			obj.UnifiedControlPlaneImage = "foo"
			obj.FeatureFlags = map[string]bool{}
		},
		func(obj *kubeadm.NodeConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			obj.CACertPath = "foo"
			obj.CACertPath = "foo"
			obj.DiscoveryFile = "foo"
			obj.DiscoveryToken = "foo"
			obj.DiscoveryTokenAPIServers = []string{"foo"}
			obj.TLSBootstrapToken = "foo"
			obj.Token = "foo"
		},
	}
}
