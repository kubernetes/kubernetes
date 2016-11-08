/*
Copyright 2016 The Kubernetes Authors.

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

package master

import (
	"crypto/x509"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestNewKubeDiscovery(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		caCert   *x509.Certificate
		expected bool
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:        kubeadmapi.API{Port: 123, AdvertiseAddresses: []string{"10.0.0.1"}},
				Networking: kubeadmapi.Networking{ServiceSubnet: "10.0.0.1/1"},
				Discovery:  kubeadmapi.Discovery{Token: &kubeadmapi.TokenDiscovery{}},
			},
			caCert: &x509.Certificate{},
		},
	}
	for _, rt := range tests {
		actual := newKubeDiscovery(rt.cfg, rt.caCert)
		if actual.Deployment == nil || actual.Secret == nil {
			t.Errorf(
				"failed newKubeDiscovery, kubeDiscovery was nil",
			)
		}
	}
}
