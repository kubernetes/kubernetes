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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestCreateCertsAndConfigForClients(t *testing.T) {
	var tests = []struct {
		a         kubeadmapi.API
		cn        []string
		caKeySize int
		expected  bool
	}{
		{
			a:         kubeadmapi.API{AdvertiseAddresses: []string{"foo"}},
			cn:        []string{"localhost"},
			caKeySize: 128,
			expected:  false,
		},
		{
			a:         kubeadmapi.API{AdvertiseAddresses: []string{"foo"}},
			cn:        []string{},
			caKeySize: 128,
			expected:  true,
		},
		{
			a:         kubeadmapi.API{AdvertiseAddresses: []string{"foo"}},
			cn:        []string{"localhost"},
			caKeySize: 2048,
			expected:  true,
		},
	}

	for _, rt := range tests {
		caKey, err := rsa.GenerateKey(rand.Reader, rt.caKeySize)
		if err != nil {
			t.Fatalf("Couldn't create rsa Private Key")
		}
		caCert := &x509.Certificate{}
		_, actual := CreateCertsAndConfigForClients(rt.a, rt.cn, caKey, caCert)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CreateCertsAndConfigForClients:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
