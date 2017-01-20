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

package certs

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestCreatePKIAssets(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.Remove(tmpdir)

	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected bool
	}{
		{
			cfg:      &kubeadmapi.MasterConfiguration{},
			expected: false,
		},
		{
			// CIDR too small
			cfg: &kubeadmapi.MasterConfiguration{
				API:        kubeadmapi.API{AdvertiseAddresses: []string{"10.0.0.1"}},
				Networking: kubeadmapi.Networking{ServiceSubnet: "10.0.0.1/1"},
			},
			expected: false,
		},
		{
			// CIDR invalid
			cfg: &kubeadmapi.MasterConfiguration{
				API:        kubeadmapi.API{AdvertiseAddresses: []string{"10.0.0.1"}},
				Networking: kubeadmapi.Networking{ServiceSubnet: "invalid"},
			},
			expected: false,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:        kubeadmapi.API{AdvertiseAddresses: []string{"10.0.0.1"}},
				Networking: kubeadmapi.Networking{ServiceSubnet: "10.0.0.1/24"},
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		_, actual := CreatePKIAssets(rt.cfg, fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir))
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CreatePKIAssets with an error:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
