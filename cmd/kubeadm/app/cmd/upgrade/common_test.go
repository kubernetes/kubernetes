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

package upgrade

import (
	"bytes"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestPrintConfiguration(t *testing.T) {
	var tests = []struct {
		cfg           *kubeadmapi.ClusterConfiguration
		buf           *bytes.Buffer
		expectedBytes []byte
	}{
		{
			cfg:           nil,
			expectedBytes: []byte(""),
		},
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.7.1",
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{
						DataDir: "/some/path",
					},
				},
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.CoreDNS,
				},
			},
			expectedBytes: []byte(`[upgrade/config] Configuration used:
	apiServer: {}
	apiVersion: kubeadm.k8s.io/v1beta1
	certificatesDir: ""
	controlPlaneEndpoint: ""
	controllerManager: {}
	dns:
	  type: CoreDNS
	etcd:
	  local:
	    dataDir: /some/path
	imageRepository: ""
	kind: ClusterConfiguration
	kubernetesVersion: v1.7.1
	networking:
	  dnsDomain: ""
	  podSubnet: ""
	  serviceSubnet: ""
	scheduler: {}
`),
		},
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.7.1",
				Networking: kubeadmapi.Networking{
					ServiceSubnet: "10.96.0.1/12",
				},
				Etcd: kubeadmapi.Etcd{
					External: &kubeadmapi.ExternalEtcd{
						Endpoints: []string{"https://one-etcd-instance:2379"},
					},
				},
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.CoreDNS,
				},
			},
			expectedBytes: []byte(`[upgrade/config] Configuration used:
	apiServer: {}
	apiVersion: kubeadm.k8s.io/v1beta1
	certificatesDir: ""
	controlPlaneEndpoint: ""
	controllerManager: {}
	dns:
	  type: CoreDNS
	etcd:
	  external:
	    caFile: ""
	    certFile: ""
	    endpoints:
	    - https://one-etcd-instance:2379
	    keyFile: ""
	imageRepository: ""
	kind: ClusterConfiguration
	kubernetesVersion: v1.7.1
	networking:
	  dnsDomain: ""
	  podSubnet: ""
	  serviceSubnet: 10.96.0.1/12
	scheduler: {}
`),
		},
	}
	for _, rt := range tests {
		rt.buf = bytes.NewBufferString("")
		printConfiguration(rt.cfg, rt.buf)
		actualBytes := rt.buf.Bytes()
		if !bytes.Equal(actualBytes, rt.expectedBytes) {
			t.Errorf(
				"failed PrintConfiguration:\n\texpected: %q\n\t  actual: %q",
				string(rt.expectedBytes),
				string(actualBytes),
			)
		}
	}
}
