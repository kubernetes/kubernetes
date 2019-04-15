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

package file

import (
	"testing"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestCreateKubeconfig(t *testing.T) {
	var tests = []struct {
		description   string
		cfg           *clientcmdapi.Config
		clustername   string
		expectedError bool
	}{
		{
			description: "Create basic kubeconfig without authInfos",
			cfg: &clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"minikube": {Server: "https://192.168.99.100:8443"},
				},
				Contexts: map[string]*clientcmdapi.Context{
					"minikube": {AuthInfo: "minikube", Cluster: "minikube"},
				},
				CurrentContext: "minikube",
			},
			clustername:   "minikube",
			expectedError: false,
		},
		{
			description: "cfg.AuthInfos hasn't the cfg.Contexts[config.CurrentContext].AuthInfo should return error",
			cfg: &clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"minikube": {Server: "https://192.168.99.100:8443"},
				},
				Contexts: map[string]*clientcmdapi.Context{
					"minikube": {AuthInfo: "minikube", Cluster: "minikube"},
				},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikubes": {Token: "minikube-token"},
				},
				CurrentContext: "minikube",
			},
			clustername:   "minikube",
			expectedError: true,
		},
		{
			description: "no authInfo.ClientCertificateData should return error",
			cfg: &clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"minikube": {Server: "https://192.168.99.100:8443"},
				},
				Contexts: map[string]*clientcmdapi.Context{
					"minikube": {AuthInfo: "minikube", Cluster: "minikube"},
				},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {
						Token:                 "minikube-token",
						ClientCertificateData: []byte(""),
						ClientKeyData:         []byte("client key data"),
					},
				},
				CurrentContext: "minikube",
			},
			clustername:   "minikube",
			expectedError: true,
		},
		{
			description: "no authInfo.ClientKeyData should return error",
			cfg: &clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"minikube": {Server: "https://192.168.99.100:8443"},
				},
				Contexts: map[string]*clientcmdapi.Context{
					"minikube": {AuthInfo: "minikube", Cluster: "minikube"},
				},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {
						Token:                 "minikube-token",
						ClientCertificateData: []byte("client certificateData data"),
						ClientKeyData:         []byte(""),
					},
				},
				CurrentContext: "minikube",
			},
			clustername:   "minikube",
			expectedError: true,
		},
		{
			description: "read ClientCertificate failed should return error",
			cfg: &clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"minikube": {Server: "https://192.168.99.100:8443"},
				},
				Contexts: map[string]*clientcmdapi.Context{
					"minikube": {AuthInfo: "minikube", Cluster: "minikube"},
				},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {
						Token:             "minikube-token",
						ClientCertificate: "tmp/client-cert-filename",
					},
				},
				CurrentContext: "minikube",
			},
			clustername:   "minikube",
			expectedError: true,
		},
		{
			description: "Create kubeconfig with authInfos",
			cfg: &clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"minikube": {Server: "https://192.168.99.100:8443"},
				},
				Contexts: map[string]*clientcmdapi.Context{
					"minikube": {AuthInfo: "minikube", Cluster: "minikube"},
				},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"minikube": {
						Token:                 "minikube-token",
						ClientCertificateData: []byte("client certificateData data"),
						ClientKeyData:         []byte("client key data"),
					},
				},
				CurrentContext: "minikube",
			},
			clustername:   "minikube",
			expectedError: false,
		},
	}

	for _, rt := range tests {
		t.Run(rt.description, func(t *testing.T) {
			_, actualError := createKubeconfig(rt.cfg, rt.clustername)

			if (actualError != nil) && !rt.expectedError {
				t.Errorf("%s unexpected failure: %v", rt.description, actualError)
			} else if (actualError == nil) && rt.expectedError {
				t.Errorf("%s passed when expected to fail", rt.description)
			}
		})
	}
}
