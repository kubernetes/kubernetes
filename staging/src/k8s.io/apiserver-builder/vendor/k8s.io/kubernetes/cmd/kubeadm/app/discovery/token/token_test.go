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

package token

import (
	"strconv"
	"testing"
	"time"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

func TestRunForEndpointsAndReturnFirst(t *testing.T) {
	tests := []struct {
		endpoints        []string
		expectedEndpoint string
	}{
		{
			endpoints:        []string{"1", "2", "3"},
			expectedEndpoint: "1",
		},
		{
			endpoints:        []string{"6", "5"},
			expectedEndpoint: "5",
		},
		{
			endpoints:        []string{"10", "4"},
			expectedEndpoint: "4",
		},
	}
	for _, rt := range tests {
		returnKubeConfig := runForEndpointsAndReturnFirst(rt.endpoints, func(endpoint string) (*clientcmdapi.Config, error) {
			timeout, _ := strconv.Atoi(endpoint)
			time.Sleep(time.Second * time.Duration(timeout))
			return kubeconfigutil.CreateBasic(endpoint, "foo", "foo", []byte{}), nil
		})
		endpoint := returnKubeConfig.Clusters[returnKubeConfig.Contexts[returnKubeConfig.CurrentContext].Cluster].Server
		if endpoint != rt.expectedEndpoint {
			t.Errorf(
				"failed TestRunForEndpointsAndReturnFirst:\n\texpected: %s\n\t  actual: %s",
				endpoint,
				rt.expectedEndpoint,
			)
		}
	}
}
