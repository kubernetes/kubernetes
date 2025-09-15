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

package discovery

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclient "k8s.io/client-go/kubernetes/fake"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestFor(t *testing.T) {
	tests := []struct {
		name   string
		d      kubeadm.JoinConfiguration
		expect bool
	}{
		{
			name:   "default Discovery",
			d:      kubeadm.JoinConfiguration{},
			expect: false,
		},
		{
			name: "file Discovery with a path",
			d: kubeadm.JoinConfiguration{
				Discovery: kubeadm.Discovery{
					File: &kubeadm.FileDiscovery{
						KubeConfigPath: "notnil",
					},
				},
			},
			expect: false,
		},
		{
			name: "file Discovery with an url",
			d: kubeadm.JoinConfiguration{
				Discovery: kubeadm.Discovery{
					File: &kubeadm.FileDiscovery{
						KubeConfigPath: "https://localhost",
					},
				},
			},
			expect: false,
		},
		{
			name: "BootstrapTokenDiscovery",
			d: kubeadm.JoinConfiguration{
				Discovery: kubeadm.Discovery{
					BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
						Token: "foo.bar@foobar",
					},
				},
			},
			expect: false,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			config := rt.d
			config.Timeouts = &kubeadm.Timeouts{
				Discovery: &metav1.Duration{Duration: 1 * time.Minute},
			}
			client := fakeclient.NewSimpleClientset()
			_, actual := For(client, &config)
			if (actual == nil) != rt.expect {
				t.Errorf(
					"failed For:\n\texpected: %t\n\t  actual: %t",
					rt.expect,
					(actual == nil),
				)
			}
		})
	}
}
