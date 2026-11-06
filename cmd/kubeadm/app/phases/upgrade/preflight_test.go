/*
Copyright The Kubernetes Authors.

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
	"context"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientsetfake "k8s.io/client-go/kubernetes/fake"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const testCorefile = `.:53 {
    errors
    health
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
    }
    prometheus :9153
    forward . /etc/resolv.conf
    cache 30
    loop
    reload
    loadbalance
}
`

func newCoreDNSClient(t *testing.T, installedTag string) *clientsetfake.Clientset {
	t.Helper()
	client := clientsetfake.NewSimpleClientset()
	replicas := int32(2)
	if _, err := client.AppsV1().Deployments(metav1.NamespaceSystem).Create(context.TODO(), &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.CoreDNSDeploymentName,
			Namespace: metav1.NamespaceSystem,
			Labels:    map[string]string{"k8s-app": "kube-dns"},
		},
		Spec: apps.DeploymentSpec{
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{Containers: []v1.Container{{Image: "registry.k8s.io/coredns/coredns:" + installedTag}}},
			},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("error creating deployment: %v", err)
	}
	if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: kubeadmconstants.CoreDNSConfigMap, Namespace: metav1.NamespaceSystem},
		Data:       map[string]string{"Corefile": testCorefile},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("error creating configmap: %v", err)
	}
	return client
}

func TestRunCoreDNSMigrationCheck(t *testing.T) {
	tests := []struct {
		name         string
		installedTag string
		imageTag     string // cfg.DNS.ImageTag, defaults to kubeadmconstants.CoreDNSVersion when empty
		expectedErr  bool
	}{
		{
			name:         "installed version matches the default target",
			installedTag: kubeadmconstants.CoreDNSVersion,
		},
		{
			name:         "installed version matches an overridden target",
			installedTag: "v1.8.0",
			imageTag:     "v1.8.0",
		},
		{
			name:         "overridden target is known and differs from the default",
			installedTag: "v1.6.2",
			imageTag:     "v1.8.0",
		},
		{
			name:         "overridden target is unknown to corefile-migration",
			installedTag: "v1.6.2",
			imageTag:     "v99.0.0-custom",
			expectedErr:  true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &kubeadmapi.ClusterConfiguration{DNS: kubeadmapi.DNS{ImageMeta: kubeadmapi.ImageMeta{ImageTag: tc.imageTag}}}
			err := RunCoreDNSMigrationCheck(newCoreDNSClient(t, tc.installedTag), sets.New[string](), cfg)
			if (err != nil) != tc.expectedErr {
				t.Fatalf("expected error: %v, got: %v", tc.expectedErr, err)
			}
		})
	}
}

func TestRunCoreDNSMigrationCheckIgnoreErrors(t *testing.T) {
	cfg := &kubeadmapi.ClusterConfiguration{DNS: kubeadmapi.DNS{ImageMeta: kubeadmapi.ImageMeta{ImageTag: "v99.0.0-custom"}}}
	client := newCoreDNSClient(t, "v1.6.2")
	if err := RunCoreDNSMigrationCheck(client, sets.New("corednsmigration", "corednsunsupportedplugins"), cfg); err != nil {
		t.Fatalf("expected the failures to be ignored, got: %v", err)
	}
}
