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

package dns

import (
	"context"
	"strings"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestCompileManifests(t *testing.T) {
	replicas := int32(coreDNSReplicas)
	var tests = []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "CoreDNSDeployment manifest",
			manifest: CoreDNSDeployment,
			data: struct {
				DeploymentName, Image, ControlPlaneTaintKey string
				Replicas                                    *int32
			}{
				DeploymentName:       "foo",
				Image:                "foo",
				ControlPlaneTaintKey: "foo",
				Replicas:             &replicas,
			},
		},
		{
			name:     "CoreDNSConfigMap manifest",
			manifest: CoreDNSConfigMap,
			data: struct{ DNSDomain string }{
				DNSDomain: "foo",
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, err := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
			if err != nil {
				t.Errorf("unexpected ParseTemplate failure: %+v", err)
			}
		})
	}
}

func TestGetDNSIP(t *testing.T) {
	var tests = []struct {
		name, svcSubnet, expectedDNSIP string
		isDualStack                    bool
	}{
		{
			name:          "subnet mask 12",
			svcSubnet:     "10.96.0.0/12",
			expectedDNSIP: "10.96.0.10",
		},
		{
			name:          "subnet mask 26",
			svcSubnet:     "10.87.116.64/26",
			expectedDNSIP: "10.87.116.74",
		},
		{
			name:          "dual-stack ipv4 primary, subnet mask 26",
			svcSubnet:     "10.87.116.64/26,fd03::/112",
			expectedDNSIP: "10.87.116.74",
		},
		{
			name:          "dual-stack ipv6 primary, subnet mask 112",
			svcSubnet:     "fd03::/112,10.87.116.64/26",
			expectedDNSIP: "fd03::a",
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			dnsIP, err := kubeadmconstants.GetDNSIP(rt.svcSubnet)
			if err != nil {
				t.Fatalf("couldn't get dnsIP : %v", err)
			}

			actualDNSIP := dnsIP.String()
			if actualDNSIP != rt.expectedDNSIP {
				t.Errorf(
					"failed GetDNSIP\n\texpected: %s\n\t  actual: %s",
					rt.expectedDNSIP,
					actualDNSIP,
				)
			}
		})
	}
}

func TestDeploymentsHaveSystemClusterCriticalPriorityClassName(t *testing.T) {
	replicas := int32(coreDNSReplicas)
	testCases := []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "CoreDNSDeployment",
			manifest: CoreDNSDeployment,
			data: struct {
				DeploymentName, Image, ControlPlaneTaintKey, CoreDNSConfigMapName string
				Replicas                                                          *int32
			}{
				DeploymentName:       "foo",
				Image:                "foo",
				ControlPlaneTaintKey: "foo",
				CoreDNSConfigMapName: "foo",
				Replicas:             &replicas,
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			deploymentBytes, _ := kubeadmutil.ParseTemplate(testCase.manifest, testCase.data)
			deployment := &apps.Deployment{}
			if err := runtime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), deploymentBytes, deployment); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if deployment.Spec.Template.Spec.PriorityClassName != "system-cluster-critical" {
				t.Errorf("expected to see system-cluster-critical priority class name. Got %q instead", deployment.Spec.Template.Spec.PriorityClassName)
			}
		})
	}
}

func TestCreateCoreDNSAddon(t *testing.T) {
	tests := []struct {
		name                 string
		initialCorefileData  string
		expectedCorefileData string
		coreDNSVersion       string
	}{
		{
			name:                "Empty Corefile",
			initialCorefileData: "",
			expectedCorefileData: `.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.6.7",
		},
		{
			name: "Default Corefile",
			initialCorefileData: `.:53 {
        errors
        health {
            lameduck 5s
        }
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            pods insecure
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
`,
			expectedCorefileData: `.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.6.7",
		},
		{
			name: "Modified Corefile with only newdefaults needed",
			initialCorefileData: `.:53 {
        errors
        log
        health
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            pods insecure
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
`,
			expectedCorefileData: `.:53 {
    errors
    log
    health {
        lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.6.2",
		},
		{
			name: "Default Corefile with rearranged plugins",
			initialCorefileData: `.:53 {
        errors
        cache 30
        prometheus :9153
        forward . /etc/resolv.conf
        loop
        reload
        loadbalance
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        health
    }
`,
			expectedCorefileData: `.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.3.1",
		},
		{
			name: "Remove Deprecated options",
			initialCorefileData: `.:53 {
        errors
        logs
        health
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }`,
			expectedCorefileData: `.:53 {
    errors
    logs
    health {
        lameduck 5s
    }
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
    ready
}
`,
			coreDNSVersion: "1.3.1",
		},
		{
			name: "Update proxy plugin to forward plugin",
			initialCorefileData: `.:53 {
        errors
        health
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
        }
        prometheus :9153
        proxy . /etc/resolv.conf
        k8s_external example.com
        cache 30
        loop
        reload
        loadbalance
    }`,
			expectedCorefileData: `.:53 {
    errors
    health {
        lameduck 5s
    }
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    k8s_external example.com
    cache 30
    loop
    reload
    loadbalance
    ready
}
`,
			coreDNSVersion: "1.3.1",
		},
		{
			name: "Modified Corefile with no migration required",
			initialCorefileData: `consul {
        errors
        forward . 10.10.96.16:8600 10.10.96.17:8600 10.10.96.18:8600 {
            max_concurrent 1000
        }
        loadbalance
        cache 5
        reload
    }
    domain.int {
       errors
       forward . 10.10.0.140 10.10.0.240 10.10.51.40 {
           max_concurrent 1000
       }
       loadbalance
       cache 3600
       reload
    }
    .:53 {
      errors
      health {
          lameduck 5s
      }
      ready
      kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
      }
      prometheus :9153
      forward . /etc/resolv.conf {
          prefer_udp
          max_concurrent 1000
      }
      cache 30
      loop
      reload
      loadbalance
    }
`,
			expectedCorefileData: `consul {
        errors
        forward . 10.10.96.16:8600 10.10.96.17:8600 10.10.96.18:8600 {
            max_concurrent 1000
        }
        loadbalance
        cache 5
        reload
    }
    domain.int {
       errors
       forward . 10.10.0.140 10.10.0.240 10.10.51.40 {
           max_concurrent 1000
       }
       loadbalance
       cache 3600
       reload
    }
    .:53 {
      errors
      health {
          lameduck 5s
      }
      ready
      kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
      }
      prometheus :9153
      forward . /etc/resolv.conf {
          prefer_udp
          max_concurrent 1000
      }
      cache 30
      loop
      reload
      loadbalance
    }
`,
			coreDNSVersion: "1.6.7",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := createClientAndCoreDNSManifest(t, tc.initialCorefileData, tc.coreDNSVersion)

			configMapBytes, err := kubeadmutil.ParseTemplate(CoreDNSConfigMap, struct{ DNSDomain, UpstreamNameserver, StubDomain string }{
				DNSDomain:          "cluster.local",
				UpstreamNameserver: "/etc/resolv.conf",
				StubDomain:         "",
			})
			if err != nil {
				t.Errorf("unexpected ParseTemplate failure: %+v", err)
			}

			err = createCoreDNSAddon(nil, nil, configMapBytes, client)
			if err != nil {
				t.Fatalf("error creating the CoreDNS Addon: %v", err)
			}
			migratedConfigMap, _ := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), kubeadmconstants.CoreDNSConfigMap, metav1.GetOptions{})
			if !strings.EqualFold(migratedConfigMap.Data["Corefile"], tc.expectedCorefileData) {
				t.Fatalf("expected to get %v, but got %v", tc.expectedCorefileData, migratedConfigMap.Data["Corefile"])
			}
		})
	}
}

func createClientAndCoreDNSManifest(t *testing.T, corefile, coreDNSVersion string) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()
	_, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.CoreDNSConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			"Corefile": corefile,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating ConfigMap: %v", err)
	}
	_, err = client.AppsV1().Deployments(metav1.NamespaceSystem).Create(context.TODO(), &apps.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.CoreDNSConfigMap,
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s-app": "kube-dns",
			},
		},
		Spec: apps.DeploymentSpec{
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image: "test:" + coreDNSVersion,
						},
					},
				},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating deployment: %v", err)
	}
	return client
}
