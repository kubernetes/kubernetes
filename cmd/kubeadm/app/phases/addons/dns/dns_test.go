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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	core "k8s.io/client-go/testing"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestCreateServiceAccount(t *testing.T) {
	tests := []struct {
		name      string
		createErr error
		expectErr bool
	}{
		{
			"error-free case",
			nil,
			false,
		},
		{
			"duplication errors should be ignored",
			apierrors.NewAlreadyExists(schema.GroupResource{}, ""),
			false,
		},
		{
			"unexpected errors should be returned",
			apierrors.NewUnauthorized(""),
			true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			if tc.createErr != nil {
				client.PrependReactor("create", "serviceaccounts", func(action core.Action) (bool, runtime.Object, error) {
					return true, nil, tc.createErr
				})
			}

			err := CreateServiceAccount(client)
			if tc.expectErr {
				if err == nil {
					t.Errorf("CreateServiceAccounts(%s) wanted err, got nil", tc.name)
				}
				return
			} else if !tc.expectErr && err != nil {
				t.Errorf("CreateServiceAccounts(%s) returned unexpected err: %v", tc.name, err)
			}

			wantResourcesCreated := 1
			if len(client.Actions()) != wantResourcesCreated {
				t.Errorf("CreateServiceAccounts(%s) should have made %d actions, but made %d", tc.name, wantResourcesCreated, len(client.Actions()))
			}

			for _, action := range client.Actions() {
				if action.GetVerb() != "create" || action.GetResource().Resource != "serviceaccounts" {
					t.Errorf("CreateServiceAccounts(%s) called [%v %v], but wanted [create serviceaccounts]",
						tc.name, action.GetVerb(), action.GetResource().Resource)
				}
			}
		})
	}
}

func TestCompileManifests(t *testing.T) {
	replicas := int32(coreDNSReplicas)
	var tests = []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "KubeDNSDeployment manifest",
			manifest: KubeDNSDeployment,
			data: struct {
				DeploymentName, KubeDNSImage, DNSMasqImage, SidecarImage, DNSBindAddr, DNSProbeAddr, DNSDomain, ControlPlaneTaintKey string
				Replicas                                                                                                             *int32
			}{
				DeploymentName:       "foo",
				KubeDNSImage:         "foo",
				DNSMasqImage:         "foo",
				SidecarImage:         "foo",
				DNSBindAddr:          "foo",
				DNSProbeAddr:         "foo",
				DNSDomain:            "foo",
				ControlPlaneTaintKey: "foo",
				Replicas:             &replicas,
			},
		},
		{
			name:     "KubeDNSService manifest",
			manifest: KubeDNSService,
			data: struct{ DNSIP string }{
				DNSIP: "foo",
			},
		},
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
			data: struct{ DNSDomain, Federation, UpstreamNameserver, StubDomain string }{
				DNSDomain:          "foo",
				Federation:         "foo",
				UpstreamNameserver: "foo",
				StubDomain:         "foo",
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
			isDualStack:   false,
		},
		{
			name:          "subnet mask 26",
			svcSubnet:     "10.87.116.64/26",
			expectedDNSIP: "10.87.116.74",
			isDualStack:   false,
		},
		{
			name:          "dual-stack ipv4 primary, subnet mask 26",
			svcSubnet:     "10.87.116.64/26,fd03::/112",
			expectedDNSIP: "10.87.116.74",
			isDualStack:   true,
		},
		{
			name:          "dual-stack ipv6 primary, subnet mask 112",
			svcSubnet:     "fd03::/112,10.87.116.64/26",
			expectedDNSIP: "fd03::a",
			isDualStack:   true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			dnsIP, err := kubeadmconstants.GetDNSIP(rt.svcSubnet, rt.isDualStack)
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

func TestTranslateStubDomainKubeDNSToCoreDNS(t *testing.T) {
	testCases := []struct {
		name      string
		configMap *v1.ConfigMap
		expectOne string
		expectTwo string
	}{
		{
			name: "valid call with multiple IPs",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"stubDomains":         `{"foo.com" : ["1.2.3.4:5300","3.3.3.3"], "my.cluster.local" : ["2.3.4.5"]}`,
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expectOne: `
    foo.com:53 {
       errors
       cache 30
       loop
       forward . 1.2.3.4:5300 3.3.3.3
    }
    
    my.cluster.local:53 {
       errors
       cache 30
       loop
       forward . 2.3.4.5
    }`,
			expectTwo: `
    my.cluster.local:53 {
       errors
       cache 30
       loop
       forward . 2.3.4.5
    }
    
    foo.com:53 {
       errors
       cache 30
       loop
       forward . 1.2.3.4:5300 3.3.3.3
    }`,
		},
		{
			name: "empty call",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
			},

			expectOne: "",
		},
		{
			name: "valid call",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"stubDomains":         `{"foo.com" : ["1.2.3.4:5300"], "my.cluster.local" : ["2.3.4.5"]}`,
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expectOne: `
    foo.com:53 {
       errors
       cache 30
       loop
       forward . 1.2.3.4:5300
    }
    
    my.cluster.local:53 {
       errors
       cache 30
       loop
       forward . 2.3.4.5
    }`,
			expectTwo: `
    my.cluster.local:53 {
       errors
       cache 30
       loop
       forward . 2.3.4.5
    }
    
    foo.com:53 {
       errors
       cache 30
       loop
       forward . 1.2.3.4:5300
    }`,
		},
		{
			name: "If Hostname present: Omit Hostname",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"stubDomains":         `{"bar.com" : ["1.2.3.4:5300","service.consul"], "my.cluster.local" : ["2.3.4.5"], "foo.com" : ["service.consul"]}`,
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expectOne: `
    bar.com:53 {
       errors
       cache 30
       loop
       forward . 1.2.3.4:5300
    }
    
    my.cluster.local:53 {
       errors
       cache 30
       loop
       forward . 2.3.4.5
    }`,
			expectTwo: `
    my.cluster.local:53 {
       errors
       cache 30
       loop
       forward . 2.3.4.5
    }
    
    bar.com:53 {
       errors
       cache 30
       loop
       forward . 1.2.3.4:5300
    }`,
		},
		{
			name: "All hostname: return empty",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"stubDomains":         `{"foo.com" : ["service.consul"], "my.cluster.local" : ["ns.foo.com"]}`,
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expectOne: "",
			expectTwo: "",
		},
		{
			name: "missing stubDomains",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expectOne: "",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			out, err := translateStubDomainOfKubeDNSToForwardCoreDNS(kubeDNSStubDomain, testCase.configMap)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !strings.EqualFold(out, testCase.expectOne) && !strings.EqualFold(out, testCase.expectTwo) {
				t.Errorf("expected to find %q or %q in output: %q", testCase.expectOne, testCase.expectTwo, out)
			}
		})
	}
}

func TestTranslateUpstreamKubeDNSToCoreDNS(t *testing.T) {
	testCases := []struct {
		name      string
		configMap *v1.ConfigMap
		expect    string
	}{
		{
			name: "expect resolv.conf",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
			},

			expect: "/etc/resolv.conf",
		},
		{
			name: "expect list of Name Server IP addresses",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"stubDomains":         ` {"foo.com" : ["1.2.3.4:5300"], "my.cluster.local" : ["2.3.4.5"]}`,
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4", "4.4.4.4"]`,
				},
			},

			expect: "8.8.8.8 8.8.4.4 4.4.4.4",
		},
		{
			name: "no stubDomains: expect list of Name Server IP addresses",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expect: "8.8.8.8 8.8.4.4",
		},
		{
			name: "Hostname present: expect NameServer to omit the hostname",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"upstreamNameservers": `["service.consul", "ns.foo.com", "8.8.4.4", "ns.moo.com", "ns.bar.com"]`,
				},
			},

			expect: "8.8.4.4",
		},
		{
			name: "All hostnames: return empty",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"upstreamNameservers": `["service.consul", "ns.foo.com"]`,
				},
			},

			expect: "",
		},
		{
			name: "IPv6: expect list of Name Server IP addresses",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"upstreamNameservers": `["[2003::1]:53", "8.8.4.4"]`,
				},
			},

			expect: "[2003::1]:53 8.8.4.4",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			out, err := translateUpstreamNameServerOfKubeDNSToUpstreamForwardCoreDNS(kubeDNSUpstreamNameservers, testCase.configMap)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !strings.EqualFold(out, testCase.expect) {
				t.Errorf("expected to find %q in output: %q", testCase.expect, out)
			}
		})
	}
}

func TestTranslateFederationKubeDNSToCoreDNS(t *testing.T) {
	testCases := []struct {
		name      string
		configMap *v1.ConfigMap
		expectOne string
		expectTwo string
	}{
		{
			name: "valid call",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"federations":         `{"foo" : "foo.feddomain.com", "bar" : "bar.feddomain.com"}`,
					"stubDomains":         `{"foo.com" : ["1.2.3.4:5300","3.3.3.3"], "my.cluster.local" : ["2.3.4.5"]}`,
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expectOne: `
        federation cluster.local {
           foo foo.feddomain.com
           bar bar.feddomain.com
        }`,
			expectTwo: `
        federation cluster.local {
           bar bar.feddomain.com
           foo foo.feddomain.com
        }`,
		},
		{
			name: "empty data",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
			},

			expectOne: "",
			expectTwo: "",
		},
		{
			name: "missing federations data",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"stubDomains":         `{"foo.com" : ["1.2.3.4:5300"], "my.cluster.local" : ["2.3.4.5"]}`,
					"upstreamNameservers": `["8.8.8.8", "8.8.4.4"]`,
				},
			},

			expectOne: "",
			expectTwo: "",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			out, err := translateFederationsofKubeDNSToCoreDNS(kubeDNSFederation, "cluster.local", testCase.configMap)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !strings.EqualFold(out, testCase.expectOne) && !strings.EqualFold(out, testCase.expectTwo) {
				t.Errorf("expected to find %q or %q in output: %q", testCase.expectOne, testCase.expectTwo, out)
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
			name:     "KubeDNSDeployment",
			manifest: KubeDNSDeployment,
			data: struct {
				DeploymentName, KubeDNSImage, DNSMasqImage, SidecarImage, DNSBindAddr, DNSProbeAddr, DNSDomain, ControlPlaneTaintKey string
				Replicas                                                                                                             *int32
			}{
				DeploymentName:       "foo",
				KubeDNSImage:         "foo",
				DNSMasqImage:         "foo",
				SidecarImage:         "foo",
				DNSBindAddr:          "foo",
				DNSProbeAddr:         "foo",
				DNSDomain:            "foo",
				ControlPlaneTaintKey: "foo",
				Replicas:             &replicas,
			},
		},
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

func TestCreateCoreDNSConfigMap(t *testing.T) {
	tests := []struct {
		name                 string
		initialCorefileData  string
		expectedCorefileData string
		coreDNSVersion       string
	}{
		{
			name: "Remove Deprecated options",
			initialCorefileData: `.:53 {
        errors
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
    health {
        lameduck 5s
    }
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
    forward . /etc/resolv.conf
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
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := createClientAndCoreDNSManifest(t, tc.initialCorefileData, tc.coreDNSVersion)
			// Get the Corefile and installed CoreDNS version.
			cm, corefile, currentInstalledCoreDNSVersion, err := GetCoreDNSInfo(client)
			if err != nil {
				t.Fatalf("unable to fetch CoreDNS current installed version and ConfigMap.")
			}
			err = migrateCoreDNSCorefile(client, cm, corefile, currentInstalledCoreDNSVersion)
			if err != nil {
				t.Fatalf("error creating the CoreDNS ConfigMap: %v", err)
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
