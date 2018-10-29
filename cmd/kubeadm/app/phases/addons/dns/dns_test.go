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
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	api "k8s.io/kubernetes/pkg/apis/core"
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
			apierrors.NewAlreadyExists(api.Resource(""), ""),
			false,
		},
		{
			"unexpected errors should be returned",
			apierrors.NewUnauthorized(""),
			true,
		},
	}

	for _, tc := range tests {
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
			continue
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

	}
}

func TestCompileManifests(t *testing.T) {
	var tests = []struct {
		manifest string
		data     interface{}
		expected bool
	}{
		{
			manifest: KubeDNSDeployment,
			data: struct{ ImageRepository, Version, DNSBindAddr, DNSProbeAddr, DNSDomain, MasterTaintKey string }{
				ImageRepository: "foo",
				Version:         "foo",
				DNSBindAddr:     "foo",
				DNSProbeAddr:    "foo",
				DNSDomain:       "foo",
				MasterTaintKey:  "foo",
			},
			expected: true,
		},
		{
			manifest: KubeDNSService,
			data: struct{ DNSIP string }{
				DNSIP: "foo",
			},
			expected: true,
		},
		{
			manifest: CoreDNSDeployment,
			data: struct{ ImageRepository, MasterTaintKey, Version string }{
				ImageRepository: "foo",
				MasterTaintKey:  "foo",
				Version:         "foo",
			},
			expected: true,
		},
		{
			manifest: KubeDNSService,
			data: struct{ DNSIP string }{
				DNSIP: "foo",
			},
			expected: true,
		},
		{
			manifest: CoreDNSConfigMap,
			data: struct{ DNSDomain, Federation, UpstreamNameserver, StubDomain string }{
				DNSDomain:          "foo",
				Federation:         "foo",
				UpstreamNameserver: "foo",
				StubDomain:         "foo",
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		_, actual := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CompileManifests:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestGetDNSIP(t *testing.T) {
	var tests = []struct {
		svcSubnet, expectedDNSIP string
	}{
		{
			svcSubnet:     "10.96.0.0/12",
			expectedDNSIP: "10.96.0.10",
		},
		{
			svcSubnet:     "10.87.116.64/26",
			expectedDNSIP: "10.87.116.74",
		},
	}
	for _, rt := range tests {
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
	}
}

func TestTranslateStubDomainKubeDNSToCoreDNS(t *testing.T) {
	testCases := []struct {
		configMap *v1.ConfigMap
		expectOne string
		expectTwo string
	}{
		{
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
       proxy . 1.2.3.4:5300 3.3.3.3
    }
    
    my.cluster.local:53 {
       errors
       cache 30
       loop
       proxy . 2.3.4.5
    }`,
			expectTwo: `
    my.cluster.local:53 {
       errors
       cache 30
       loop
       proxy . 2.3.4.5
    }
    
    foo.com:53 {
       errors
       cache 30
       loop
       proxy . 1.2.3.4:5300 3.3.3.3
    }`,
		},
		{
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
			},

			expectOne: "",
		},
		{
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
       proxy . 1.2.3.4:5300
    }
    
    my.cluster.local:53 {
       errors
       cache 30
       loop
       proxy . 2.3.4.5
    }`,
			expectTwo: `
    my.cluster.local:53 {
       errors
       cache 30
       loop
       proxy . 2.3.4.5
    }
    
    foo.com:53 {
       errors
       cache 30
       loop
       proxy . 1.2.3.4:5300
    }`,
		},
		{
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
		out, err := translateStubDomainOfKubeDNSToProxyCoreDNS(kubeDNSStubDomain, testCase.configMap)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !strings.Contains(out, testCase.expectOne) && !strings.Contains(out, testCase.expectTwo) {
			t.Errorf("expected to find %q or %q in output: %q", testCase.expectOne, testCase.expectTwo, out)
		}
	}
}

func TestTranslateUpstreamKubeDNSToCoreDNS(t *testing.T) {
	testCases := []struct {
		configMap *v1.ConfigMap
		expect    string
	}{
		{
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-dns",
					Namespace: "kube-system",
				},
			},

			expect: "/etc/resolv.conf",
		},
		{
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
	}
	for _, testCase := range testCases {
		out, err := translateUpstreamNameServerOfKubeDNSToUpstreamProxyCoreDNS(kubeDNSUpstreamNameservers, testCase.configMap)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !strings.Contains(out, testCase.expect) {
			t.Errorf("expected to find %q in output: %q", testCase.expect, out)
		}
	}
}

func TestTranslateFederationKubeDNSToCoreDNS(t *testing.T) {
	testCases := []struct {
		configMap *v1.ConfigMap
		expectOne string
		expectTwo string
	}{
		{
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
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kubedns",
					Namespace: "kube-system",
				},
			},

			expectOne: "",
		},
		{
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
		},
	}
	for _, testCase := range testCases {
		out, err := translateFederationsofKubeDNSToCoreDNS(kubeDNSFederation, "cluster.local", testCase.configMap)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !strings.Contains(out, testCase.expectOne) && !strings.Contains(out, testCase.expectTwo) {
			t.Errorf("expected to find %q or %q in output: %q", testCase.expectOne, testCase.expectTwo, out)
		}
	}
}
