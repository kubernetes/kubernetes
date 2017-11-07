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

package kubelet

import (
	"fmt"
	"net"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
)

func TestNetworkHostGetsPodNotFound(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	actualPod, _ := nh.GetPodByName("", "")
	if actualPod != nil {
		t.Fatalf("Was expected nil, received %v instead", actualPod)
	}
}

func TestNetworkHostGetsKubeClient(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.GetKubeClient() != testKubelet.fakeKubeClient {
		t.Fatalf("NetworkHost client does not match testKubelet's client")
	}
}

func TestNetworkHostGetsRuntime(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.GetRuntime() != testKubelet.fakeRuntime {
		t.Fatalf("NetworkHost runtime does not match testKubelet's runtime")
	}
}

func TestNetworkHostSupportsLegacyFeatures(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.SupportsLegacyFeatures() == false {
		t.Fatalf("SupportsLegacyFeatures should not be false")
	}
}

func TestNoOpHostGetsName(t *testing.T) {
	nh := NoOpLegacyHost{}
	pod, err := nh.GetPodByName("", "")
	if pod != nil && err != true {
		t.Fatalf("noOpLegacyHost getpodbyname expected to be nil and true")
	}
}

func TestNoOpHostGetsKubeClient(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.GetKubeClient() != nil {
		t.Fatalf("noOpLegacyHost client expected to be nil")
	}
}

func TestNoOpHostGetsRuntime(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.GetRuntime() != nil {
		t.Fatalf("noOpLegacyHost runtime expected to be nil")
	}
}

func TestNoOpHostSupportsLegacyFeatures(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.SupportsLegacyFeatures() != false {
		t.Fatalf("noOpLegacyHost legacy features expected to be false")
	}
}

func TestNodeIPParam(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	tests := []struct {
		nodeIP   string
		success  bool
		testName string
	}{
		{
			nodeIP:   "",
			success:  true,
			testName: "IP not set",
		},
		{
			nodeIP:   "127.0.0.1",
			success:  false,
			testName: "loopback address",
		},
		{
			nodeIP:   "FE80::0202:B3FF:FE1E:8329",
			success:  false,
			testName: "IPv6 address",
		},
		{
			nodeIP:   "1.2.3.4",
			success:  false,
			testName: "IPv4 address that doesn't belong to host",
		},
	}
	for _, test := range tests {
		kubelet.nodeIP = net.ParseIP(test.nodeIP)
		err := kubelet.validateNodeIP()
		if test.success {
			assert.NoError(t, err, "test %s", test.testName)
		} else {
			assert.Error(t, err, fmt.Sprintf("test %s", test.testName))
		}
	}
}

func TestParseResolvConf(t *testing.T) {
	testCases := []struct {
		data        string
		nameservers []string
		searches    []string
		options     []string
	}{
		{"", []string{}, []string{}, []string{}},
		{" ", []string{}, []string{}, []string{}},
		{"\n", []string{}, []string{}, []string{}},
		{"\t\n\t", []string{}, []string{}, []string{}},
		{"#comment\n", []string{}, []string{}, []string{}},
		{" #comment\n", []string{}, []string{}, []string{}},
		{"#comment\n#comment", []string{}, []string{}, []string{}},
		{"#comment\nnameserver", []string{}, []string{}, []string{}},
		{"#comment\nnameserver\nsearch", []string{}, []string{}, []string{}},
		{"nameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}, []string{}},
		{" nameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}, []string{}},
		{"\tnameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}, []string{}},
		{"nameserver\t1.2.3.4", []string{"1.2.3.4"}, []string{}, []string{}},
		{"nameserver \t 1.2.3.4", []string{"1.2.3.4"}, []string{}, []string{}},
		{"nameserver 1.2.3.4\nnameserver 5.6.7.8", []string{"1.2.3.4", "5.6.7.8"}, []string{}, []string{}},
		{"nameserver 1.2.3.4 #comment", []string{"1.2.3.4"}, []string{}, []string{}},
		{"search foo", []string{}, []string{"foo"}, []string{}},
		{"search foo bar", []string{}, []string{"foo", "bar"}, []string{}},
		{"search foo bar bat\n", []string{}, []string{"foo", "bar", "bat"}, []string{}},
		{"search foo\nsearch bar", []string{}, []string{"bar"}, []string{}},
		{"nameserver 1.2.3.4\nsearch foo bar", []string{"1.2.3.4"}, []string{"foo", "bar"}, []string{}},
		{"nameserver 1.2.3.4\nsearch foo\nnameserver 5.6.7.8\nsearch bar", []string{"1.2.3.4", "5.6.7.8"}, []string{"bar"}, []string{}},
		{"#comment\nnameserver 1.2.3.4\n#comment\nsearch foo\ncomment", []string{"1.2.3.4"}, []string{"foo"}, []string{}},
		{"options ndots:5 attempts:2", []string{}, []string{}, []string{"ndots:5", "attempts:2"}},
		{"options ndots:1\noptions ndots:5 attempts:3", []string{}, []string{}, []string{"ndots:5", "attempts:3"}},
		{"nameserver 1.2.3.4\nsearch foo\nnameserver 5.6.7.8\nsearch bar\noptions ndots:5 attempts:4", []string{"1.2.3.4", "5.6.7.8"}, []string{"bar"}, []string{"ndots:5", "attempts:4"}},
	}
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	for i, tc := range testCases {
		ns, srch, opts, err := kubelet.parseResolvConf(strings.NewReader(tc.data))
		require.NoError(t, err)
		assert.EqualValues(t, tc.nameservers, ns, "test case [%d]: name servers", i)
		assert.EqualValues(t, tc.searches, srch, "test case [%d] searches", i)
		assert.EqualValues(t, tc.options, opts, "test case [%d] options", i)
	}
}

func TestComposeDNSSearch(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	recorder := record.NewFakeRecorder(20)
	kubelet.recorder = recorder

	pod := podWithUIDNameNs("", "test_pod", "testNS")
	kubelet.clusterDomain = "TEST"

	testCases := []struct {
		dnsNames     []string
		hostNames    []string
		resultSearch []string
		events       []string
	}{
		{
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST"},
			[]string{},
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST"},
			[]string{},
		},

		{
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST"},
			[]string{"AAA", "svc.TEST", "BBB", "TEST"},
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST", "AAA", "BBB"},
			[]string{},
		},

		{
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST"},
			[]string{"AAA", strings.Repeat("B", 256), "BBB"},
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST", "AAA"},
			[]string{"Search Line limits were exceeded, some dns names have been omitted, the applied search line is: testNS.svc.TEST svc.TEST TEST AAA"},
		},

		{
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST"},
			[]string{"AAA", "TEST", "BBB", "TEST", "CCC", "DDD"},
			[]string{"testNS.svc.TEST", "svc.TEST", "TEST", "AAA", "BBB", "CCC"},
			[]string{
				"Search Line limits were exceeded, some dns names have been omitted, the applied search line is: testNS.svc.TEST svc.TEST TEST AAA BBB CCC",
			},
		},
	}

	fetchEvent := func(recorder *record.FakeRecorder) string {
		select {
		case event := <-recorder.Events:
			return event
		default:
			return "No more events!"
		}
	}

	for i, tc := range testCases {
		dnsSearch := kubelet.formDNSSearch(tc.hostNames, pod)
		assert.EqualValues(t, tc.resultSearch, dnsSearch, "test [%d]", i)
		for _, expectedEvent := range tc.events {
			expected := fmt.Sprintf("%s %s %s", v1.EventTypeWarning, "DNSSearchForming", expectedEvent)
			event := fetchEvent(recorder)
			assert.Equal(t, expected, event, "test [%d]", i)
		}
	}
}

func TestGetClusterDNS(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	clusterNS := "203.0.113.1"
	kubelet.clusterDomain = "kubernetes.io"
	kubelet.clusterDNS = []net.IP{net.ParseIP(clusterNS)}

	pods := newTestPods(4)
	pods[0].Spec.DNSPolicy = v1.DNSClusterFirstWithHostNet
	pods[1].Spec.DNSPolicy = v1.DNSClusterFirst
	pods[2].Spec.DNSPolicy = v1.DNSClusterFirst
	pods[2].Spec.HostNetwork = false
	pods[3].Spec.DNSPolicy = v1.DNSDefault

	options := make([]struct {
		DNS       []string
		DNSSearch []string
	}, 4)
	for i, pod := range pods {
		var err error
		options[i].DNS, options[i].DNSSearch, _, _, err = kubelet.GetClusterDNS(pod)
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	if len(options[0].DNS) != 1 || options[0].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %+v", clusterNS, options[0].DNS)
	}
	if len(options[0].DNSSearch) == 0 || options[0].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected search %s, got %+v", ".svc."+kubelet.clusterDomain, options[0].DNSSearch)
	}
	if len(options[1].DNS) != 1 || options[1].DNS[0] != "127.0.0.1" {
		t.Errorf("expected nameserver 127.0.0.1, got %+v", options[1].DNS)
	}
	if len(options[1].DNSSearch) != 1 || options[1].DNSSearch[0] != "." {
		t.Errorf("expected search \".\", got %+v", options[1].DNSSearch)
	}
	if len(options[2].DNS) != 1 || options[2].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %+v", clusterNS, options[2].DNS)
	}
	if len(options[2].DNSSearch) == 0 || options[2].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected search %s, got %+v", ".svc."+kubelet.clusterDomain, options[2].DNSSearch)
	}
	if len(options[3].DNS) != 1 || options[3].DNS[0] != "127.0.0.1" {
		t.Errorf("expected nameserver 127.0.0.1, got %+v", options[3].DNS)
	}
	if len(options[3].DNSSearch) != 1 || options[3].DNSSearch[0] != "." {
		t.Errorf("expected search \".\", got %+v", options[3].DNSSearch)
	}

	kubelet.resolverConfig = "/etc/resolv.conf"
	for i, pod := range pods {
		var err error
		options[i].DNS, options[i].DNSSearch, _, _, err = kubelet.GetClusterDNS(pod)
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	t.Logf("nameservers %+v", options[1].DNS)
	if len(options[0].DNS) != 1 {
		t.Errorf("expected cluster nameserver only, got %+v", options[0].DNS)
	} else if options[0].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %v", clusterNS, options[0].DNS[0])
	}
	expLength := len(options[1].DNSSearch) + 3
	if expLength > 6 {
		expLength = 6
	}
	if len(options[0].DNSSearch) != expLength {
		t.Errorf("expected prepend of cluster domain, got %+v", options[0].DNSSearch)
	} else if options[0].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected domain %s, got %s", ".svc."+kubelet.clusterDomain, options[0].DNSSearch)
	}
	if len(options[2].DNS) != 1 {
		t.Errorf("expected cluster nameserver only, got %+v", options[2].DNS)
	} else if options[2].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %v", clusterNS, options[2].DNS[0])
	}
	if len(options[2].DNSSearch) != expLength {
		t.Errorf("expected prepend of cluster domain, got %+v", options[2].DNSSearch)
	} else if options[2].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected domain %s, got %s", ".svc."+kubelet.clusterDomain, options[0].DNSSearch)
	}
}

func TestGetIPTablesMark(t *testing.T) {
	tests := []struct {
		bit    int
		expect string
	}{
		{
			14,
			"0x00004000/0x00004000",
		},
		{
			15,
			"0x00008000/0x00008000",
		},
	}
	for _, tc := range tests {
		res := getIPTablesMark(tc.bit)
		assert.Equal(t, tc.expect, res, "input %d", tc.bit)
	}
}
