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
	"net"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/bandwidth"
)

func TestNodeIPParam(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
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
		if err != nil && test.success {
			t.Errorf("Test: %s, expected no error but got: %v", test.testName, err)
		} else if err == nil && !test.success {
			t.Errorf("Test: %s, expected an error", test.testName)
		}
	}
}

type countingDNSScrubber struct {
	counter *int
}

func (cds countingDNSScrubber) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	(*cds.counter)++
	return nameservers, searches
}

func TestParseResolvConf(t *testing.T) {
	testCases := []struct {
		data        string
		nameservers []string
		searches    []string
	}{
		{"", []string{}, []string{}},
		{" ", []string{}, []string{}},
		{"\n", []string{}, []string{}},
		{"\t\n\t", []string{}, []string{}},
		{"#comment\n", []string{}, []string{}},
		{" #comment\n", []string{}, []string{}},
		{"#comment\n#comment", []string{}, []string{}},
		{"#comment\nnameserver", []string{}, []string{}},
		{"#comment\nnameserver\nsearch", []string{}, []string{}},
		{"nameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{" nameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"\tnameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"nameserver\t1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"nameserver \t 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"nameserver 1.2.3.4\nnameserver 5.6.7.8", []string{"1.2.3.4", "5.6.7.8"}, []string{}},
		{"search foo", []string{}, []string{"foo"}},
		{"search foo bar", []string{}, []string{"foo", "bar"}},
		{"search foo bar bat\n", []string{}, []string{"foo", "bar", "bat"}},
		{"search foo\nsearch bar", []string{}, []string{"bar"}},
		{"nameserver 1.2.3.4\nsearch foo bar", []string{"1.2.3.4"}, []string{"foo", "bar"}},
		{"nameserver 1.2.3.4\nsearch foo\nnameserver 5.6.7.8\nsearch bar", []string{"1.2.3.4", "5.6.7.8"}, []string{"bar"}},
		{"#comment\nnameserver 1.2.3.4\n#comment\nsearch foo\ncomment", []string{"1.2.3.4"}, []string{"foo"}},
	}
	for i, tc := range testCases {
		ns, srch, err := parseResolvConf(strings.NewReader(tc.data), nil)
		if err != nil {
			t.Errorf("expected success, got %v", err)
			continue
		}
		if !reflect.DeepEqual(ns, tc.nameservers) {
			t.Errorf("[%d] expected nameservers %#v, got %#v", i, tc.nameservers, ns)
		}
		if !reflect.DeepEqual(srch, tc.searches) {
			t.Errorf("[%d] expected searches %#v, got %#v", i, tc.searches, srch)
		}

		counter := 0
		cds := countingDNSScrubber{&counter}
		ns, srch, err = parseResolvConf(strings.NewReader(tc.data), cds)
		if err != nil {
			t.Errorf("expected success, got %v", err)
			continue
		}
		if !reflect.DeepEqual(ns, tc.nameservers) {
			t.Errorf("[%d] expected nameservers %#v, got %#v", i, tc.nameservers, ns)
		}
		if !reflect.DeepEqual(srch, tc.searches) {
			t.Errorf("[%d] expected searches %#v, got %#v", i, tc.searches, srch)
		}
		if counter != 1 {
			t.Errorf("[%d] expected dnsScrubber to have been called: got %d", i, counter)
		}
	}
}

func TestCleanupBandwidthLimits(t *testing.T) {
	testPod := func(name, ingress string) *api.Pod {
		pod := podWithUidNameNs("", name, "")

		if len(ingress) != 0 {
			pod.Annotations["kubernetes.io/ingress-bandwidth"] = ingress
		}

		return pod
	}

	// TODO(random-liu): We removed the test case for pod status not cached here. We should add a higher
	// layer status getter function and test that function instead.
	tests := []struct {
		status           *api.PodStatus
		pods             []*api.Pod
		inputCIDRs       []string
		expectResetCIDRs []string
		name             string
	}{
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodRunning,
			},
			pods: []*api.Pod{
				testPod("foo", "10M"),
				testPod("bar", ""),
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"2.3.4.5/32", "5.6.7.8/32"},
			name:             "pod running",
		},
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodFailed,
			},
			pods: []*api.Pod{
				testPod("foo", "10M"),
				testPod("bar", ""),
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			name:             "pod not running",
		},
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodFailed,
			},
			pods: []*api.Pod{
				testPod("foo", ""),
				testPod("bar", ""),
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			name:             "no bandwidth limits",
		},
	}
	for _, test := range tests {
		shaper := &bandwidth.FakeShaper{
			CIDRs: test.inputCIDRs,
		}

		testKube := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		testKube.kubelet.shaper = shaper

		for _, pod := range test.pods {
			testKube.kubelet.statusManager.SetPodStatus(pod, *test.status)
		}

		err := testKube.kubelet.cleanupBandwidthLimits(test.pods)
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", test.name, err)
		}
		if !reflect.DeepEqual(shaper.ResetCIDRs, test.expectResetCIDRs) {
			t.Errorf("[%s]\nexpected: %v, saw: %v", test.name, test.expectResetCIDRs, shaper.ResetCIDRs)
		}
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
		if res != tc.expect {
			t.Errorf("getIPTablesMark output unexpected result: %v when input bit is %d. Expect result: %v", res, tc.bit, tc.expect)
		}
	}
}
