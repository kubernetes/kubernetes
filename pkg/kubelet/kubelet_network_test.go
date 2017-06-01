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
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/bandwidth"
)

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
		{"nameserver 1.2.3.4 #comment", []string{"1.2.3.4"}, []string{}},
		{"search foo", []string{}, []string{"foo"}},
		{"search foo bar", []string{}, []string{"foo", "bar"}},
		{"search foo bar bat\n", []string{}, []string{"foo", "bar", "bat"}},
		{"search foo\nsearch bar", []string{}, []string{"bar"}},
		{"nameserver 1.2.3.4\nsearch foo bar", []string{"1.2.3.4"}, []string{"foo", "bar"}},
		{"nameserver 1.2.3.4\nsearch foo\nnameserver 5.6.7.8\nsearch bar", []string{"1.2.3.4", "5.6.7.8"}, []string{"bar"}},
		{"#comment\nnameserver 1.2.3.4\n#comment\nsearch foo\ncomment", []string{"1.2.3.4"}, []string{"foo"}},
	}
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	for i, tc := range testCases {
		ns, srch, err := kubelet.parseResolvConf(strings.NewReader(tc.data))
		require.NoError(t, err)
		assert.EqualValues(t, tc.nameservers, ns, "test case [%d]: name servers", i)
		assert.EqualValues(t, tc.searches, srch, "test case [%d] searches", i)
	}
}

func TestComposeDNSSearch(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	recorder := record.NewFakeRecorder(20)
	kubelet.recorder = recorder

	pod := podWithUidNameNs("", "test_pod", "testNS")
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
			[]string{
				"Found and omitted duplicated dns domain in host search line: 'svc.TEST' during merging with cluster dns domains",
				"Found and omitted duplicated dns domain in host search line: 'TEST' during merging with cluster dns domains",
			},
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
				"Found and omitted duplicated dns domain in host search line: 'TEST' during merging with cluster dns domains",
				"Found and omitted duplicated dns domain in host search line: 'TEST' during merging with cluster dns domains",
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

func TestCleanupBandwidthLimits(t *testing.T) {
	testPod := func(name, ingress string) *v1.Pod {
		pod := podWithUidNameNs("", name, "")

		if len(ingress) != 0 {
			pod.Annotations["kubernetes.io/ingress-bandwidth"] = ingress
		}

		return pod
	}

	// TODO(random-liu): We removed the test case for pod status not cached here. We should add a higher
	// layer status getter function and test that function instead.
	tests := []struct {
		status           *v1.PodStatus
		pods             []*v1.Pod
		inputCIDRs       []string
		expectResetCIDRs []string
		name             string
	}{
		{
			status: &v1.PodStatus{
				PodIP: "1.2.3.4",
				Phase: v1.PodRunning,
			},
			pods: []*v1.Pod{
				testPod("foo", "10M"),
				testPod("bar", ""),
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"2.3.4.5/32", "5.6.7.8/32"},
			name:             "pod running",
		},
		{
			status: &v1.PodStatus{
				PodIP: "1.2.3.4",
				Phase: v1.PodFailed,
			},
			pods: []*v1.Pod{
				testPod("foo", "10M"),
				testPod("bar", ""),
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			name:             "pod not running",
		},
		{
			status: &v1.PodStatus{
				PodIP: "1.2.3.4",
				Phase: v1.PodFailed,
			},
			pods: []*v1.Pod{
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
		defer testKube.Cleanup()
		testKube.kubelet.shaper = shaper

		for _, pod := range test.pods {
			testKube.kubelet.statusManager.SetPodStatus(pod, *test.status)
		}

		err := testKube.kubelet.cleanupBandwidthLimits(test.pods)
		assert.NoError(t, err, "test [%s]", test.name)
		assert.EqualValues(t, test.expectResetCIDRs, shaper.ResetCIDRs, "test[%s]", test.name)
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
