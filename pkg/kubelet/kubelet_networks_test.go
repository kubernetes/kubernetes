/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io"
	"io/ioutil"
	"net"
	"os"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

func TestNodeIPParam(t *testing.T) {
	testKubelet := newTestKubelet(t)
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

// Tests that identify the host port conflicts are detected correctly.
func TestGetHostPortConflicts(t *testing.T) {
	pods := []*api.Pod{
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 81}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 82}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 83}}}}}},
	}
	// Pods should not cause any conflict.
	if hasHostPortConflicts(pods) {
		t.Errorf("expected no conflicts, Got conflicts")
	}

	expected := &api.Pod{
		Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 81}}}}},
	}
	// The new pod should cause conflict and be reported.
	pods = append(pods, expected)
	if !hasHostPortConflicts(pods) {
		t.Errorf("expected conflict, Got no conflicts")
	}
}

// Tests that we handle port conflicts correctly by setting the failed status in status map.
func TestHandlePortConflicts(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)

	kl.nodeLister = testNodeLister{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: kl.nodeName},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	kl.nodeInfo = testNodeInfo{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: kl.nodeName},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}

	spec := api.PodSpec{NodeName: kl.nodeName, Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "newpod",
				Namespace: "foo",
			},
			Spec: spec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "987654321",
				Name:      "oldpod",
				Namespace: "foo",
			},
			Spec: spec,
		},
	}
	// Make sure the Pods are in the reverse order of creation time.
	pods[1].CreationTimestamp = unversioned.NewTime(time.Now())
	pods[0].CreationTimestamp = unversioned.NewTime(time.Now().Add(1 * time.Second))
	// The newer pod should be rejected.
	notfittingPod := pods[0]
	fittingPod := pods[1]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	// notfittingPod should be Failed
	status, found := kl.statusManager.GetPodStatus(notfittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
	// fittingPod should be Pending
	status, found = kl.statusManager.GetPodStatus(fittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", fittingPod.UID)
	}
	if status.Phase != api.PodPending {
		t.Fatalf("expected pod status %q. Got %q.", api.PodPending, status.Phase)
	}
}

func TestCleanupBandwidthLimits(t *testing.T) {
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
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
						Annotations: map[string]string{
							"kubernetes.io/ingress-bandwidth": "10M",
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
						Annotations: map[string]string{
							"kubernetes.io/ingress-bandwidth": "10M",
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
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

		testKube := newTestKubelet(t)
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

func TestExtractBandwidthResources(t *testing.T) {
	four, _ := resource.ParseQuantity("4M")
	ten, _ := resource.ParseQuantity("10M")
	twenty, _ := resource.ParseQuantity("20M")
	tests := []struct {
		pod             *api.Pod
		expectedIngress *resource.Quantity
		expectedEgress  *resource.Quantity
		expectError     bool
	}{
		{
			pod: &api.Pod{},
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/ingress-bandwidth": "10M",
					},
				},
			},
			expectedIngress: ten,
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/egress-bandwidth": "10M",
					},
				},
			},
			expectedEgress: ten,
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/ingress-bandwidth": "4M",
						"kubernetes.io/egress-bandwidth":  "20M",
					},
				},
			},
			expectedIngress: four,
			expectedEgress:  twenty,
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/ingress-bandwidth": "foo",
					},
				},
			},
			expectError: true,
		},
	}
	for _, test := range tests {
		ingress, egress, err := bandwidth.ExtractPodBandwidthResources(test.pod.Annotations)
		if test.expectError {
			if err == nil {
				t.Errorf("unexpected non-error")
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !reflect.DeepEqual(ingress, test.expectedIngress) {
			t.Errorf("expected: %v, saw: %v", ingress, test.expectedIngress)
		}
		if !reflect.DeepEqual(egress, test.expectedEgress) {
			t.Errorf("expected: %v, saw: %v", egress, test.expectedEgress)
		}
	}
}
