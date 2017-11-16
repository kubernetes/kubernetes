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
	"fmt"
	"net"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
	for i, tc := range testCases {
		ns, srch, opts, err := parseResolvConf(strings.NewReader(tc.data))
		require.NoError(t, err)
		assert.EqualValues(t, tc.nameservers, ns, "test case [%d]: name servers", i)
		assert.EqualValues(t, tc.searches, srch, "test case [%d] searches", i)
		assert.EqualValues(t, tc.options, opts, "test case [%d] options", i)
	}
}

func TestComposeDNSSearch(t *testing.T) {
	recorder := record.NewFakeRecorder(20)
	nodeRef := &v1.ObjectReference{
		Kind:      "Node",
		Name:      string("testNode"),
		UID:       types.UID("testNode"),
		Namespace: "",
	}
	testClusterDNSDomain := "TEST"

	configurer := NewConfigurer(recorder, nodeRef, nil, nil, testClusterDNSDomain, "")

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:         "",
			Name:        "test_pod",
			Namespace:   "testNS",
			Annotations: map[string]string{},
		},
	}

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
		dnsSearch := configurer.formDNSSearch(tc.hostNames, pod)
		assert.EqualValues(t, tc.resultSearch, dnsSearch, "test [%d]", i)
		for _, expectedEvent := range tc.events {
			expected := fmt.Sprintf("%s %s %s", v1.EventTypeWarning, "DNSSearchForming", expectedEvent)
			event := fetchEvent(recorder)
			assert.Equal(t, expected, event, "test [%d]", i)
		}
	}
}

func TestGetClusterDNS(t *testing.T) {
	recorder := record.NewFakeRecorder(20)
	nodeRef := &v1.ObjectReference{
		Kind:      "Node",
		Name:      string("testNode"),
		UID:       types.UID("testNode"),
		Namespace: "",
	}
	clusterNS := "203.0.113.1"
	testClusterDNSDomain := "kubernetes.io"
	testClusterDNS := []net.IP{net.ParseIP(clusterNS)}

	configurer := NewConfigurer(recorder, nodeRef, nil, testClusterDNS, testClusterDNSDomain, "")

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
		options[i].DNS, options[i].DNSSearch, _, _, err = configurer.GetClusterDNS(pod)
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	if len(options[0].DNS) != 1 || options[0].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %+v", clusterNS, options[0].DNS)
	}
	if len(options[0].DNSSearch) == 0 || options[0].DNSSearch[0] != ".svc."+configurer.ClusterDomain {
		t.Errorf("expected search %s, got %+v", ".svc."+configurer.ClusterDomain, options[0].DNSSearch)
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
	if len(options[2].DNSSearch) == 0 || options[2].DNSSearch[0] != ".svc."+configurer.ClusterDomain {
		t.Errorf("expected search %s, got %+v", ".svc."+configurer.ClusterDomain, options[2].DNSSearch)
	}
	if len(options[3].DNS) != 1 || options[3].DNS[0] != "127.0.0.1" {
		t.Errorf("expected nameserver 127.0.0.1, got %+v", options[3].DNS)
	}
	if len(options[3].DNSSearch) != 1 || options[3].DNSSearch[0] != "." {
		t.Errorf("expected search \".\", got %+v", options[3].DNSSearch)
	}

	testResolverConfig := "/etc/resolv.conf"
	configurer = NewConfigurer(recorder, nodeRef, nil, testClusterDNS, testClusterDNSDomain, testResolverConfig)
	for i, pod := range pods {
		var err error
		options[i].DNS, options[i].DNSSearch, _, _, err = configurer.GetClusterDNS(pod)
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
	} else if options[0].DNSSearch[0] != ".svc."+configurer.ClusterDomain {
		t.Errorf("expected domain %s, got %s", ".svc."+configurer.ClusterDomain, options[0].DNSSearch)
	}
	if len(options[2].DNS) != 1 {
		t.Errorf("expected cluster nameserver only, got %+v", options[2].DNS)
	} else if options[2].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %v", clusterNS, options[2].DNS[0])
	}
	if len(options[2].DNSSearch) != expLength {
		t.Errorf("expected prepend of cluster domain, got %+v", options[2].DNSSearch)
	} else if options[2].DNSSearch[0] != ".svc."+configurer.ClusterDomain {
		t.Errorf("expected domain %s, got %s", ".svc."+configurer.ClusterDomain, options[0].DNSSearch)
	}
}

func newTestPods(count int) []*v1.Pod {
	pods := make([]*v1.Pod, count)
	for i := 0; i < count; i++ {
		pods[i] = &v1.Pod{
			Spec: v1.PodSpec{
				HostNetwork: true,
			},
			ObjectMeta: metav1.ObjectMeta{
				UID:  types.UID(10000 + i),
				Name: fmt.Sprintf("pod%d", i),
			},
		}
	}
	return pods
}
