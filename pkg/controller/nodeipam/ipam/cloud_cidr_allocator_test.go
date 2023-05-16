//go:build !providerless
// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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

package ipam

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
)

func hasNodeInProcessing(ca *cloudCIDRAllocator, name string) bool {
	ca.lock.Lock()
	defer ca.lock.Unlock()

	_, found := ca.nodesInProcessing[name]
	return found
}

func TestBoundedRetries(t *testing.T) {
	clientSet := fake.NewSimpleClientset()
	updateChan := make(chan string, 1) // need to buffer as we are using only on go routine
	sharedInfomer := informers.NewSharedInformerFactory(clientSet, 1*time.Hour)
	ca := &cloudCIDRAllocator{
		client:            clientSet,
		nodeUpdateChannel: updateChan,
		nodeLister:        sharedInfomer.Core().V1().Nodes().Lister(),
		nodesSynced:       sharedInfomer.Core().V1().Nodes().Informer().HasSynced,
		nodesInProcessing: map[string]*nodeProcessingInfo{},
	}
	logger, ctx := ktesting.NewTestContext(t)
	go ca.worker(ctx)
	nodeName := "testNode"
	ca.AllocateOrOccupyCIDR(logger, &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
	})
	for hasNodeInProcessing(ca, nodeName) {
		// wait for node to finish processing (should terminate and not time out)
	}
}

func withinExpectedRange(got time.Duration, expected time.Duration) bool {
	return got >= expected/2 && got <= 3*expected/2
}

func TestNodeUpdateRetryTimeout(t *testing.T) {
	for _, tc := range []struct {
		count int
		want  time.Duration
	}{
		{count: 0, want: 250 * time.Millisecond},
		{count: 1, want: 500 * time.Millisecond},
		{count: 2, want: 1000 * time.Millisecond},
		{count: 3, want: 2000 * time.Millisecond},
		{count: 50, want: 5000 * time.Millisecond},
	} {
		t.Run(fmt.Sprintf("count %d", tc.count), func(t *testing.T) {
			if got := nodeUpdateRetryTimeout(tc.count); !withinExpectedRange(got, tc.want) {
				t.Errorf("nodeUpdateRetryTimeout(tc.count) = %v; want %v", got, tc.want)
			}
		})
	}
}

func TestNeedPodCIDRsUpdate(t *testing.T) {
	for _, tc := range []struct {
		desc         string
		cidrs        []string
		nodePodCIDRs []string
		want         bool
		wantErr      bool
	}{
		{
			desc:  "ipv4 and not assigned CIDRs to node",
			cidrs: []string{"10.10.10.0/24"},
			want:  true,
		}, {
			desc:  "ipv6 and not assigned CIDRs to node",
			cidrs: []string{"2001:db8::/64"},
			want:  true,
		}, {
			desc:  "dual stack and not assigned CIDRs to node",
			cidrs: []string{"10.10.10.0/24", "2001:db8::/64"},
			want:  true,
		}, {
			desc:         "ipv4 and assigned CIDRs to node",
			cidrs:        []string{"10.10.10.0/24"},
			nodePodCIDRs: []string{"10.10.10.0/24"},
			want:         false,
		}, {
			desc:         "ipv6 and assigned CIDRs to node",
			cidrs:        []string{"2001:db8::/64"},
			nodePodCIDRs: []string{"2001:db8::/64"},
			want:         false,
		}, {
			desc:         "dual stack and one empty",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDRs: []string{"10.10.10.0/24"},
			want:         true,
		}, {
			desc:         "dual stack and one empty wrong order",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDRs: []string{"2001:db8::/64"},
			wantErr:      true,
		}, {
			desc:         "dual stack and none empty matching",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDRs: []string{"10.10.10.0/24", "2001:db8::/64"},
			want:         false,
		}, {
			desc:         "dual stack and none empty no matching",
			cidrs:        []string{"10.10.1.0/24", "2001:db8::/64"},
			nodePodCIDRs: []string{"10.10.10.0/24", "2001:db8::/64"},
			wantErr:      true,
		},
	} {
		var node v1.Node
		node.Spec.PodCIDRs = tc.nodePodCIDRs
		logger, _ := ktesting.NewTestContext(t)
		t.Run(tc.desc, func(t *testing.T) {
			got, err := needPodCIDRsUpdate(logger, &node, tc.cidrs)
			if tc.wantErr == (err == nil) {
				t.Errorf("err: %v, wantErr: %v", err, tc.wantErr)
			}
			if err == nil && got != tc.want {
				t.Errorf("got: %v, want: %v", got, tc.want)
			}
		})
	}
}

func Test_filterCIDRs(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		cidrs []string
		want  []string
	}{
		{
			desc:  "ipv4 only",
			cidrs: []string{"10.10.0.0/24"},
			want:  []string{"10.10.0.0/24"},
		}, {
			desc:  "ipv6 only",
			cidrs: []string{"2001:db8::/64"},
			want:  []string{"2001:db8::/64"},
		}, {
			desc:  "ipv4 ipv6",
			cidrs: []string{"10.10.0.0/24", "2001:db8::/64"},
			want:  []string{"10.10.0.0/24", "2001:db8::/64"},
		}, {
			desc:  "ipv6 ipv4",
			cidrs: []string{"2001:db8::/64", "10.10.0.0/24"},
			want:  []string{"2001:db8::/64", "10.10.0.0/24"},
		}, {
			desc:  "multiple ipv4",
			cidrs: []string{"10.10.0.0/24", "192.168.0.0/26"},
			want:  []string{"10.10.0.0/24"},
		}, {
			desc:  "ipv6 only",
			cidrs: []string{"2001:db8::/64", "2001:db8:1234::/64"},
			want:  []string{"2001:db8::/64"},
		}, {
			desc:  "ipv4 ipv6 ipv6",
			cidrs: []string{"10.10.0.0/24", "2001:db8::/64", "2001:db8:1234::/64"},
			want:  []string{"10.10.0.0/24", "2001:db8::/64"},
		}, {
			desc:  "ipv6 ipv4 ipv6",
			cidrs: []string{"2001:db8::/64", "10.10.0.0/24", "2001:db8:1234::/64"},
			want:  []string{"2001:db8::/64", "10.10.0.0/24"},
		},
	} {

		t.Run(tc.desc, func(t *testing.T) {
			got := filterCIDRs(tc.cidrs)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got: %v, want: %v", got, tc.want)
			}
		})
	}

}
