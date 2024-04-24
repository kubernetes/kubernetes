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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	netutils "k8s.io/utils/net"
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
	_, ctx := ktesting.NewTestContext(t)
	go ca.worker(ctx)
	nodeName := "testNode"
	if err := ca.AllocateOrOccupyCIDR(ctx, &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
	}); err != nil {
		t.Errorf("unexpected error in AllocateOrOccupyCIDR: %v", err)
	}
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
		nodePodCIDR  string
		nodePodCIDRs []string
		want         bool
		wantErr      bool
	}{
		{
			desc:         "want error - invalid cidr",
			cidrs:        []string{"10.10.10.0/24"},
			nodePodCIDR:  "10.10..0/24",
			nodePodCIDRs: []string{"10.10..0/24"},
			want:         true,
		},
		{
			desc:         "want error - cidr len 2 but not dual stack",
			cidrs:        []string{"10.10.10.0/24", "10.10.11.0/24"},
			nodePodCIDR:  "10.10.10.0/24",
			nodePodCIDRs: []string{"10.10.10.0/24", "2001:db8::/64"},
			wantErr:      true,
		},
		{
			desc:         "want false - matching v4 only cidr",
			cidrs:        []string{"10.10.10.0/24"},
			nodePodCIDR:  "10.10.10.0/24",
			nodePodCIDRs: []string{"10.10.10.0/24"},
			want:         false,
		},
		{
			desc:  "want false - nil node.Spec.PodCIDR",
			cidrs: []string{"10.10.10.0/24"},
			want:  true,
		},
		{
			desc:         "want true - non matching v4 only cidr",
			cidrs:        []string{"10.10.10.0/24"},
			nodePodCIDR:  "10.10.11.0/24",
			nodePodCIDRs: []string{"10.10.11.0/24"},
			want:         true,
		},
		{
			desc:         "want false - matching v4 and v6 cidrs",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDR:  "10.10.10.0/24",
			nodePodCIDRs: []string{"10.10.10.0/24", "2001:db8::/64"},
			want:         false,
		},
		{
			desc:         "want false - matching v4 and v6 cidrs, different strings but same CIDRs",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDR:  "10.10.10.0/24",
			nodePodCIDRs: []string{"10.10.10.0/24", "2001:db8:0::/64"},
			want:         false,
		},
		{
			desc:         "want true - matching v4 and non matching v6 cidrs",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDR:  "10.10.10.0/24",
			nodePodCIDRs: []string{"10.10.10.0/24", "2001:dba::/64"},
			want:         true,
		},
		{
			desc:  "want true - nil node.Spec.PodCIDRs",
			cidrs: []string{"10.10.10.0/24", "2001:db8::/64"},
			want:  true,
		},
		{
			desc:         "want true - matching v6 and non matching v4 cidrs",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDR:  "10.10.1.0/24",
			nodePodCIDRs: []string{"10.10.1.0/24", "2001:db8::/64"},
			want:         true,
		},
		{
			desc:         "want true - missing v6",
			cidrs:        []string{"10.10.10.0/24", "2001:db8::/64"},
			nodePodCIDR:  "10.10.10.0/24",
			nodePodCIDRs: []string{"10.10.10.0/24"},
			want:         true,
		},
	} {
		var node v1.Node
		node.Spec.PodCIDR = tc.nodePodCIDR
		node.Spec.PodCIDRs = tc.nodePodCIDRs
		netCIDRs, err := netutils.ParseCIDRs(tc.cidrs)
		if err != nil {
			t.Errorf("failed to parse %v as CIDRs: %v", tc.cidrs, err)
		}
		logger, _ := ktesting.NewTestContext(t)
		t.Run(tc.desc, func(t *testing.T) {
			got, err := needPodCIDRsUpdate(logger, &node, netCIDRs)
			if tc.wantErr == (err == nil) {
				t.Errorf("err: %v, wantErr: %v", err, tc.wantErr)
			}
			if err == nil && got != tc.want {
				t.Errorf("got: %v, want: %v", got, tc.want)
			}
		})
	}
}
