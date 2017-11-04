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

package sync

import (
	"context"
	"fmt"
	"net"
	"reflect"
	"testing"
	"time"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/controller/node/ipam/cidrset"
	"k8s.io/kubernetes/pkg/controller/node/ipam/test"

	"k8s.io/api/core/v1"
)

var (
	_, clusterCIDRRange, _ = net.ParseCIDR("10.1.0.0/16")
)

type fakeEvent struct {
	nodeName string
	reason   string
}

type fakeAPIs struct {
	aliasRange    *net.IPNet
	aliasErr      error
	addAliasErr   error
	nodeRet       *v1.Node
	nodeErr       error
	updateNodeErr error
	resyncTimeout time.Duration
	reportChan    chan struct{}

	updateNodeNetworkUnavailableErr error

	calls   []string
	events  []fakeEvent
	results []error
}

func (f *fakeAPIs) Alias(ctx context.Context, nodeName string) (*net.IPNet, error) {
	f.calls = append(f.calls, fmt.Sprintf("alias %v", nodeName))
	return f.aliasRange, f.aliasErr
}

func (f *fakeAPIs) AddAlias(ctx context.Context, nodeName string, cidrRange *net.IPNet) error {
	f.calls = append(f.calls, fmt.Sprintf("addAlias %v %v", nodeName, cidrRange))
	return f.addAliasErr
}

func (f *fakeAPIs) Node(ctx context.Context, name string) (*v1.Node, error) {
	f.calls = append(f.calls, fmt.Sprintf("node %v", name))
	return f.nodeRet, f.nodeErr
}

func (f *fakeAPIs) UpdateNodePodCIDR(ctx context.Context, node *v1.Node, cidrRange *net.IPNet) error {
	f.calls = append(f.calls, fmt.Sprintf("updateNode %v", node))
	return f.updateNodeErr
}

func (f *fakeAPIs) UpdateNodeNetworkUnavailable(nodeName string, unavailable bool) error {
	f.calls = append(f.calls, fmt.Sprintf("updateNodeNetworkUnavailable %v %v", nodeName, unavailable))
	return f.updateNodeNetworkUnavailableErr
}

func (f *fakeAPIs) EmitNodeWarningEvent(nodeName, reason, fmtStr string, args ...interface{}) {
	f.events = append(f.events, fakeEvent{nodeName, reason})
}

func (f *fakeAPIs) ReportResult(err error) {
	glog.V(2).Infof("ReportResult %v", err)
	f.results = append(f.results, err)
	if f.reportChan != nil {
		f.reportChan <- struct{}{}
	}
}

func (f *fakeAPIs) ResyncTimeout() time.Duration {
	if f.resyncTimeout == 0 {
		return time.Second * 10000
	}
	return f.resyncTimeout
}

func (f *fakeAPIs) dumpTrace() {
	for i, x := range f.calls {
		glog.Infof("trace %v: %v", i, x)
	}
}

var nodeWithoutCIDRRange = &v1.Node{
	ObjectMeta: metav1.ObjectMeta{Name: "node1"},
}

var nodeWithCIDRRange = &v1.Node{
	ObjectMeta: metav1.ObjectMeta{Name: "node1"},
	Spec:       v1.NodeSpec{PodCIDR: "10.1.1.0/24"},
}

func TestNodeSyncUpdate(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		desc string
		mode NodeSyncMode
		node *v1.Node
		fake fakeAPIs

		events    []fakeEvent
		wantError bool
	}{
		{
			desc: "validate range ==",
			mode: SyncFromCloud,
			node: nodeWithCIDRRange,
			fake: fakeAPIs{
				aliasRange: test.MustParseCIDR(nodeWithCIDRRange.Spec.PodCIDR),
			},
		},
		{
			desc:   "validate range !=",
			mode:   SyncFromCloud,
			node:   nodeWithCIDRRange,
			fake:   fakeAPIs{aliasRange: test.MustParseCIDR("192.168.0.0/24")},
			events: []fakeEvent{{"node1", "CloudCIDRAllocatorMismatch"}},
		},
		{
			desc:      "update alias from node",
			mode:      SyncFromCloud,
			node:      nodeWithCIDRRange,
			events:    []fakeEvent{{"node1", "CloudCIDRAllocatorInvalidMode"}},
			wantError: true,
		},
		{
			desc: "update alias from node",
			mode: SyncFromCluster,
			node: nodeWithCIDRRange,
			// XXX/bowei -- validation
		},
		{
			desc: "update node from alias",
			mode: SyncFromCloud,
			node: nodeWithoutCIDRRange,
			fake: fakeAPIs{aliasRange: test.MustParseCIDR("10.1.2.3/16")},
			// XXX/bowei -- validation
		},
		{
			desc:      "update node from alias",
			mode:      SyncFromCluster,
			node:      nodeWithoutCIDRRange,
			fake:      fakeAPIs{aliasRange: test.MustParseCIDR("10.1.2.3/16")},
			events:    []fakeEvent{{"node1", "CloudCIDRAllocatorInvalidMode"}},
			wantError: true,
		},
		{
			desc:      "allocate range",
			mode:      SyncFromCloud,
			node:      nodeWithoutCIDRRange,
			events:    []fakeEvent{{"node1", "CloudCIDRAllocatorInvalidMode"}},
			wantError: true,
		},
		{
			desc: "allocate range",
			mode: SyncFromCluster,
			node: nodeWithoutCIDRRange,
		},
		{
			desc: "update with node==nil",
			mode: SyncFromCluster,
			node: nil,
			fake: fakeAPIs{
				nodeRet: nodeWithCIDRRange,
			},
			wantError: false,
		},
	} {
		sync := New(&tc.fake, &tc.fake, &tc.fake, tc.mode, "node1", cidrset.NewCIDRSet(clusterCIDRRange, 24))
		doneChan := make(chan struct{})

		// Do a single step of the loop.
		go sync.Loop(doneChan)
		sync.Update(tc.node)
		close(sync.opChan)
		<-doneChan
		tc.fake.dumpTrace()

		if !reflect.DeepEqual(tc.fake.events, tc.events) {
			t.Errorf("%v, %v; fake.events = %#v, want %#v", tc.desc, tc.mode, tc.fake.events, tc.events)
		}

		var hasError bool
		for _, r := range tc.fake.results {
			hasError = hasError || (r != nil)
		}
		if hasError != tc.wantError {
			t.Errorf("%v, %v; hasError = %t, errors = %v, want %t",
				tc.desc, tc.mode, hasError, tc.fake.events, tc.wantError)
		}
	}
}

func TestNodeSyncResync(t *testing.T) {
	fake := &fakeAPIs{
		nodeRet:       nodeWithCIDRRange,
		resyncTimeout: time.Millisecond,
		reportChan:    make(chan struct{}),
	}
	sync := New(fake, fake, fake, SyncFromCluster, "node1", cidrset.NewCIDRSet(clusterCIDRRange, 24))
	doneChan := make(chan struct{})

	go sync.Loop(doneChan)
	<-fake.reportChan
	close(sync.opChan)
	// Unblock loop().
	go func() {
		<-fake.reportChan
	}()
	<-doneChan
	fake.dumpTrace()
}

func TestNodeSyncDelete(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		desc string
		mode NodeSyncMode
		node *v1.Node
		fake fakeAPIs
	}{
		{
			desc: "delete",
			mode: SyncFromCluster,
			node: nodeWithCIDRRange,
		},
		{
			desc: "delete without CIDR range",
			mode: SyncFromCluster,
			node: nodeWithoutCIDRRange,
		},
		{
			desc: "delete with invalid CIDR range",
			mode: SyncFromCluster,
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec:       v1.NodeSpec{PodCIDR: "invalid"},
			},
		},
	} {
		sync := New(&tc.fake, &tc.fake, &tc.fake, tc.mode, "node1", cidrset.NewCIDRSet(clusterCIDRRange, 24))
		doneChan := make(chan struct{})

		// Do a single step of the loop.
		go sync.Loop(doneChan)
		sync.Delete(tc.node)
		<-doneChan
		tc.fake.dumpTrace()

		/*
			if !reflect.DeepEqual(tc.fake.events, tc.events) {
				t.Errorf("%v, %v; fake.events = %#v, want %#v", tc.desc, tc.mode, tc.fake.events, tc.events)
			}

			var hasError bool
			for _, r := range tc.fake.results {
				hasError = hasError || (r != nil)
			}
			if hasError != tc.wantError {
				t.Errorf("%v, %v; hasError = %t, errors = %v, want %t",
					tc.desc, tc.mode, hasError, tc.fake.events, tc.wantError)
			}
		*/
	}
}
