/*
Copyright 2019 The Kubernetes Authors.

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

package metaproxier

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/proxy"
)

// recordingProxier implements proxy.Provider and records each invocation so tests
// can assert dual-stack dispatch without a full proxier implementation.
type recordingProxier struct {
	mu sync.Mutex

	calls []string

	// syncLoopExit, when non-nil, causes SyncLoop to block until the channel is closed.
	syncLoopExit <-chan struct{}
	// onSyncLoopEnter, when non-nil, is invoked after recording SyncLoop and before blocking.
	onSyncLoopEnter func()
}

func (r *recordingProxier) record(method string, args ...string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, fmt.Sprintf("%s(%s)", method, strings.Join(args, ", ")))
}

func (r *recordingProxier) getCalls() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.calls))
	copy(out, r.calls)
	r.calls = nil
	return out
}

func (r *recordingProxier) Sync() {
	r.record("Sync")
}

func (r *recordingProxier) SyncLoop() {
	r.record("SyncLoop")
	if r.onSyncLoopEnter != nil {
		r.onSyncLoopEnter()
	}
	if r.syncLoopExit != nil {
		<-r.syncLoopExit
	}
}

func (r *recordingProxier) OnServiceAdd(service *v1.Service) {
	r.record("OnServiceAdd", service.Name)
}

func (r *recordingProxier) OnServiceUpdate(oldSvc, service *v1.Service) {
	r.record("OnServiceUpdate", oldSvc.Name, service.Name)
}

func (r *recordingProxier) OnServiceDelete(service *v1.Service) {
	r.record("OnServiceDelete", service.Name)
}

func (r *recordingProxier) OnServiceSynced() {
	r.record("OnServiceSynced")
}

func (r *recordingProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	r.record("OnEndpointSliceAdd", string(endpointSlice.AddressType))
}

func (r *recordingProxier) OnEndpointSliceUpdate(oldEs, newEs *discovery.EndpointSlice) {
	r.record("OnEndpointSliceUpdate", string(oldEs.AddressType), string(newEs.AddressType))
}

func (r *recordingProxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	r.record("OnEndpointSliceDelete", string(endpointSlice.AddressType))
}

func (r *recordingProxier) OnEndpointSlicesSynced() {
	r.record("OnEndpointSlicesSynced")
}

func (r *recordingProxier) OnTopologyChange(labels map[string]string) {
	r.record("OnTopologyChange")
}

func (r *recordingProxier) OnServiceCIDRsChanged(cidrs []string) {
	r.record("OnServiceCIDRsChanged", strings.Join(cidrs, ", "))
}

var _ proxy.Provider = (*recordingProxier)(nil)

func TestMetaProxier_GenericCallbacksFanOutToBothProxiers(t *testing.T) {
	ipv4 := &recordingProxier{}
	ipv6 := &recordingProxier{}
	mp := NewMetaProxier(ipv4, ipv6)

	mp.Sync()

	svcAdd := &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "add"}}
	mp.OnServiceAdd(svcAdd)

	oldSvc := &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "old"}}
	newSvc := &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "new"}}
	mp.OnServiceUpdate(oldSvc, newSvc)

	del := &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "gone"}}
	mp.OnServiceDelete(del)

	mp.OnServiceSynced()

	mp.OnEndpointSlicesSynced()

	mp.OnTopologyChange(map[string]string{"topology.kubernetes.io/zone": "a"})
	mp.OnServiceCIDRsChanged([]string{"10.0.0.0/16", "2001:db8::/64"})

	want := []string{
		"Sync()",
		"OnServiceAdd(add)",
		"OnServiceUpdate(old, new)",
		"OnServiceDelete(gone)",
		"OnServiceSynced()",
		"OnEndpointSlicesSynced()",
		"OnTopologyChange()",
		"OnServiceCIDRsChanged(10.0.0.0/16, 2001:db8::/64)",
	}
	if diff := cmp.Diff(want, ipv4.getCalls()); diff != "" {
		t.Errorf("ipv4 (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, ipv6.getCalls()); diff != "" {
		t.Errorf("ipv6 (-want +got):\n%s", diff)
	}
}

func TestMetaProxier_EndpointSlice_RoutesByAddressType(t *testing.T) {
	ipv4 := &recordingProxier{}
	ipv6 := &recordingProxier{}
	mp := NewMetaProxier(ipv4, ipv6)

	eps4 := &discovery.EndpointSlice{AddressType: discovery.AddressTypeIPv4}
	eps6 := &discovery.EndpointSlice{AddressType: discovery.AddressTypeIPv6}

	mp.OnEndpointSliceAdd(eps4)
	mp.OnEndpointSliceAdd(eps6)

	old4 := &discovery.EndpointSlice{AddressType: discovery.AddressTypeIPv4}
	new4 := &discovery.EndpointSlice{AddressType: discovery.AddressTypeIPv4}
	mp.OnEndpointSliceUpdate(old4, new4)

	mp.OnEndpointSliceDelete(eps4)

	if diff := cmp.Diff([]string{
		"OnEndpointSliceAdd(IPv4)",
		"OnEndpointSliceUpdate(IPv4, IPv4)",
		"OnEndpointSliceDelete(IPv4)",
	}, ipv4.getCalls()); diff != "" {
		t.Errorf("ipv4 (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{
		"OnEndpointSliceAdd(IPv6)",
	}, ipv6.getCalls()); diff != "" {
		t.Errorf("ipv6 (-want +got):\n%s", diff)
	}
}

func TestMetaProxier_EndpointSlice_UnsupportedAddressType_NoDownstreamCalls(t *testing.T) {
	ipv4 := &recordingProxier{}
	ipv6 := &recordingProxier{}
	mp := NewMetaProxier(ipv4, ipv6)

	fqdnSlice := &discovery.EndpointSlice{AddressType: discovery.AddressTypeFQDN}

	mp.OnEndpointSliceAdd(fqdnSlice)
	mp.OnEndpointSliceUpdate(fqdnSlice, fqdnSlice)
	mp.OnEndpointSliceDelete(fqdnSlice)

	if got := ipv4.getCalls(); len(got) != 0 {
		t.Errorf("expected ipv4 no calls for FQDN slices, got %v", got)
	}
	if got := ipv6.getCalls(); len(got) != 0 {
		t.Errorf("expected ipv6 no calls for FQDN slices, got %v", got)
	}
}

func TestMetaProxier_SyncLoop_InvokesBothChildSyncLoops(t *testing.T) {
	exit := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)

	ipv4 := &recordingProxier{
		syncLoopExit:    exit,
		onSyncLoopEnter: wg.Done,
	}
	ipv6 := &recordingProxier{
		syncLoopExit:    exit,
		onSyncLoopEnter: wg.Done,
	}
	mp := NewMetaProxier(ipv4, ipv6)

	stopped := make(chan struct{})
	go func() {
		mp.SyncLoop()
		close(stopped)
	}()

	bothStarted := make(chan struct{})
	go func() {
		wg.Wait()
		close(bothStarted)
	}()

	select {
	case <-bothStarted:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timed out waiting for both child SyncLoop calls: ipv4=%v ipv6=%v", ipv4.getCalls(), ipv6.getCalls())
	}

	close(exit)

	select {
	case <-stopped:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("SyncLoop did not return after unblocking children")
	}

	if diff := cmp.Diff([]string{"SyncLoop()"}, ipv4.getCalls()); diff != "" {
		t.Errorf("ipv4 (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"SyncLoop()"}, ipv6.getCalls()); diff != "" {
		t.Errorf("ipv6 (-want +got):\n%s", diff)
	}
}
