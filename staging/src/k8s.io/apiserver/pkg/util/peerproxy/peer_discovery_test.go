/*
Copyright 2025 The Kubernetes Authors.

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

package peerproxy

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/transport"

	v1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestAddPeerDiscoveryInfo(t *testing.T) {
	localServerID := "local-server"
	testCases := []struct {
		desc                string
		lease               *v1.Lease
		reconcilerEndpoints map[string]string
		existingCache       map[string]*peerAggDiscoveryInfo
		wantCache           map[string]*peerAggDiscoveryInfo
	}{
		{
			desc:      "nil leaser",
			lease:     nil,
			wantCache: map[string]*peerAggDiscoveryInfo{},
		},
		{
			desc: "valid lease, nil holderIdentity",
			lease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{},
		},
		{
			desc: "valid local apiserver lease",
			lease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: localServerID},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{},
		},
		{
			desc: "valid lease, new server",
			lease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			reconcilerEndpoints: map[string]string{
				"remote-1": "127.0.0.1:6443",
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				// empty servedResources since rerouted discovery call is served with
				// 503 in the test setup.
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
		{
			desc: "valid lease, existing server, different holderIdentity",
			lease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-2")},
			},
			reconcilerEndpoints: map[string]string{
				"remote-1": "127.0.0.1:6443",
				"remote-2": "127.0.0.1:6445",
			},
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-2", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			fakeReconciler := newFakeReconciler()
			for serverID, endpoint := range tt.reconcilerEndpoints {
				fakeReconciler.setEndpoint(serverID, endpoint)
			}

			h := &peerProxyHandler{
				serverId:                      localServerID,
				peerAggDiscoveryResponseCache: tt.existingCache,
				discoverySerializer:           serializer.NewCodecFactory(runtime.NewScheme()),
				reconciler:                    fakeReconciler,
				proxyClientConfig:             &transport.Config{},
			}
			if tt.existingCache == nil {
				h.peerAggDiscoveryResponseCache = make(map[string]*peerAggDiscoveryInfo)
			}

			h.addPeerDiscoveryInfo(tt.lease)
			assert.Equal(t, tt.wantCache, h.peerAggDiscoveryResponseCache)

		})
	}
}

func TestDeletePeerDiscoveryInfo(t *testing.T) {
	testCases := []struct {
		desc          string
		lease         *v1.Lease
		existingCache map[string]*peerAggDiscoveryInfo
		wantCache     map[string]*peerAggDiscoveryInfo
	}{
		{
			desc: "valid lease, existing server",
			lease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
			},
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{},
		},
		{
			desc:  "nil lease",
			lease: nil,
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
		{
			desc: "valid lease, non-existing server",
			lease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-2"},
			},
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			h := &peerProxyHandler{
				serverId:                      "local-server",
				peerAggDiscoveryResponseCache: tt.existingCache,
			}

			h.deletePeerDiscoveryInfo(tt.lease)

			assert.Equal(t, tt.wantCache, h.peerAggDiscoveryResponseCache)
		})
	}
}

func TestUpdatePeerDiscoveryInfo(t *testing.T) {
	localServerID := "local-server"
	testCases := []struct {
		desc                string
		oldLease            *v1.Lease
		newLease            *v1.Lease
		reconcilerEndpoints map[string]string
		existingCache       map[string]*peerAggDiscoveryInfo
		wantCache           map[string]*peerAggDiscoveryInfo
	}{
		{
			desc:      "nil old and new lease",
			oldLease:  nil,
			newLease:  nil,
			wantCache: map[string]*peerAggDiscoveryInfo{},
		},
		{
			desc:     "nil old lease, valid new lease",
			oldLease: nil,
			newLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			reconcilerEndpoints: map[string]string{
				"remote-1": "127.0.0.1:6443",
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
		{
			desc: "valid old lease, nil new lease",
			oldLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			newLease: nil,
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{},
		},
		{
			desc: "local server lease update, no change",
			oldLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: localServerID},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			newLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: localServerID},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{},
		},
		{
			desc: "valid old lease, valid new lease, same holderIdentity",
			oldLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			newLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
		{
			desc: "valid old lease, valid new lease, different holderIdentity",
			oldLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			newLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-2")},
			},
			reconcilerEndpoints: map[string]string{
				"remote-1": "127.0.0.1:6443",
			},
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-2", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
		{
			desc: "valid old lease, valid new lease, different name",
			oldLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-1"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-1")},
			},
			newLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{Name: "remote-2"},
				Spec:       v1.LeaseSpec{HolderIdentity: proto.String("holder-2")},
			},
			reconcilerEndpoints: map[string]string{
				"remote-2": "127.0.0.1:6443",
			},
			existingCache: map[string]*peerAggDiscoveryInfo{
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
			wantCache: map[string]*peerAggDiscoveryInfo{
				"remote-2": {holderIdentity: "holder-2", servedResources: map[schema.GroupVersion][]string{}},
				"remote-1": {holderIdentity: "holder-1", servedResources: map[schema.GroupVersion][]string{}},
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			fakeReconciler := newFakeReconciler()
			for serverID, endpoint := range tt.reconcilerEndpoints {
				fakeReconciler.setEndpoint(serverID, endpoint)
			}

			h := &peerProxyHandler{
				serverId:                      localServerID,
				peerAggDiscoveryResponseCache: tt.existingCache,
				discoverySerializer:           serializer.NewCodecFactory(runtime.NewScheme()),
				reconciler:                    fakeReconciler,
				proxyClientConfig:             &transport.Config{},
			}
			if tt.existingCache == nil {
				h.peerAggDiscoveryResponseCache = make(map[string]*peerAggDiscoveryInfo)
			}

			h.updatePeerDiscoveryInfo(tt.oldLease, tt.newLease)
			assert.Equal(t, tt.wantCache, h.peerAggDiscoveryResponseCache)
		})
	}
}

type fakeReconciler struct {
	endpoints map[string]string
}

func newFakeReconciler() *fakeReconciler {
	return &fakeReconciler{
		endpoints: make(map[string]string),
	}
}

func (f *fakeReconciler) UpdateLease(serverID string, publicIP string, ports []corev1.EndpointPort) error {
	return nil
}

func (f *fakeReconciler) DeleteLease(serverID string) error {
	return nil
}

func (f *fakeReconciler) Destroy() {
}

func (f *fakeReconciler) GetEndpoint(serverID string) (string, error) {
	endpoint, ok := f.endpoints[serverID]
	if !ok {
		return "", fmt.Errorf("endpoint not found for serverID: %s", serverID)
	}
	return endpoint, nil
}

func (f *fakeReconciler) RemoveLease(serverID string) error {
	return nil
}

func (f *fakeReconciler) StopReconciling() {
}

func (f *fakeReconciler) setEndpoint(serverID, endpoint string) {
	f.endpoints[serverID] = endpoint
}
