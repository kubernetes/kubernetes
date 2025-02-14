/*
Copyright 2024 The Kubernetes Authors.

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

package plugin

import (
	"context"
	"fmt"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/timestamppb"

	"k8s.io/apimachinery/pkg/util/wait"
	externaljwtv1alpha1 "k8s.io/externaljwt/apis/v1alpha1"
	"k8s.io/kubernetes/pkg/serviceaccount"

	utilnettesting "k8s.io/apimachinery/pkg/util/net/testing"
)

func TestExternalPublicKeyGetter(t *testing.T) {

	invalidKid := string(make([]byte, 1025))
	testCases := []struct {
		desc                  string
		expectedErr           error
		supportedKeys         map[string]supportedKeyT
		wantVerificationKeys  *VerificationKeys
		refreshHintSec        int
		dataTimeStamp         *timestamppb.Timestamp
		supportedKeysOverride []*externaljwtv1alpha1.Key
	}{
		{
			desc: "single key in signer",
			supportedKeys: map[string]supportedKeyT{
				"key-1": {
					key: &rsaKey1.PublicKey,
				},
			},
			wantVerificationKeys: &VerificationKeys{
				Keys: []serviceaccount.PublicKey{
					{
						KeyID:                    "key-1",
						PublicKey:                &rsaKey1.PublicKey,
						ExcludeFromOIDCDiscovery: false,
					},
				},
			},
			refreshHintSec: 20,
			dataTimeStamp:  timestamppb.New(time.Time{}),
		},
		{
			desc: "multiple keys in signer",
			supportedKeys: map[string]supportedKeyT{
				"key-1": {
					key: &rsaKey1.PublicKey,
				},
				"key-2": {
					key:             &rsaKey2.PublicKey,
					excludeFromOidc: true,
				},
			},
			wantVerificationKeys: &VerificationKeys{
				Keys: []serviceaccount.PublicKey{
					{
						KeyID:                    "key-1",
						PublicKey:                &rsaKey1.PublicKey,
						ExcludeFromOIDCDiscovery: false,
					},
					{
						KeyID:                    "key-2",
						PublicKey:                &rsaKey2.PublicKey,
						ExcludeFromOIDCDiscovery: true,
					},
				},
			},
			refreshHintSec: 10,
			dataTimeStamp:  timestamppb.New(time.Time{}),
		},
		{
			desc: "empty kid",
			supportedKeys: map[string]supportedKeyT{
				"": {
					key: &rsaKey1.PublicKey,
				},
				"key-2": {
					key:             &rsaKey2.PublicKey,
					excludeFromOidc: true,
				},
			},
			expectedErr:    fmt.Errorf("found invalid public key id %q", ""),
			refreshHintSec: 10,
			dataTimeStamp:  timestamppb.New(time.Time{}),
		},
		{
			desc: "kid longer than 1024",
			supportedKeys: map[string]supportedKeyT{
				invalidKid: {
					key: &rsaKey1.PublicKey,
				},
				"key-2": {
					key:             &rsaKey2.PublicKey,
					excludeFromOidc: true,
				},
			},
			expectedErr:    fmt.Errorf("found invalid public key id %q", invalidKid),
			refreshHintSec: 10,
			dataTimeStamp:  timestamppb.New(time.Time{}),
		},
		{
			desc:           "no keys",
			supportedKeys:  map[string]supportedKeyT{},
			expectedErr:    fmt.Errorf("found no keys"),
			refreshHintSec: 10,
			dataTimeStamp:  timestamppb.New(time.Time{}),
		},
		{
			desc: "invalid data timestamp",
			supportedKeys: map[string]supportedKeyT{
				"key-2": {
					key:             &rsaKey2.PublicKey,
					excludeFromOidc: true,
				},
			},
			expectedErr:    fmt.Errorf("invalid data timestamp"),
			refreshHintSec: 10,
			dataTimeStamp:  nil,
		},
		{
			desc:           "empty public key",
			expectedErr:    fmt.Errorf("found empty public key"),
			refreshHintSec: 10,
			dataTimeStamp:  timestamppb.New(time.Time{}),
			supportedKeys: map[string]supportedKeyT{
				"key-2": {
					key:             &rsaKey2.PublicKey,
					excludeFromOidc: true,
				},
			},
			supportedKeysOverride: []*externaljwtv1alpha1.Key{
				{
					KeyId: "kid",
					Key:   nil,
				},
			},
		},
	}

	for i, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx := context.Background()

			sockname := utilnettesting.MakeSocketNameForTest(t, fmt.Sprintf("test-external-public-key-getter-%d-%d.sock", time.Now().Nanosecond(), i))

			addr := &net.UnixAddr{Name: sockname, Net: "unix"}
			listener, err := net.ListenUnix(addr.Network(), addr)
			if err != nil {
				t.Fatalf("Failed to start fake backend: %v", err)
			}

			grpcServer := grpc.NewServer()

			backend := &dummyExtrnalSigner{
				supportedKeys:      tc.supportedKeys,
				refreshHintSeconds: tc.refreshHintSec,
			}
			backend.DataTimeStamp = tc.dataTimeStamp
			backend.SupportedKeysOverride = tc.supportedKeysOverride
			externaljwtv1alpha1.RegisterExternalJWTSignerServer(grpcServer, backend)

			defer grpcServer.Stop()
			go func() {
				if err := grpcServer.Serve(listener); err != nil {
					panic(fmt.Errorf("error returned from grpcServer: %w", err))
				}
			}()

			clientConn, err := grpc.DialContext(
				ctx,
				sockname,
				grpc.WithContextDialer(func(ctx context.Context, path string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "unix", path)
				}),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("Failed to dial buffconn client: %v", err)
			}
			defer func() {
				_ = clientConn.Close()
			}()

			plugin := newPlugin("iss", clientConn, true)

			signingKeys, err := plugin.keyCache.getTokenVerificationKeys(ctx)
			if err != nil {
				if tc.expectedErr == nil {
					t.Fatalf("error getting supported keys: %v", err)
				}
				if !strings.Contains(err.Error(), tc.expectedErr.Error()) {
					t.Fatalf("want error: %v, got error: %v", tc.expectedErr, err)
					return
				}
			}

			if tc.expectedErr == nil {
				if diff := cmp.Diff(signingKeys.Keys, tc.wantVerificationKeys.Keys, cmpopts.SortSlices(sortPublicKeySlice)); diff != "" {
					t.Fatalf("Bad result from GetTokenSigningKeys; diff (-got +want)\n%s", diff)
				}
				expectedRefreshHintSec := time.Now().Add(time.Duration(tc.refreshHintSec) * time.Second)
				difference := signingKeys.NextRefreshHint.Sub(expectedRefreshHintSec).Seconds()
				if difference > 1 || difference < -1 { // tolerate 1 sec of skew for test
					t.Fatalf("refreshHint not as expected; got: %v want: %v", signingKeys.NextRefreshHint, expectedRefreshHintSec)
				}
			}
		})
	}
}

func TestInitialFill(t *testing.T) {
	ctx := context.Background()

	sockname := utilnettesting.MakeSocketNameForTest(t, fmt.Sprintf("test-initial-fill-%d.sock", time.Now().Nanosecond()))

	addr := &net.UnixAddr{Name: sockname, Net: "unix"}
	listener, err := net.ListenUnix(addr.Network(), addr)
	if err != nil {
		t.Fatalf("Failed to start fake backend: %v", err)
	}

	grpcServer := grpc.NewServer()

	supportedKeys := map[string]supportedKeyT{
		"key-1": {
			key: &rsaKey1.PublicKey,
		},
	}
	wantPubKeys := []serviceaccount.PublicKey{
		{
			KeyID:                    "key-1",
			PublicKey:                &rsaKey1.PublicKey,
			ExcludeFromOIDCDiscovery: false,
		},
	}

	backend := &dummyExtrnalSigner{
		supportedKeys:      supportedKeys,
		refreshHintSeconds: 10,
		DataTimeStamp:      timestamppb.New(time.Time{}),
	}
	externaljwtv1alpha1.RegisterExternalJWTSignerServer(grpcServer, backend)

	defer grpcServer.Stop()
	go func() {
		if err := grpcServer.Serve(listener); err != nil {
			panic(fmt.Errorf("error returned from grpcServer: %w", err))
		}
	}()

	clientConn, err := grpc.DialContext(
		ctx,
		sockname,
		grpc.WithContextDialer(func(ctx context.Context, path string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial buffconn client: %v", err)
	}
	defer func() { _ = clientConn.Close() }()

	plugin := newPlugin("iss", clientConn, true)

	if err := plugin.keyCache.initialFill(ctx); err != nil {
		t.Fatalf("Error during InitialFill: %v", err)
	}

	gotPubKeys := plugin.keyCache.GetPublicKeys(ctx, "")
	if diff := cmp.Diff(gotPubKeys, wantPubKeys); diff != "" {
		t.Fatalf("Bad public keys; diff (-got +want)\n%s", diff)
	}
}

func TestReflectChanges(t *testing.T) {
	ctx := context.Background()

	sockname := utilnettesting.MakeSocketNameForTest(t, fmt.Sprintf("test-reflect-changes-%d.sock", time.Now().Nanosecond()))

	addr := &net.UnixAddr{Name: sockname, Net: "unix"}
	listener, err := net.ListenUnix(addr.Network(), addr)
	if err != nil {
		t.Fatalf("Failed to start fake backend: %v", err)
	}

	grpcServer := grpc.NewServer()

	supportedKeysT1 := map[string]supportedKeyT{
		"key-1": {
			key: &rsaKey1.PublicKey,
		},
	}
	wantPubKeysT1 := []serviceaccount.PublicKey{
		{
			KeyID:                    "key-1",
			PublicKey:                &rsaKey1.PublicKey,
			ExcludeFromOIDCDiscovery: false,
		},
	}

	backend := &dummyExtrnalSigner{
		supportedKeys:      supportedKeysT1,
		refreshHintSeconds: 10,
		DataTimeStamp:      timestamppb.New(time.Time{}),
	}
	externaljwtv1alpha1.RegisterExternalJWTSignerServer(grpcServer, backend)

	defer grpcServer.Stop()
	go func() {
		if err := grpcServer.Serve(listener); err != nil {
			panic(fmt.Errorf("error returned from grpcServer: %w", err))
		}
	}()

	clientConn, err := grpc.DialContext(
		ctx,
		sockname,
		grpc.WithContextDialer(func(ctx context.Context, path string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial buffconn client: %v", err)
	}
	defer func() { _ = clientConn.Close() }()

	plugin := newPlugin("iss", clientConn, true)

	dummyListener := &dummyListener{}
	plugin.keyCache.AddListener(dummyListener)

	dummyListener.waitForCount(t, 0)
	if err := plugin.keyCache.initialFill(ctx); err != nil {
		t.Fatalf("Error during InitialFill: %v", err)
	}
	dummyListener.waitForCount(t, 1)

	gotPubKeysT1 := plugin.keyCache.GetPublicKeys(ctx, "")
	if diff := cmp.Diff(gotPubKeysT1, wantPubKeysT1, cmpopts.SortSlices(sortPublicKeySlice)); diff != "" {
		t.Fatalf("Bad public keys; diff (-got +want)\n%s", diff)
	}

	dummyListener.waitForCount(t, 1)
	if err := plugin.keyCache.syncKeys(ctx); err != nil {
		t.Fatalf("Error while calling syncKeys: %v", err)
	}
	dummyListener.waitForCount(t, 1)

	supportedKeysT2 := map[string]supportedKeyT{
		"key-1": {
			key:             &rsaKey1.PublicKey,
			excludeFromOidc: true,
		},
		"key-2": {
			key: &rsaKey2.PublicKey,
		},
	}
	wantPubKeysT2 := []serviceaccount.PublicKey{
		{
			KeyID:                    "key-1",
			PublicKey:                &rsaKey1.PublicKey,
			ExcludeFromOIDCDiscovery: true,
		},
		{
			KeyID:                    "key-2",
			PublicKey:                &rsaKey2.PublicKey,
			ExcludeFromOIDCDiscovery: false,
		},
	}

	backend.keyLock.Lock()
	backend.supportedKeys = supportedKeysT2
	backend.keyLock.Unlock()

	dummyListener.waitForCount(t, 1)
	if err := plugin.keyCache.syncKeys(ctx); err != nil {
		t.Fatalf("Error while calling syncKeys: %v", err)
	}
	dummyListener.waitForCount(t, 2)

	gotPubKeysT2 := plugin.keyCache.GetPublicKeys(ctx, "")
	if diff := cmp.Diff(gotPubKeysT2, wantPubKeysT2, cmpopts.SortSlices(sortPublicKeySlice)); diff != "" {
		t.Fatalf("Bad public keys; diff (-got +want)\n%s", diff)
	}
	dummyListener.waitForCount(t, 2)
}

type dummyListener struct {
	count atomic.Int64
}

func (d *dummyListener) waitForCount(t *testing.T, expect int) {
	t.Helper()
	err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 10*time.Second, true, func(_ context.Context) (bool, error) {
		actual := int(d.count.Load())
		switch {
		case actual > expect:
			return false, fmt.Errorf("expected %d broadcasts, got %d broadcasts", expect, actual)
		case actual == expect:
			return true, nil
		default:
			t.Logf("expected %d broadcasts, got %d broadcasts, waiting...", expect, actual)
			return false, nil
		}
	})
	if err != nil {
		t.Fatal(err)
	}
}

func (d *dummyListener) Enqueue() {
	d.count.Add(1)
}

func TestKeysChanged(t *testing.T) {
	testcases := []struct {
		name    string
		oldKeys VerificationKeys
		newKeys VerificationKeys
		expect  bool
	}{
		{
			name:    "empty",
			oldKeys: VerificationKeys{},
			newKeys: VerificationKeys{},
			expect:  false,
		},
		{
			name:    "identical",
			oldKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}, {KeyID: "b"}}},
			newKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}, {KeyID: "b"}}},
			expect:  false,
		},
		{
			name:    "changed datatimestamp",
			oldKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}, {KeyID: "b"}}},
			newKeys: VerificationKeys{DataTimestamp: time.Unix(1001, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}, {KeyID: "b"}}},
			expect:  true,
		},
		{
			name:    "reordered keyid",
			oldKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}, {KeyID: "b"}}},
			newKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "b"}, {KeyID: "a"}}},
			expect:  true,
		},
		{
			name:    "changed keyid",
			oldKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}}},
			newKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "b"}}},
			expect:  true,
		},
		{
			name:    "added key",
			oldKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}}},
			newKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}, {KeyID: "b"}}},
			expect:  true,
		},
		{
			name:    "removed key",
			oldKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}, {KeyID: "b"}}},
			newKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a"}}},
			expect:  true,
		},
		{
			name:    "changed oidc",
			oldKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a", ExcludeFromOIDCDiscovery: false}}},
			newKeys: VerificationKeys{DataTimestamp: time.Unix(1000, 0), Keys: []serviceaccount.PublicKey{{KeyID: "a", ExcludeFromOIDCDiscovery: true}}},
			expect:  true,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			result := keysChanged(&tc.oldKeys, &tc.newKeys)
			if result != tc.expect {
				t.Errorf("got %v, expected %v", result, tc.expect)
			}
		})
	}
}
