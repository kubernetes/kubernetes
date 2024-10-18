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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	ejspb "k8s.io/externaljwt/apis/v1alpha1"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

func TestInitialFill(t *testing.T) {
	ctx := context.Background()

	sockname := fmt.Sprintf("@test-initial-fill.sock")

	addr := &net.UnixAddr{Name: sockname, Net: "unix"}
	listener, err := net.ListenUnix(addr.Network(), addr)
	if err != nil {
		t.Fatalf("Failed to start fake backend: %v", err)
	}

	grpcServer := grpc.NewServer()

	signingPrivateKeys := map[string]any{
		"key-1": rsaKey1,
	}
	wantPubKeys := []serviceaccount.PublicKey{
		{
			KeyID:     "key-1",
			PublicKey: &rsaKey1.PublicKey,
		},
	}

	backend := &dummyExtrnalSigner{
		signingPrivateKeys: signingPrivateKeys,
	}
	ejspb.RegisterExternalJWTSignerServer(grpcServer, backend)

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
	defer clientConn.Close()

	plugin := NewPlugin("iss", clientConn)

	keyCache := NewKeyCache(plugin)

	if err := keyCache.InitialFill(ctx); err != nil {
		t.Fatalf("Error during InitialFill: %v", err)
	}

	gotPubKeys := keyCache.GetPublicKeys("")
	if diff := cmp.Diff(gotPubKeys, wantPubKeys); diff != "" {
		t.Fatalf("Bad public keys; diff (-got +want)\n%s", diff)
	}
}

func TestReflectChanges(t *testing.T) {
	ctx := context.Background()

	sockname := fmt.Sprintf("@test-reflect-changes.sock")

	addr := &net.UnixAddr{Name: sockname, Net: "unix"}
	listener, err := net.ListenUnix(addr.Network(), addr)
	if err != nil {
		t.Fatalf("Failed to start fake backend: %v", err)
	}

	grpcServer := grpc.NewServer()

	signingPrivateKeysT1 := map[string]any{
		"key-1": rsaKey1,
	}
	wantPubKeysT1 := []serviceaccount.PublicKey{
		{
			KeyID:     "key-1",
			PublicKey: &rsaKey1.PublicKey,
		},
	}

	backend := &dummyExtrnalSigner{
		signingPrivateKeys: signingPrivateKeysT1,
	}
	ejspb.RegisterExternalJWTSignerServer(grpcServer, backend)

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
	defer clientConn.Close()

	plugin := NewPlugin("iss", clientConn)

	keyCache := NewKeyCache(plugin)

	if err := keyCache.InitialFill(ctx); err != nil {
		t.Fatalf("Error during InitialFill: %v", err)
	}

	gotPubKeysT1 := keyCache.GetPublicKeys("")
	if diff := cmp.Diff(gotPubKeysT1, wantPubKeysT1, cmpopts.SortSlices(sortPublicKeySlice)); diff != "" {
		t.Fatalf("Bad public keys; diff (-got +want)\n%s", diff)
	}

	if err := keyCache.syncKeys(ctx); err != nil {
		t.Fatalf("Error while calling syncKeys: %v", err)
	}

	signingPrivateKeysT2 := map[string]any{
		"key-1": rsaKey1,
		"key-2": rsaKey2,
	}
	wantPubKeysT2 := []serviceaccount.PublicKey{
		{
			KeyID:     "key-1",
			PublicKey: &rsaKey1.PublicKey,
		},
		{
			KeyID:     "key-2",
			PublicKey: &rsaKey2.PublicKey,
		},
	}

	backend.keyLock.Lock()
	backend.signingPrivateKeys = signingPrivateKeysT2
	backend.keyLock.Unlock()

	if err := keyCache.syncKeys(ctx); err != nil {
		t.Fatalf("Error while calling syncKeys: %v", err)
	}

	gotPubKeysT2 := keyCache.GetPublicKeys("")
	if diff := cmp.Diff(gotPubKeysT2, wantPubKeysT2, cmpopts.SortSlices(sortPublicKeySlice)); diff != "" {
		t.Fatalf("Bad public keys; diff (-got +want)\n%s", diff)
	}
}
