//go:build linux

/*
Copyright The Kubernetes Authors.

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

package nftables

import (
	"context"
	"errors"
	"testing"

	"sigs.k8s.io/knftables"
)

func addKubeProxyTable(t *testing.T, nft *knftables.Fake) {
	t.Helper()
	tx := nft.NewTransaction()
	tx.Add(&knftables.Table{})
	if err := nft.Run(context.Background(), tx); err != nil {
		t.Fatalf("Run: %v", err)
	}
}

func TestCleanupLeftoversDeletesExistingTables(t *testing.T) {
	ctx := context.Background()
	ipv4 := knftables.NewFake(knftables.IPv4Family, kubeProxyTable)
	ipv6 := knftables.NewFake(knftables.IPv6Family, kubeProxyTable)
	addKubeProxyTable(t, ipv4)
	addKubeProxyTable(t, ipv6)

	if got := cleanupLeftoversForFamily(ctx, ipv4); got {
		t.Fatalf("cleanupLeftoversForFamily(ipv4) = %v, want false", got)
	}
	if got := cleanupLeftoversForFamily(ctx, ipv6); got {
		t.Fatalf("cleanupLeftoversForFamily(ipv6) = %v, want false", got)
	}
	if ipv4.Table != nil {
		t.Error("IPv4 kube-proxy table still present after cleanup")
	}
	if ipv6.Table != nil {
		t.Error("IPv6 kube-proxy table still present after cleanup")
	}
}

func TestCleanupLeftoversNotFoundIgnored(t *testing.T) {
	ctx := context.Background()
	// No table added: delete is a no-op / NotFound on fake.
	nft := knftables.NewFake(knftables.IPv4Family, kubeProxyTable)
	if got := cleanupLeftoversForFamily(ctx, nft); got {
		t.Fatalf("cleanupLeftoversForFamily() = %v, want false when table is absent", got)
	}
}

// fakeWithRunErr embeds knftables.Fake and returns runErr from Run for testing non-NotFound failures.
type fakeWithRunErr struct {
	*knftables.Fake
	runErr error
}

func (f *fakeWithRunErr) Run(ctx context.Context, tx *knftables.Transaction) error {
	return f.runErr
}

func TestCleanupLeftoversRunErrorSetsEncountered(t *testing.T) {
	ctx := context.Background()
	runErr := errors.New("nft run failed")
	nft := &fakeWithRunErr{Fake: knftables.NewFake(knftables.IPv4Family, kubeProxyTable), runErr: runErr}
	if got := cleanupLeftoversForFamily(ctx, nft); !got {
		t.Fatal("cleanupLeftoversForFamily() = false, want true when Run returns a non-NotFound error")
	}
}
