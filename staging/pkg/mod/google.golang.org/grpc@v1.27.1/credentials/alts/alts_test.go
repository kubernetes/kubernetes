/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package alts

import (
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"
	altspb "google.golang.org/grpc/credentials/alts/internal/proto/grpc_gcp"
)

func TestInfoServerName(t *testing.T) {
	// This is not testing any handshaker functionality, so it's fine to only
	// use NewServerCreds and not NewClientCreds.
	alts := NewServerCreds(DefaultServerOptions())
	if got, want := alts.Info().ServerName, ""; got != want {
		t.Fatalf("%v.Info().ServerName = %v, want %v", alts, got, want)
	}
}

func TestOverrideServerName(t *testing.T) {
	wantServerName := "server.name"
	// This is not testing any handshaker functionality, so it's fine to only
	// use NewServerCreds and not NewClientCreds.
	c := NewServerCreds(DefaultServerOptions())
	c.OverrideServerName(wantServerName)
	if got, want := c.Info().ServerName, wantServerName; got != want {
		t.Fatalf("c.Info().ServerName = %v, want %v", got, want)
	}
}

func TestCloneClient(t *testing.T) {
	wantServerName := "server.name"
	opt := DefaultClientOptions()
	opt.TargetServiceAccounts = []string{"not", "empty"}
	c := NewClientCreds(opt)
	c.OverrideServerName(wantServerName)
	cc := c.Clone()
	if got, want := cc.Info().ServerName, wantServerName; got != want {
		t.Fatalf("cc.Info().ServerName = %v, want %v", got, want)
	}
	cc.OverrideServerName("")
	if got, want := c.Info().ServerName, wantServerName; got != want {
		t.Fatalf("Change in clone should not affect the original, c.Info().ServerName = %v, want %v", got, want)
	}
	if got, want := cc.Info().ServerName, ""; got != want {
		t.Fatalf("cc.Info().ServerName = %v, want %v", got, want)
	}

	ct := c.(*altsTC)
	cct := cc.(*altsTC)

	if ct.side != cct.side {
		t.Errorf("cc.side = %q, want %q", cct.side, ct.side)
	}
	if ct.hsAddress != cct.hsAddress {
		t.Errorf("cc.hsAddress = %q, want %q", cct.hsAddress, ct.hsAddress)
	}
	if !reflect.DeepEqual(ct.accounts, cct.accounts) {
		t.Errorf("cc.accounts = %q, want %q", cct.accounts, ct.accounts)
	}
}

func TestCloneServer(t *testing.T) {
	wantServerName := "server.name"
	c := NewServerCreds(DefaultServerOptions())
	c.OverrideServerName(wantServerName)
	cc := c.Clone()
	if got, want := cc.Info().ServerName, wantServerName; got != want {
		t.Fatalf("cc.Info().ServerName = %v, want %v", got, want)
	}
	cc.OverrideServerName("")
	if got, want := c.Info().ServerName, wantServerName; got != want {
		t.Fatalf("Change in clone should not affect the original, c.Info().ServerName = %v, want %v", got, want)
	}
	if got, want := cc.Info().ServerName, ""; got != want {
		t.Fatalf("cc.Info().ServerName = %v, want %v", got, want)
	}

	ct := c.(*altsTC)
	cct := cc.(*altsTC)

	if ct.side != cct.side {
		t.Errorf("cc.side = %q, want %q", cct.side, ct.side)
	}
	if ct.hsAddress != cct.hsAddress {
		t.Errorf("cc.hsAddress = %q, want %q", cct.hsAddress, ct.hsAddress)
	}
	if !reflect.DeepEqual(ct.accounts, cct.accounts) {
		t.Errorf("cc.accounts = %q, want %q", cct.accounts, ct.accounts)
	}
}

func TestInfo(t *testing.T) {
	// This is not testing any handshaker functionality, so it's fine to only
	// use NewServerCreds and not NewClientCreds.
	c := NewServerCreds(DefaultServerOptions())
	info := c.Info()
	if got, want := info.ProtocolVersion, ""; got != want {
		t.Errorf("info.ProtocolVersion=%v, want %v", got, want)
	}
	if got, want := info.SecurityProtocol, "alts"; got != want {
		t.Errorf("info.SecurityProtocol=%v, want %v", got, want)
	}
	if got, want := info.SecurityVersion, "1.0"; got != want {
		t.Errorf("info.SecurityVersion=%v, want %v", got, want)
	}
	if got, want := info.ServerName, ""; got != want {
		t.Errorf("info.ServerName=%v, want %v", got, want)
	}
}

func TestCompareRPCVersions(t *testing.T) {
	for _, tc := range []struct {
		v1     *altspb.RpcProtocolVersions_Version
		v2     *altspb.RpcProtocolVersions_Version
		output int
	}{
		{
			version(3, 2),
			version(2, 1),
			1,
		},
		{
			version(3, 2),
			version(3, 1),
			1,
		},
		{
			version(2, 1),
			version(3, 2),
			-1,
		},
		{
			version(3, 1),
			version(3, 2),
			-1,
		},
		{
			version(3, 2),
			version(3, 2),
			0,
		},
	} {
		if got, want := compareRPCVersions(tc.v1, tc.v2), tc.output; got != want {
			t.Errorf("compareRPCVersions(%v, %v)=%v, want %v", tc.v1, tc.v2, got, want)
		}
	}
}

func TestCheckRPCVersions(t *testing.T) {
	for _, tc := range []struct {
		desc             string
		local            *altspb.RpcProtocolVersions
		peer             *altspb.RpcProtocolVersions
		output           bool
		maxCommonVersion *altspb.RpcProtocolVersions_Version
	}{
		{
			"local.max > peer.max and local.min > peer.min",
			versions(2, 1, 3, 2),
			versions(1, 2, 2, 1),
			true,
			version(2, 1),
		},
		{
			"local.max > peer.max and local.min < peer.min",
			versions(1, 2, 3, 2),
			versions(2, 1, 2, 1),
			true,
			version(2, 1),
		},
		{
			"local.max > peer.max and local.min = peer.min",
			versions(2, 1, 3, 2),
			versions(2, 1, 2, 1),
			true,
			version(2, 1),
		},
		{
			"local.max < peer.max and local.min > peer.min",
			versions(2, 1, 2, 1),
			versions(1, 2, 3, 2),
			true,
			version(2, 1),
		},
		{
			"local.max = peer.max and local.min > peer.min",
			versions(2, 1, 2, 1),
			versions(1, 2, 2, 1),
			true,
			version(2, 1),
		},
		{
			"local.max < peer.max and local.min < peer.min",
			versions(1, 2, 2, 1),
			versions(2, 1, 3, 2),
			true,
			version(2, 1),
		},
		{
			"local.max < peer.max and local.min = peer.min",
			versions(1, 2, 2, 1),
			versions(1, 2, 3, 2),
			true,
			version(2, 1),
		},
		{
			"local.max = peer.max and local.min < peer.min",
			versions(1, 2, 2, 1),
			versions(2, 1, 2, 1),
			true,
			version(2, 1),
		},
		{
			"all equal",
			versions(2, 1, 2, 1),
			versions(2, 1, 2, 1),
			true,
			version(2, 1),
		},
		{
			"max is smaller than min",
			versions(2, 1, 1, 2),
			versions(2, 1, 1, 2),
			false,
			nil,
		},
		{
			"no overlap, local > peer",
			versions(4, 3, 6, 5),
			versions(1, 0, 2, 1),
			false,
			nil,
		},
		{
			"no overlap, local < peer",
			versions(1, 0, 2, 1),
			versions(4, 3, 6, 5),
			false,
			nil,
		},
		{
			"no overlap, max < min",
			versions(6, 5, 4, 3),
			versions(2, 1, 1, 0),
			false,
			nil,
		},
	} {
		output, maxCommonVersion := checkRPCVersions(tc.local, tc.peer)
		if got, want := output, tc.output; got != want {
			t.Errorf("%v: checkRPCVersions(%v, %v)=(%v, _), want (%v, _)", tc.desc, tc.local, tc.peer, got, want)
		}
		if got, want := maxCommonVersion, tc.maxCommonVersion; !proto.Equal(got, want) {
			t.Errorf("%v: checkRPCVersions(%v, %v)=(_, %v), want (_, %v)", tc.desc, tc.local, tc.peer, got, want)
		}
	}
}

func version(major, minor uint32) *altspb.RpcProtocolVersions_Version {
	return &altspb.RpcProtocolVersions_Version{
		Major: major,
		Minor: minor,
	}
}

func versions(minMajor, minMinor, maxMajor, maxMinor uint32) *altspb.RpcProtocolVersions {
	return &altspb.RpcProtocolVersions{
		MinRpcVersion: version(minMajor, minMinor),
		MaxRpcVersion: version(maxMajor, maxMinor),
	}
}
