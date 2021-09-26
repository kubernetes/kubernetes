// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

// Whether or not to run tests with inode fixtures.
const (
	checkInode   = true
	noCheckInode = false
)

func TestNetUnix(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatalf("failed to open procfs: %v", err)
	}

	got, err := fs.NetUNIX()
	if err != nil {
		t.Fatalf("failed to get UNIX socket data: %v", err)
	}

	testNetUNIX(t, checkInode, got)
}

func TestNetUnixNoInode(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatalf("failed to open procfs: %v", err)
	}

	got, err := readNetUNIX(fs.proc.Path("net/unix_without_inode"))
	if err != nil {
		t.Fatalf("failed to read UNIX socket data: %v", err)
	}

	testNetUNIX(t, noCheckInode, got)
}

func testNetUNIX(t *testing.T, testInode bool, got *NetUNIX) {
	t.Helper()

	// First verify that the input data matches a prepopulated structure.

	want := []*NetUNIXLine{
		{
			KernelPtr: "0000000000000000",
			RefCount:  2,
			Flags:     1 << 16,
			Type:      1,
			State:     1,
			Inode:     3442596,
			Path:      "/var/run/postgresql/.s.PGSQL.5432",
		},
		{
			KernelPtr: "0000000000000000",
			RefCount:  10,
			Flags:     1 << 16,
			Type:      5,
			State:     1,
			Inode:     10061,
			Path:      "/run/udev/control",
		},
		{
			KernelPtr: "0000000000000000",
			RefCount:  7,
			Flags:     0,
			Type:      2,
			State:     1,
			Inode:     12392,
			Path:      "/dev/log",
		},
		{
			KernelPtr: "0000000000000000",
			RefCount:  3,
			Flags:     0,
			Type:      1,
			State:     3,
			Inode:     4787297,
			Path:      "/var/run/postgresql/.s.PGSQL.5432",
		},
		{
			KernelPtr: "0000000000000000",
			RefCount:  3,
			Flags:     0,
			Type:      1,
			State:     3,
			Inode:     5091797,
		},
	}

	// Enable the fixtures to be used for multiple tests by clearing the inode
	// field when appropriate.
	if !testInode {
		for i := 0; i < len(want); i++ {
			want[i].Inode = 0
		}
	}

	if diff := cmp.Diff(want, got.Rows); diff != "" {
		t.Fatalf("unexpected /proc/net/unix data (-want +got):\n%s", diff)
	}

	// Now test the field enumerations and ensure they match up correctly
	// with the constants used to generate readable strings.

	wantFlags := []NetUNIXFlags{
		netUnixFlagListen,
		netUnixFlagListen,
		netUnixFlagDefault,
		netUnixFlagDefault,
		netUnixFlagDefault,
	}

	wantType := []NetUNIXType{
		netUnixTypeStream,
		netUnixTypeSeqpacket,
		netUnixTypeDgram,
		netUnixTypeStream,
		netUnixTypeStream,
	}

	wantState := []NetUNIXState{
		netUnixStateUnconnected,
		netUnixStateUnconnected,
		netUnixStateUnconnected,
		netUnixStateConnected,
		netUnixStateConnected,
	}

	var (
		gotFlags []NetUNIXFlags
		gotType  []NetUNIXType
		gotState []NetUNIXState
	)

	for _, r := range got.Rows {
		gotFlags = append(gotFlags, r.Flags)
		gotType = append(gotType, r.Type)
		gotState = append(gotState, r.State)
	}

	if diff := cmp.Diff(wantFlags, gotFlags); diff != "" {
		t.Fatalf("unexpected /proc/net/unix flags (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff(wantType, gotType); diff != "" {
		t.Fatalf("unexpected /proc/net/unix types (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff(wantState, gotState); diff != "" {
		t.Fatalf("unexpected /proc/net/unix states (-want +got):\n%s", diff)
	}
}
