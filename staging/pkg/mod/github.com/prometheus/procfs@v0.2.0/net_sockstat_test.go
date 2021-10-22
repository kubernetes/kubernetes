// Copyright 2019 The Prometheus Authors
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
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// The majority of test cases are covered in Test_parseSockstat. The top level
// tests just check the basics for integration test purposes.

func TestNetSockstat(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatalf("failed to open procfs: %v", err)
	}

	stat, err := fs.NetSockstat()
	if err != nil {
		t.Fatalf("failed to get sockstat: %v", err)
	}

	// IPv4 stats should include Used.
	if stat.Used == nil {
		t.Fatalf("IPv4 sockstat used value was nil")
	}
	if diff := cmp.Diff(1602, *stat.Used); diff != "" {
		t.Fatalf("unexpected IPv4 used sockets (-want +got):\n%s", diff)
	}

	// TCP occurs first; do a basic sanity check.
	if diff := cmp.Diff("TCP", stat.Protocols[0].Protocol); diff != "" {
		t.Fatalf("unexpected socket protocol (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(35, stat.Protocols[0].InUse); diff != "" {
		t.Fatalf("unexpected number of TCP sockets (-want +got):\n%s", diff)
	}
}

func TestNetSockstat6(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatalf("failed to open procfs: %v", err)
	}

	stat, err := fs.NetSockstat6()
	if err != nil {
		t.Fatalf("failed to get sockstat: %v", err)
	}

	// IPv6 stats should not include Used.
	if stat.Used != nil {
		t.Fatalf("IPv6 sockstat used value was not nil")
	}

	// TCP6 occurs first; do a basic sanity check.
	if diff := cmp.Diff("TCP6", stat.Protocols[0].Protocol); diff != "" {
		t.Fatalf("unexpected socket protocol (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(17, stat.Protocols[0].InUse); diff != "" {
		t.Fatalf("unexpected number of TCP sockets (-want +got):\n%s", diff)
	}
}

func Test_readSockstatIsNotExist(t *testing.T) {
	// On a machine with IPv6 disabled for example, we want to ensure that
	// readSockstat returns an error that is compatible with os.IsNotExist.
	//
	// We can use a synthetic file path here to verify this behavior.
	_, err := readSockstat("/does/not/exist")
	if err == nil || !os.IsNotExist(err) {
		t.Fatalf("error is not compatible with os.IsNotExist: %#v", err)
	}
}

func Test_parseSockstat(t *testing.T) {
	tests := []struct {
		name string
		s    string
		ok   bool
		stat *NetSockstat
	}{
		{
			name: "empty",
			ok:   true,
			stat: &NetSockstat{},
		},
		{
			name: "bad line",
			s: `
sockets: used
`,
		},
		{
			name: "bad key/value pairs",
			s: `
TCP: inuse 32 orphan
`,
		},
		{
			name: "IPv4",
			s: `
sockets: used 1591
TCP: inuse 32 orphan 0 tw 0 alloc 58 mem 13
UDP: inuse 8 mem 115
UDPLITE: inuse 0
RAW: inuse 0
FRAG: inuse 0 memory 0
			`,
			ok: true,
			stat: &NetSockstat{
				Used: intp(1591),
				Protocols: []NetSockstatProtocol{
					{
						Protocol: "TCP",
						InUse:    32,
						Orphan:   intp(0),
						TW:       intp(0),
						Alloc:    intp(58),
						Mem:      intp(13),
					},
					{
						Protocol: "UDP",
						InUse:    8,
						Mem:      intp(115),
					},
					{
						Protocol: "UDPLITE",
					},
					{
						Protocol: "RAW",
					},
					{
						Protocol: "FRAG",
						Memory:   intp(0),
					},
				},
			},
		},
		{
			name: "IPv6",
			s: `
TCP6: inuse 24
UDP6: inuse 9
UDPLITE6: inuse 0
RAW6: inuse 1
FRAG6: inuse 0 memory 0
			`,
			ok: true,
			stat: &NetSockstat{
				Protocols: []NetSockstatProtocol{
					{
						Protocol: "TCP6",
						InUse:    24,
					},
					{
						Protocol: "UDP6",
						InUse:    9,
					},
					{
						Protocol: "UDPLITE6",
					},
					{
						Protocol: "RAW6",
						InUse:    1,
					},
					{
						Protocol: "FRAG6",
						Memory:   intp(0),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stat, err := parseSockstat(strings.NewReader(strings.TrimSpace(tt.s)))
			if err != nil {
				if tt.ok {
					t.Fatalf("failed to parse sockstats: %v", err)
				}

				t.Logf("OK error: %v", err)
				return
			}
			if !tt.ok {
				t.Fatal("expected an error, but none occurred")
			}

			if diff := cmp.Diff(tt.stat, stat); diff != "" {
				t.Errorf("unexpected sockstats (-want +got):\n%s", diff)
			}
		})
	}
}

func intp(i int) *int { return &i }
