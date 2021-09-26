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

package nfs_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/prometheus/procfs/nfs"
)

func TestNewNFSdServerRPCStats(t *testing.T) {
	tests := []struct {
		name    string
		content string
		stats   *nfs.ServerRPCStats
		invalid bool
	}{
		{
			name:    "invalid file",
			content: "invalid",
			invalid: true,
		}, {
			name: "good file",
			content: `rc 0 6 18622
fh 0 0 0 0 0
io 157286400 0
th 8 0 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
ra 32 0 0 0 0 0 0 0 0 0 0 0
net 18628 0 18628 6
rpc 18628 0 0 0 0
proc2 18 2 69 0 0 4410 0 0 0 0 0 0 0 0 0 0 0 99 2
proc3 22 2 112 0 2719 111 0 0 0 0 0 0 0 0 0 0 0 27 216 0 2 1 0
proc4 2 2 10853
proc4ops 72 0 0 0 1098 2 0 0 0 0 8179 5896 0 0 0 0 5900 0 0 2 0 2 0 9609 0 2 150 1272 0 0 0 1236 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
`,
			stats: &nfs.ServerRPCStats{
				ReplyCache: nfs.ReplyCache{
					Hits:    0,
					Misses:  6,
					NoCache: 18622,
				},
				FileHandles: nfs.FileHandles{
					Stale:        0,
					TotalLookups: 0,
					AnonLookups:  0,
					DirNoCache:   0,
					NoDirNoCache: 0,
				},
				InputOutput: nfs.InputOutput{
					Read:  157286400,
					Write: 0,
				},
				Threads: nfs.Threads{
					Threads: 8,
					FullCnt: 0,
				},
				ReadAheadCache: nfs.ReadAheadCache{
					CacheSize:      32,
					CacheHistogram: []uint64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					NotFound:       0,
				},
				Network: nfs.Network{
					NetCount:   18628,
					UDPCount:   0,
					TCPCount:   18628,
					TCPConnect: 6,
				},
				ServerRPC: nfs.ServerRPC{
					RPCCount: 18628,
					BadCnt:   0,
					BadFmt:   0,
					BadAuth:  0,
					BadcInt:  0,
				},
				V2Stats: nfs.V2Stats{
					Null:     2,
					GetAttr:  69,
					SetAttr:  0,
					Root:     0,
					Lookup:   4410,
					ReadLink: 0,
					Read:     0,
					WrCache:  0,
					Write:    0,
					Create:   0,
					Remove:   0,
					Rename:   0,
					Link:     0,
					SymLink:  0,
					MkDir:    0,
					RmDir:    0,
					ReadDir:  99,
					FsStat:   2,
				},
				V3Stats: nfs.V3Stats{
					Null:        2,
					GetAttr:     112,
					SetAttr:     0,
					Lookup:      2719,
					Access:      111,
					ReadLink:    0,
					Read:        0,
					Write:       0,
					Create:      0,
					MkDir:       0,
					SymLink:     0,
					MkNod:       0,
					Remove:      0,
					RmDir:       0,
					Rename:      0,
					Link:        0,
					ReadDir:     27,
					ReadDirPlus: 216,
					FsStat:      0,
					FsInfo:      2,
					PathConf:    1,
					Commit:      0,
				},
				ServerV4Stats: nfs.ServerV4Stats{
					Null:     2,
					Compound: 10853,
				},
				V4Ops: nfs.V4Ops{
					Op0Unused:    0,
					Op1Unused:    0,
					Op2Future:    0,
					Access:       1098,
					Close:        2,
					Commit:       0,
					Create:       0,
					DelegPurge:   0,
					DelegReturn:  0,
					GetAttr:      8179,
					GetFH:        5896,
					Link:         0,
					Lock:         0,
					Lockt:        0,
					Locku:        0,
					Lookup:       5900,
					LookupRoot:   0,
					Nverify:      0,
					Open:         2,
					OpenAttr:     0,
					OpenConfirm:  2,
					OpenDgrd:     0,
					PutFH:        9609,
					PutPubFH:     0,
					PutRootFH:    2,
					Read:         150,
					ReadDir:      1272,
					ReadLink:     0,
					Remove:       0,
					Rename:       0,
					Renew:        1236,
					RestoreFH:    0,
					SaveFH:       0,
					SecInfo:      0,
					SetAttr:      0,
					Verify:       3,
					Write:        3,
					RelLockOwner: 0,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stats, err := nfs.ParseServerRPCStats(strings.NewReader(tt.content))

			if tt.invalid && err == nil {
				t.Fatal("expected an error, but none occurred")
			}
			if !tt.invalid && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if want, have := tt.stats, stats; !reflect.DeepEqual(want, have) {
				t.Fatalf("unexpected NFS stats:\nwant:\n%v\nhave:\n%v", want, have)
			}
		})
	}
}
