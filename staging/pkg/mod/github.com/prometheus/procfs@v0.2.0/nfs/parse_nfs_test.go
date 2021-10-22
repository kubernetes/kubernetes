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

func TestNewNFSClientRPCStats(t *testing.T) {
	tests := []struct {
		name    string
		content string
		stats   *nfs.ClientRPCStats
		invalid bool
	}{
		{
			name:    "invalid file",
			content: "invalid",
			invalid: true,
		}, {
			name: "good old kernel version file",
			content: `net 70 70 69 45
rpc 1218785755 374636 1218815394
proc2 18 16 57 74 52 71 73 45 86 0 52 83 61 17 53 50 23 70 82
proc3 22 0 1061909262 48906 4077635 117661341 5 29391916 2570425 2993289 590 0 0 7815 15 1130 0 3983 92385 13332 2 1 23729
proc4 48 98 51 54 83 85 23 24 1 28 73 68 83 12 84 39 68 59 58 88 29 74 69 96 21 84 15 53 86 54 66 56 97 36 49 32 85 81 11 58 32 67 13 28 35 90 1 26 1337
`,
			stats: &nfs.ClientRPCStats{
				Network: nfs.Network{
					NetCount:   70,
					UDPCount:   70,
					TCPCount:   69,
					TCPConnect: 45,
				},
				ClientRPC: nfs.ClientRPC{
					RPCCount:        1218785755,
					Retransmissions: 374636,
					AuthRefreshes:   1218815394,
				},
				V2Stats: nfs.V2Stats{
					Null:     16,
					GetAttr:  57,
					SetAttr:  74,
					Root:     52,
					Lookup:   71,
					ReadLink: 73,
					Read:     45,
					WrCache:  86,
					Write:    0,
					Create:   52,
					Remove:   83,
					Rename:   61,
					Link:     17,
					SymLink:  53,
					MkDir:    50,
					RmDir:    23,
					ReadDir:  70,
					FsStat:   82,
				},
				V3Stats: nfs.V3Stats{
					Null:        0,
					GetAttr:     1061909262,
					SetAttr:     48906,
					Lookup:      4077635,
					Access:      117661341,
					ReadLink:    5,
					Read:        29391916,
					Write:       2570425,
					Create:      2993289,
					MkDir:       590,
					SymLink:     0,
					MkNod:       0,
					Remove:      7815,
					RmDir:       15,
					Rename:      1130,
					Link:        0,
					ReadDir:     3983,
					ReadDirPlus: 92385,
					FsStat:      13332,
					FsInfo:      2,
					PathConf:    1,
					Commit:      23729},
				ClientV4Stats: nfs.ClientV4Stats{
					Null:               98,
					Read:               51,
					Write:              54,
					Commit:             83,
					Open:               85,
					OpenConfirm:        23,
					OpenNoattr:         24,
					OpenDowngrade:      1,
					Close:              28,
					Setattr:            73,
					FsInfo:             68,
					Renew:              83,
					SetClientID:        12,
					SetClientIDConfirm: 84,
					Lock:               39,
					Lockt:              68,
					Locku:              59,
					Access:             58,
					Getattr:            88,
					Lookup:             29,
					LookupRoot:         74,
					Remove:             69,
					Rename:             96,
					Link:               21,
					Symlink:            84,
					Create:             15,
					Pathconf:           53,
					StatFs:             86,
					ReadLink:           54,
					ReadDir:            66,
					ServerCaps:         56,
					DelegReturn:        97,
					GetACL:             36,
					SetACL:             49,
					FsLocations:        32,
					ReleaseLockowner:   85,
					Secinfo:            81,
					FsidPresent:        11,
					ExchangeID:         58,
					CreateSession:      32,
					DestroySession:     67,
					Sequence:           13,
					GetLeaseTime:       28,
					ReclaimComplete:    35,
					LayoutGet:          90,
					GetDeviceInfo:      1,
					LayoutCommit:       26,
					LayoutReturn:       1337,
					SecinfoNoName:      0,
					TestStateID:        0,
					FreeStateID:        0,
					GetDeviceList:      0,
					BindConnToSession:  0,
					DestroyClientID:    0,
					Seek:               0,
					Allocate:           0,
					DeAllocate:         0,
					LayoutStats:        0,
					Clone:              0,
				},
			},
		}, {
			name: "good file",
			content: `net 18628 0 18628 6
rpc 4329785 0 4338291
proc2 18 2 69 0 0 4410 0 0 0 0 0 0 0 0 0 0 0 99 2
proc3 22 1 4084749 29200 94754 32580 186 47747 7981 8639 0 6356 0 6962 0 7958 0 0 241 4 4 2 39
proc4 61 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
`,
			stats: &nfs.ClientRPCStats{
				Network: nfs.Network{
					NetCount:   18628,
					UDPCount:   0,
					TCPCount:   18628,
					TCPConnect: 6,
				},
				ClientRPC: nfs.ClientRPC{
					RPCCount:        4329785,
					Retransmissions: 0,
					AuthRefreshes:   4338291,
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
					Null:        1,
					GetAttr:     4084749,
					SetAttr:     29200,
					Lookup:      94754,
					Access:      32580,
					ReadLink:    186,
					Read:        47747,
					Write:       7981,
					Create:      8639,
					MkDir:       0,
					SymLink:     6356,
					MkNod:       0,
					Remove:      6962,
					RmDir:       0,
					Rename:      7958,
					Link:        0,
					ReadDir:     0,
					ReadDirPlus: 241,
					FsStat:      4,
					FsInfo:      4,
					PathConf:    2,
					Commit:      39,
				},
				ClientV4Stats: nfs.ClientV4Stats{
					Null:               1,
					Read:               0,
					Write:              0,
					Commit:             0,
					Open:               0,
					OpenConfirm:        0,
					OpenNoattr:         0,
					OpenDowngrade:      0,
					Close:              0,
					Setattr:            0,
					FsInfo:             0,
					Renew:              0,
					SetClientID:        1,
					SetClientIDConfirm: 1,
					Lock:               0,
					Lockt:              0,
					Locku:              0,
					Access:             0,
					Getattr:            0,
					Lookup:             0,
					LookupRoot:         0,
					Remove:             2,
					Rename:             0,
					Link:               0,
					Symlink:            0,
					Create:             0,
					Pathconf:           0,
					StatFs:             0,
					ReadLink:           0,
					ReadDir:            0,
					ServerCaps:         0,
					DelegReturn:        0,
					GetACL:             0,
					SetACL:             0,
					FsLocations:        0,
					ReleaseLockowner:   0,
					Secinfo:            0,
					FsidPresent:        0,
					ExchangeID:         0,
					CreateSession:      0,
					DestroySession:     0,
					Sequence:           0,
					GetLeaseTime:       0,
					ReclaimComplete:    0,
					LayoutGet:          0,
					GetDeviceInfo:      0,
					LayoutCommit:       0,
					LayoutReturn:       0,
					SecinfoNoName:      0,
					TestStateID:        0,
					FreeStateID:        0,
					GetDeviceList:      0,
					BindConnToSession:  0,
					DestroyClientID:    0,
					Seek:               0,
					Allocate:           0,
					DeAllocate:         0,
					LayoutStats:        0,
					Clone:              0,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stats, err := nfs.ParseClientRPCStats(strings.NewReader(tt.content))

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
