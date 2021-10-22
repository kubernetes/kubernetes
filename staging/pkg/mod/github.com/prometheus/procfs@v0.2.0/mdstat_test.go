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

import "testing"

func TestFS_MDStat(t *testing.T) {
	fs := getProcFixtures(t)
	mdStats, err := fs.MDStat()

	if err != nil {
		t.Fatalf("parsing of reference-file failed entirely: %s", err)
	}

	refs := map[string]MDStat{
		"md127": {Name: "md127", ActivityState: "active", DisksActive: 2, DisksTotal: 2, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 312319552, BlocksSynced: 312319552},
		"md0":   {Name: "md0", ActivityState: "active", DisksActive: 2, DisksTotal: 2, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 248896, BlocksSynced: 248896},
		"md4":   {Name: "md4", ActivityState: "inactive", DisksActive: 0, DisksTotal: 0, DisksFailed: 1, DisksSpare: 1, BlocksTotal: 4883648, BlocksSynced: 4883648},
		"md6":   {Name: "md6", ActivityState: "recovering", DisksActive: 1, DisksTotal: 2, DisksFailed: 1, DisksSpare: 1, BlocksTotal: 195310144, BlocksSynced: 16775552},
		"md3":   {Name: "md3", ActivityState: "active", DisksActive: 8, DisksTotal: 8, DisksFailed: 0, DisksSpare: 2, BlocksTotal: 5853468288, BlocksSynced: 5853468288},
		"md8":   {Name: "md8", ActivityState: "resyncing", DisksActive: 2, DisksTotal: 2, DisksFailed: 0, DisksSpare: 2, BlocksTotal: 195310144, BlocksSynced: 16775552},
		"md7":   {Name: "md7", ActivityState: "active", DisksActive: 3, DisksTotal: 4, DisksFailed: 1, DisksSpare: 0, BlocksTotal: 7813735424, BlocksSynced: 7813735424},
		"md9":   {Name: "md9", ActivityState: "resyncing", DisksActive: 4, DisksTotal: 4, DisksSpare: 1, DisksFailed: 2, BlocksTotal: 523968, BlocksSynced: 0},
		"md10":  {Name: "md10", ActivityState: "active", DisksActive: 2, DisksTotal: 2, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 314159265, BlocksSynced: 314159265},
		"md11":  {Name: "md11", ActivityState: "resyncing", DisksActive: 2, DisksTotal: 2, DisksFailed: 1, DisksSpare: 2, BlocksTotal: 4190208, BlocksSynced: 0},
		"md12":  {Name: "md12", ActivityState: "active", DisksActive: 2, DisksTotal: 2, DisksSpare: 0, DisksFailed: 0, BlocksTotal: 3886394368, BlocksSynced: 3886394368},
		"md120": {Name: "md120", ActivityState: "active", DisksActive: 2, DisksTotal: 2, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 2095104, BlocksSynced: 2095104},
		"md126": {Name: "md126", ActivityState: "active", DisksActive: 2, DisksTotal: 2, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 1855870976, BlocksSynced: 1855870976},
		"md219": {Name: "md219", ActivityState: "inactive", DisksTotal: 0, DisksFailed: 0, DisksActive: 0, DisksSpare: 3, BlocksTotal: 7932, BlocksSynced: 7932},
		"md00":  {Name: "md00", ActivityState: "active", DisksActive: 1, DisksTotal: 1, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 4186624, BlocksSynced: 4186624},
		"md101": {Name: "md101", ActivityState: "active", DisksActive: 3, DisksTotal: 3, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 322560, BlocksSynced: 322560},
		"md201": {Name: "md201", ActivityState: "checking", DisksActive: 2, DisksTotal: 2, DisksFailed: 0, DisksSpare: 0, BlocksTotal: 1993728, BlocksSynced: 114176},
	}

	if want, have := len(refs), len(mdStats); want != have {
		t.Errorf("want %d parsed md-devices, have %d", want, have)
	}
	for _, md := range mdStats {
		if want, have := refs[md.Name], md; want != have {
			t.Errorf("%s: want %v, have %v", md.Name, want, have)
		}
	}

}

func TestInvalidMdstat(t *testing.T) {
	invalidMount := []byte(`
Personalities : [invalid]
md3 : invalid
      314159265 blocks 64k chunks

unused devices: <none>
`)

	_, err := parseMDStat(invalidMount)
	if err == nil {
		t.Fatalf("parsing of invalid reference file did not find any errors")
	}
}
