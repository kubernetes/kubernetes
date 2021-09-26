// Copyright 2020 The Prometheus Authors
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
	"reflect"
	"testing"
)

func TestParseCgroupString(t *testing.T) {
	tests := []struct {
		name      string
		s         string
		shouldErr bool
		cgroup    *Cgroup
	}{
		{
			name:      "cgroups-v1 simple line",
			s:         "10:rdma:/",
			shouldErr: false,
			cgroup: &Cgroup{
				HierarchyID: 10,
				Controllers: []string{"rdma"},
				Path:        "/",
			},
		},
		{
			name:      "cgroups-v1 multi-hier line",
			s:         "3:cpu,cpuacct:/user.slice/user-1000.slice/session-10.scope",
			shouldErr: false,
			cgroup: &Cgroup{
				HierarchyID: 3,
				Controllers: []string{"cpu", "cpuacct"},
				Path:        "/user.slice/user-1000.slice/session-10.scope",
			},
		},
		{
			name:      "cgroup-v2 line",
			s:         "0::/user.slice/user-1000.slice/user@1000.service/gnome-terminal-server.service",
			shouldErr: false,
			cgroup: &Cgroup{
				HierarchyID: 0,
				Controllers: nil,
				Path:        "/user.slice/user-1000.slice/user@1000.service/gnome-terminal-server.service",
			},
		},
		{
			name:      "extra fields (such as those added by later kernel versions)",
			s:         "0::/:foobar",
			shouldErr: false,
			cgroup: &Cgroup{
				HierarchyID: 0,
				Controllers: nil,
				Path:        "/",
			},
		},
		{
			name:      "bad hierarchy ID field",
			s:         "a:cpu:/",
			shouldErr: true,
			cgroup:    nil,
		},
	}

	for i, test := range tests {
		t.Logf("[%02d] test %q", i, test.name)

		cgroup, err := parseCgroupString(test.s)

		if test.shouldErr && err == nil {
			t.Errorf("%s: expected an error, but none occurred", test.name)
		}
		if !test.shouldErr && err != nil {
			t.Errorf("%s: unexpected error: %v", test.name, err)
		}

		if want, have := test.cgroup, cgroup; !reflect.DeepEqual(want, have) {
			t.Errorf("cgroup:\nwant:\n%+v\nhave:\n%+v", want, have)
		}
	}

}
