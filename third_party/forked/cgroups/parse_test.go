/*
   Copyright The containerd Authors.

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

package cgroups

import (
	"strings"
	"testing"
)

func TestParseCgroupFromReaderUnified(t *testing.T) {
	const data = `10:devices:/user.slice
	9:net_cls,net_prio:/
	8:blkio:/
	7:freezer:/
	6:perf_event:/
	5:cpuset:/
	4:memory:/
	3:pids:/user.slice/user-1000.slice/user@1000.service
	2:cpu,cpuacct:/
	1:name=systemd:/user.slice/user-1000.slice/user@1000.service/gnome-terminal-server.service
	0::/user.slice/user-1000.slice/user@1000.service/gnome-terminal-server.service`
	r := strings.NewReader(data)
	paths, unified, err := parseCgroupFromReaderUnified(r)
	if err != nil {
		t.Fatal(err)
	}
	for subsystem, path := range paths {
		if subsystem == "" {
			t.Fatalf("empty subsystem for %q", path)
		}
	}
	unifiedExpected := "/user.slice/user-1000.slice/user@1000.service/gnome-terminal-server.service"
	if unified != unifiedExpected {
		t.Fatalf("expected %q, got %q", unifiedExpected, unified)
	}
}
