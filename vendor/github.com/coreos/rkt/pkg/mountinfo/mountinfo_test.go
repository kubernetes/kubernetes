// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build linux

package mountinfo

import (
	"bytes"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var mountinfo = `39 21 0:36 / /tmp rw shared:91 -
18 39 0:18 / /sys rw,nosuid,nodev,noexec,relatime shared:7 - sysfs sysfs rw
19 39 0:4 / /proc rw,nosuid,nodev,noexec,relatime shared:13 - proc proc rw
71 21 0:39 / /var/lib/rkt rw,relatime shared:26 -
69 21 0:19 /home /home rw,relatime shared:27 -
70 20 0:41 / /run/user/1000 rw,nosuid,nodev,relatime shared:28 -
109 70 0:43 / /run/user/1000/gvfs rw,nosuid,nodev,relatime shared:61 -
126 71 0:45 / /prefix/stage1/rootfs rw,relatime master:1 -
131 126 0:46 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs rw,relatime shared:1 master:2 -
193 131 0:19 /nixos /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs ro,relatime -
195 193 0:17 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/sys rw,relatime -
196 195 0:26 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/sys/fs/cgroup rw,nosuid,nodev,noexec -
197 196 0:27 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime -
206 193 0:6 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/dev rw,relatime -
207 206 0:23 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/dev/shm rw,relatime -
208 206 0:14 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/dev/pts rw,nosuid,noexec,relatime -
209 206 0:35 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/dev/hugepages rw,relatime -
210 206 0:16 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/dev/mqueue rw,relatime -
211 193 0:18 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/run rw,relatime -
213 211 0:41 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/run/user/1000 rw,nosuid,nodev,relatime -
214 213 0:43 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/run/user/1000/gvfs rw,nosuid,nodev,relatime -
215 193 0:19 /nixos/nix/store /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/nix/store ro,relatime -
217 193 0:36 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp rw -
218 217 0:42 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/c rw,relatime -
219 218 0:44 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/c/b-b rw,relatime -
220 217 0:17 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys rw,relatime -
221 220 0:26 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup rw,nosuid,nodev,noexec -
222 221 0:27 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime -
223 221 0:28 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup/net_cls rw,nosuid,nodev,noexec,relatime -
224 221 0:29 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime -
225 221 0:30 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime -
226 221 0:31 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup/freezer rw,nosuid,nodev,noexec,relatime -
227 221 0:32 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime -
228 221 0:33 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/tmp/sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime -
231 193 0:38 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/var/lib/cni rw,relatime -
232 193 0:39 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/var/lib/rkt rw,relatime -
233 232 0:45 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs rw,relatime -
234 233 0:46 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs rw,relatime -
235 234 0:6 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/dev rw,relatime -
236 235 0:23 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/dev/shm rw,relatime -
237 235 0:14 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/dev/pts rw,nosuid,noexec,relatime -
238 235 0:35 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/dev/hugepages rw,relatime -
239 235 0:16 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/dev/mqueue rw,relatime -
240 234 0:5 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/proc rw,relatime -
241 234 0:17 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys rw,relatime -
242 241 0:26 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup rw,nosuid,nodev,noexec -
243 242 0:27 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime -
244 242 0:28 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/net_cls rw,nosuid,nodev,noexec,relatime -
245 242 0:29 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime -
246 242 0:30 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime -
247 242 0:31 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/freezer rw,nosuid,nodev,noexec,relatime -
248 242 0:32 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime -
249 242 0:33 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime -
250 242 0:34 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/blkio rw,nosuid,nodev,noexec,relatime -
251 242 0:29 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/sys/fs/cgroup/cros rw,relatime -
252 234 0:47 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/tmp rw,relatime -
253 234 0:19 /nixos/etc/resolv.conf /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/prefix/stage1/rootfs/opt/stage2/busybox/rootfs/etc/resolv.conf ro,relatime -
254 193 0:19 /home /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/home rw,relatime -
255 193 179:1 / /prefix/stage1/rootfs/opt/stage2/busybox/rootfs/rootfs/mnt/unknown ro,relatime -
`

// safeOrder checks for transitivity and (un)mount order sanity
func safeOrder(ms Mounts) error {
	for i, m := range ms {
		for j, mnext := range ms[i:] {
			k := i + j
			if !ms.Less(i, k) {
				return fmt.Errorf("Transitivity check failed for %d(%d) and %d(%d)", i, m.ID, k, mnext.ID)
			}
			if strings.HasPrefix(mnext.MountPoint, m.MountPoint+string(filepath.Separator)) {
				return fmt.Errorf("Must not unmount \n%q before\n%q.", m.MountPoint, mnext.MountPoint)
			}
		}
	}
	return nil
}

func TestMountOrdering(t *testing.T) {
	tests := []struct {
		prefix     string
		ids        int
		shouldPass bool
		mi         string
		remounts   int
	}{
		{
			prefix:     "/prefix",
			ids:        52,
			shouldPass: true,
			mi:         mountinfo,
			remounts:   2,
		},
	}

	for i, tt := range tests {
		mi := strings.NewReader(tt.mi)
		mnts, err := parseMountinfo(mi)
		if err != nil {
			t.Errorf("problems finding mount points: %v", err)
		}
		mnts = mnts.Filter(HasPrefix(tt.prefix))

		requestedRemounts := 0
		for i := len(mnts) - 1; i >= 0; i-- {
			mnt := mnts[i]
			if mnt.NeedsRemountPrivate() {
				t.Logf("remounting: %+v", mnt)
				requestedRemounts++
			}
		}

		if requestedRemounts != tt.remounts {
			t.Fatalf("test  %d: didn't find the expected number of remounts. found %d but wanted %d.", i, requestedRemounts, tt.remounts)
		}

		if len(mnts) != tt.ids {
			t.Fatalf("test  %d: didn't find the expected number of mounts. found %d but wanted %d.", i, len(mnts), tt.ids)
		}

		for _, mntCur := range mnts {
			t.Logf("Unmounting %d: %q", mntCur.ID, mntCur.MountPoint)
		}

		if err := safeOrder(mnts); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMountinfoParser(t *testing.T) {
	mountTests := []struct {
		mountinfo string
		prefix    string

		winfos Mounts
		werr   bool
	}{
		{
			"plaintext garbage",
			"",

			nil,
			true,
		},
		{
			mountinfo,
			"/proc",

			Mounts{&Mount{19, 39, 0, 4, "/", "/proc", map[string]struct{}{"shared": {}}}},
			false,
		},
		{
			mountinfo,
			"/nonexisting",

			nil,
			false,
		},
	}

	for i, tt := range mountTests {
		mi := bytes.NewBufferString(tt.mountinfo)
		mnts, err := parseMountinfo(mi)
		if err != nil && !tt.werr {
			t.Errorf("#%d: got unexpected error %v", i, err)
		}
		if err == nil && tt.werr {
			t.Errorf("#%d: expected error, got nil", i)
		}

		mnts = mnts.Filter(HasPrefix(tt.prefix))
		if mnts == nil && tt.winfos != nil {
			t.Errorf("#%d: expected mounts %v, got nil", i, tt.winfos)
		}
		if len(mnts) > 0 {
			if len(tt.winfos) == 0 {
				t.Errorf("#%d: no mounts expected, got %v", i, mnts)
			}
			for j, m := range mnts {
				if m == nil {
					t.Errorf("#%d.%d: got nil mountinfo entry", i, j)
				}
				if !reflect.DeepEqual(m, tt.winfos[j]) {
					t.Errorf("#%d.%d: expected mountinfo %v, got %v", i, j, tt.winfos[j], m)
				}
			}
		}
	}
}
