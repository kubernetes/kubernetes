// Copyright 2015 The appc Authors
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

package main

import (
	"strings"
	"testing"
)

var mountinfoTests = []struct {
	mountinfo     string
	mountPoint    string
	readOnly      bool
	expectMounted bool
	expectErr     bool
}{
	{
		`15 19 0:4 / /proc rw,nosuid,nodev,noexec,relatime shared:5 - proc proc rw
16 19 0:15 / /sys rw,nosuid,nodev,noexec,relatime shared:6 - sysfs sys rw
17 19 0:6 / /dev rw,nosuid,relatime shared:2 - devtmpfs dev rw,size=6089152k,nr_inodes=1522288,mode=755
18 19 0:16 / /run rw,nosuid,nodev,relatime shared:12 - tmpfs run rw,mode=755
19 0 254:0 / / rw,relatime shared:1 - ext4 /dev/mapper/cryptroot rw,discard,data=ordered
20 16 0:17 / /sys/kernel/security rw,nosuid,nodev,noexec,relatime shared:7 - securityfs securityfs rw
21 17 0:18 / /dev/shm rw,nosuid,nodev shared:3 - tmpfs tmpfs rw
66 19 8:2 / /mountPoint1 rw,relatime,nodev shared:26 slave:88 - ext4 /dev/sda2 rw,discard,data=ordered`,
		"/mountPoint1",
		false,
		true,
		false,
	},
	{
		`15 19 0:4 / /proc rw,nosuid,nodev,noexec,relatime shared:5 - proc proc rw
16 19 0:15 / /sys rw,nosuid,nodev,noexec,relatime shared:6 - sysfs sys rw
17 19 0:6 / /dev rw,nosuid,relatime shared:2 - devtmpfs dev rw,size=6089152k,nr_inodes=1522288,mode=755
18 19 0:16 / /run rw,nosuid,nodev,relatime shared:12 - tmpfs run rw,mode=755
66 19 8:2 / /mountPoint2 relatime,ro,nodev shared:26 slave:88 - ext4 /dev/sda2 rw,discard,data=ordered
19 0 254:0 / / rw,relatime shared:1 - ext4 /dev/mapper/cryptroot rw,discard,data=ordered
20 16 0:17 / /sys/kernel/security rw,nosuid,nodev,noexec,relatime shared:7 - securityfs securityfs rw
21 17 0:18 / /dev/shm rw,nosuid,nodev shared:3 - tmpfs tmpfs rw`,
		"/mountPoint2",
		true,
		true,
		false,
	},
	{
		`15 19 0:4 / /proc rw,nosuid,nodev,noexec,relatime shared:5 - proc proc rw
16 19 0:15 / /sys rw,nosuid,nodev,noexec,relatime shared:6 - sysfs sys rw
17 19 0:6 / /dev rw,nosuid,relatime shared:2 - devtmpfs dev rw,size=6089152k,nr_inodes=1522288,mode=755
18 19 0:16 / /run rw,nosuid,nodev,relatime shared:12 - tmpfs run rw,mode=755
19 0 254:0 / / rw,relatime shared:1 - ext4 /dev/mapper/cryptroot rw,discard,data=ordered
20 16 0:17 / /sys/kernel/security rw,nosuid,nodev,noexec,relatime shared:7 - securityfs securityfs rw
21 17 0:18 / /dev/shm rw,nosuid,nodev shared:3 - tmpfs tmpfs rw`,
		"/mountPoint3",
		false,
		false,
		false,
	},
	{
		`15 19 0:4 / /proc rw,nosuid,nodev,noexec,relatime shared:5 - proc proc rw
12 19 8:2 /var/tmp /mountPoint4 rw,relatime,nodev - ext4 /dev/sda2 rw,discard,data=ordered
16 19 - sys rw
17 19 0:6 / /dev rw,nosuid,relatime shared:2 - devtmpfs dev rw,size=6089152k,nr_inodes=1522288,mode=755
18 19 0:16 / /run rw,nosuid,nodev,relatime shared:12 - tmpfs run rw,mode=755
19 0 254:0 / / rw,relatime shared:1 - ext4 /dev/mapper/cryptroot rw,discard,data=ordered
20 16 0:17 / /sys/kernel/security rw,nosuid,nodev,noexec,relatime shared:7 - securityfs securityfs rw
21 17 0:18 / /dev/shm rw,nosuid,nodev shared:3 - tmpfs tmpfs rw`,
		"/mountPoint4",
		false,
		false,
		true,
	},
}

func TestCheckMountLinux(t *testing.T) {
	for i, mi := range mountinfoTests {
		isMounted, ro, err := parseMountinfo(strings.NewReader(mi.mountinfo), mi.mountPoint)
		if err != nil {
			if mi.expectErr {
				continue
			} else {
				t.Fatalf("#%d: unexpected error: `%v`", i, err)
			}
		}
		if isMounted != mi.expectMounted {
			t.Fatalf("#%d: expected isMounted=`%v` but got `%v`", i, mi.expectMounted, isMounted)
		}
		if ro != mi.readOnly {
			t.Fatalf("#%d: expected readOnly=`%v` but got `%v`", i, mi.readOnly, ro)
		}
	}
}
