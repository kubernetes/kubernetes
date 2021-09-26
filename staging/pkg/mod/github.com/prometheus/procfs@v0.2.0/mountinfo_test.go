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
	"reflect"
	"testing"
)

func TestMountInfo(t *testing.T) {
	tests := []struct {
		name    string
		s       string
		mount   *MountInfo
		invalid bool
	}{
		{
			name:    "Regular sysfs mounted at /sys",
			s:       "16 21 0:16 / /sys rw,nosuid,nodev,noexec,relatime shared:7 - sysfs sysfs rw",
			invalid: false,
			mount: &MountInfo{
				MountID:        16,
				ParentID:       21,
				MajorMinorVer:  "0:16",
				Root:           "/",
				MountPoint:     "/sys",
				Options:        map[string]string{"rw": "", "nosuid": "", "nodev": "", "noexec": "", "relatime": ""},
				OptionalFields: map[string]string{"shared": "7"},
				FSType:         "sysfs",
				Source:         "sysfs",
				SuperOptions:   map[string]string{"rw": ""},
			},
		},
		{
			name:    "Not enough information",
			s:       "hello",
			invalid: true,
		},
		{
			name: "Tmpfs mounted at /run",
			s:    "225 20 0:39 / /run/user/112 rw,nosuid,nodev,relatime shared:177 - tmpfs tmpfs rw,size=405096k,mode=700,uid=112,gid=116",
			mount: &MountInfo{
				MountID:        225,
				ParentID:       20,
				MajorMinorVer:  "0:39",
				Root:           "/",
				MountPoint:     "/run/user/112",
				Options:        map[string]string{"rw": "", "nosuid": "", "nodev": "", "relatime": ""},
				OptionalFields: map[string]string{"shared": "177"},
				FSType:         "tmpfs",
				Source:         "tmpfs",
				SuperOptions:   map[string]string{"rw": "", "size": "405096k", "mode": "700", "uid": "112", "gid": "116"},
			},
			invalid: false,
		},
		{
			name: "Tmpfs mounted at /run, but no optional values",
			s:    "225 20 0:39 / /run/user/112 rw,nosuid,nodev,relatime  - tmpfs tmpfs rw,size=405096k,mode=700,uid=112,gid=116",
			mount: &MountInfo{
				MountID:        225,
				ParentID:       20,
				MajorMinorVer:  "0:39",
				Root:           "/",
				MountPoint:     "/run/user/112",
				Options:        map[string]string{"rw": "", "nosuid": "", "nodev": "", "relatime": ""},
				OptionalFields: nil,
				FSType:         "tmpfs",
				Source:         "tmpfs",
				SuperOptions:   map[string]string{"rw": "", "size": "405096k", "mode": "700", "uid": "112", "gid": "116"},
			},
			invalid: false,
		},
		{
			name: "Tmpfs mounted at /run, with multiple optional values",
			s:    "225 20 0:39 / /run/user/112 rw,nosuid,nodev,relatime shared:177 master:8 - tmpfs tmpfs rw,size=405096k,mode=700,uid=112,gid=116",
			mount: &MountInfo{
				MountID:        225,
				ParentID:       20,
				MajorMinorVer:  "0:39",
				Root:           "/",
				MountPoint:     "/run/user/112",
				Options:        map[string]string{"rw": "", "nosuid": "", "nodev": "", "relatime": ""},
				OptionalFields: map[string]string{"shared": "177", "master": "8"},
				FSType:         "tmpfs",
				Source:         "tmpfs",
				SuperOptions:   map[string]string{"rw": "", "size": "405096k", "mode": "700", "uid": "112", "gid": "116"},
			},
			invalid: false,
		},
		{
			name: "Tmpfs mounted at /run, with a mixture of valid and invalid optional values",
			s:    "225 20 0:39 / /run/user/112 rw,nosuid,nodev,relatime shared:177 master:8 foo:bar - tmpfs tmpfs rw,size=405096k,mode=700,uid=112,gid=116",
			mount: &MountInfo{
				MountID:        225,
				ParentID:       20,
				MajorMinorVer:  "0:39",
				Root:           "/",
				MountPoint:     "/run/user/112",
				Options:        map[string]string{"rw": "", "nosuid": "", "nodev": "", "relatime": ""},
				OptionalFields: map[string]string{"shared": "177", "master": "8"},
				FSType:         "tmpfs",
				Source:         "tmpfs",
				SuperOptions:   map[string]string{"rw": "", "size": "405096k", "mode": "700", "uid": "112", "gid": "116"},
			},
			invalid: false,
		},
		{
			name: "CIFS mounted at /with/a-hyphen",
			s:    "454 29 0:87 / /with/a-hyphen rw,relatime shared:255 - cifs //remote-storage/Path rw,vers=3.1.1,cache=strict,username=user,uid=1000,forceuid,gid=0,noforcegid,addr=127.0.0.1,file_mode=0755,dir_mode=0755,soft,nounix,serverino,mapposix,echo_interval=60,actimeo=1",
			mount: &MountInfo{
				MountID:        454,
				ParentID:       29,
				MajorMinorVer:  "0:87",
				Root:           "/",
				MountPoint:     "/with/a-hyphen",
				Options:        map[string]string{"rw": "", "relatime": ""},
				OptionalFields: map[string]string{"shared": "255"},
				FSType:         "cifs",
				Source:         "//remote-storage/Path",
				SuperOptions:   map[string]string{"rw": "", "vers": "3.1.1", "cache": "strict", "username": "user", "uid": "1000", "forceuid": "", "gid": "0", "noforcegid": "", "addr": "127.0.0.1", "file_mode": "0755", "dir_mode": "0755", "soft": "", "nounix": "", "serverino": "", "mapposix": "", "echo_interval": "60", "actimeo": "1"},
			},
			invalid: false,
		},
		{
			name: "Docker overlay with 10 fields (no optional fields)",
			s:    "137 45 253:2 /lib/docker/overlay2 /var/lib/docker/overlay2 rw,relatime - ext4 /dev/mapper/vg0-lv_var rw,data=ordered",
			mount: &MountInfo{
				MountID:        137,
				ParentID:       45,
				MajorMinorVer:  "253:2",
				Root:           "/lib/docker/overlay2",
				MountPoint:     "/var/lib/docker/overlay2",
				Options:        map[string]string{"rw": "", "relatime": ""},
				OptionalFields: map[string]string{},
				FSType:         "ext4",
				Source:         "/dev/mapper/vg0-lv_var",
				SuperOptions:   map[string]string{"rw": "", "data": "ordered"},
			},
		},
		{
			name: "bind chroot bind mount with 10 fields (no optional fields)",
			s:    "157 47 253:2 /etc/named /var/named/chroot/etc/named rw,relatime - ext4 /dev/mapper/vg0-lv_root rw,data=ordered",
			mount: &MountInfo{
				MountID:        157,
				ParentID:       47,
				MajorMinorVer:  "253:2",
				Root:           "/etc/named",
				MountPoint:     "/var/named/chroot/etc/named",
				Options:        map[string]string{"rw": "", "relatime": ""},
				OptionalFields: map[string]string{},
				FSType:         "ext4",
				Source:         "/dev/mapper/vg0-lv_root",
				SuperOptions:   map[string]string{"rw": "", "data": "ordered"},
			},
		},
	}

	for i, test := range tests {
		t.Logf("[%02d] test %q", i, test.name)

		mount, err := parseMountInfoString(test.s)

		if test.invalid && err == nil {
			t.Error("expected an error, but none occurred")
		}
		if !test.invalid && err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if want, have := test.mount, mount; !reflect.DeepEqual(want, have) {
			t.Errorf("mounts:\nwant:\n%+v\nhave:\n%+v", want, have)
		}
	}
}
