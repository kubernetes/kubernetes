// +build !windows

/*
Copyright 2019 The Kubernetes Authors.

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

package mount

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeFile(content string) (string, string, error) {
	tempDir, err := ioutil.TempDir("", "mounter_shared_test")
	if err != nil {
		return "", "", err
	}
	filename := filepath.Join(tempDir, "mountinfo")
	err = ioutil.WriteFile(filename, []byte(content), 0600)
	if err != nil {
		os.RemoveAll(tempDir)
		return "", "", err
	}
	return tempDir, filename, nil
}

func TestParseMountInfo(t *testing.T) {
	info :=
		`62 0 253:0 / / rw,relatime shared:1 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
78 62 0:41 / /tmp rw,nosuid,nodev shared:30 - tmpfs tmpfs rw,seclabel
80 62 0:42 / /var/lib/nfs/rpc_pipefs rw,relatime shared:31 - rpc_pipefs sunrpc rw
82 62 0:43 / /var/lib/foo rw,relatime shared:32 - tmpfs tmpfs rw
83 63 0:44 / /var/lib/bar rw,relatime - tmpfs tmpfs rw
227 62 253:0 /var/lib/docker/devicemapper /var/lib/docker/devicemapper rw,relatime - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
224 62 253:0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
76 17 8:1 / /mnt/stateful_partition rw,nosuid,nodev,noexec,relatime - ext4 /dev/sda1 rw,commit=30,data=ordered
80 17 8:1 /var /var rw,nosuid,nodev,noexec,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
189 80 8:1 /var/lib/kubelet /var/lib/kubelet rw,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
818 77 8:40 / /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdc rw,data=ordered
819 78 8:48 / /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
900 100 8:48 /dir1 /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volume-subpaths/vol1/subpath1/0 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
901 101 8:1 /dir1 /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volume-subpaths/vol1/subpath1/1 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
902 102 8:1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volumes/kubernetes.io~empty-dir/vol1/dir1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volume-subpaths/vol1/subpath1/0 rw,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
903 103 8:1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volumes/kubernetes.io~empty-dir/vol2/dir1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volume-subpaths/vol1/subpath1/1 rw,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
178 25 253:0 /etc/bar /var/lib/kubelet/pods/12345/volume-subpaths/vol1/subpath1/0 rw,relatime shared:1 - ext4 /dev/sdb2 rw,errors=remount-ro,data=ordered
698 186 0:41 /tmp1/dir1 /var/lib/kubelet/pods/41135147-e697-11e7-9342-42010a800002/volume-subpaths/vol1/subpath1/0 rw shared:26 - tmpfs tmpfs rw
918 77 8:50 / /var/lib/kubelet/pods/2345/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdc rw,data=ordered
919 78 8:58 / /var/lib/kubelet/pods/2345/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
920 100 8:50 /dir1 /var/lib/kubelet/pods/2345/volume-subpaths/vol1/subpath1/0 rw,relatime shared:290 - ext4 /dev/sdc rw,data=ordered
150 23 1:58 / /media/nfs_vol rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
151 24 1:58 / /media/nfs_bindmount rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs/foo rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
134 23 0:58 / /var/lib/kubelet/pods/43219158-e5e1-11e7-a392-0e858b8eaf40/volumes/kubernetes.io~nfs/nfs1 rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
187 23 0:58 / /var/lib/kubelet/pods/1fc5ea21-eff4-11e7-ac80-0e858b8eaf40/volumes/kubernetes.io~nfs/nfs2 rw,relatime shared:96 - nfs4 172.18.4.223:/srv/nfs2 rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
188 24 0:58 / /var/lib/kubelet/pods/43219158-e5e1-11e7-a392-0e858b8eaf40/volume-subpaths/nfs1/subpath1/0 rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs/foo rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
347 60 0:71 / /var/lib/kubelet/pods/13195d46-f9fa-11e7-bbf1-5254007a695a/volumes/kubernetes.io~nfs/vol2 rw,relatime shared:170 - nfs 172.17.0.3:/exports/2 rw,vers=3,rsize=1048576,wsize=1048576,namlen=255,hard,proto=tcp,timeo=600,retrans=2,sec=sys,mountaddr=172.17.0.3,mountvers=3,mountport=20048,mountproto=udp,local_lock=none,addr=172.17.0.3
222 24 253:0 /tmp/src /mnt/dst rw,relatime shared:1 - ext4 /dev/mapper/vagrant--vg-root rw,errors=remount-ro,data=ordered
28 18 0:24 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,mode=755
29 28 0:25 / /sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime shared:10 - cgroup cgroup rw,xattr,release_agent=/lib/systemd/systemd-cgroups-agent,name=systemd
31 28 0:27 / /sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime shared:13 - cgroup cgroup rw,cpuset
32 28 0:28 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime shared:14 - cgroup cgroup rw,cpu,cpuacct
33 28 0:29 / /sys/fs/cgroup/freezer rw,nosuid,nodev,noexec,relatime shared:15 - cgroup cgroup rw,freezer
34 28 0:30 / /sys/fs/cgroup/net_cls,net_prio rw,nosuid,nodev,noexec,relatime shared:16 - cgroup cgroup rw,net_cls,net_prio
35 28 0:31 / /sys/fs/cgroup/pids rw,nosuid,nodev,noexec,relatime shared:17 - cgroup cgroup rw,pids
36 28 0:32 / /sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime shared:18 - cgroup cgroup rw,devices
37 28 0:33 / /sys/fs/cgroup/hugetlb rw,nosuid,nodev,noexec,relatime shared:19 - cgroup cgroup rw,hugetlb
38 28 0:34 / /sys/fs/cgroup/blkio rw,nosuid,nodev,noexec,relatime shared:20 - cgroup cgroup rw,blkio
39 28 0:35 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:21 - cgroup cgroup rw,memory
40 28 0:36 / /sys/fs/cgroup/perf_event rw,nosuid,nodev,noexec,relatime shared:22 - cgroup cgroup rw,perf_event
`
	tempDir, filename, err := writeFile(info)
	if err != nil {
		t.Fatalf("cannot create temporary file: %v", err)
	}
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name         string
		id           int
		expectedInfo MountInfo
	}{
		{
			"simple bind mount",
			189,
			MountInfo{
				ID:             189,
				ParentID:       80,
				Major:          8,
				Minor:          1,
				Root:           "/var/lib/kubelet",
				Source:         "/dev/sda1",
				MountPoint:     "/var/lib/kubelet",
				OptionalFields: []string{"shared:30"},
				FsType:         "ext4",
				MountOptions:   []string{"rw", "relatime"},
				SuperOptions:   []string{"rw", "commit=30", "data=ordered"},
			},
		},
		{
			"bind mount a directory",
			222,
			MountInfo{
				ID:             222,
				ParentID:       24,
				Major:          253,
				Minor:          0,
				Root:           "/tmp/src",
				Source:         "/dev/mapper/vagrant--vg-root",
				MountPoint:     "/mnt/dst",
				OptionalFields: []string{"shared:1"},
				FsType:         "ext4",
				MountOptions:   []string{"rw", "relatime"},
				SuperOptions:   []string{"rw", "errors=remount-ro", "data=ordered"},
			},
		},
		{
			"more than one optional fields",
			224,
			MountInfo{
				ID:             224,
				ParentID:       62,
				Major:          253,
				Minor:          0,
				Root:           "/var/lib/docker/devicemapper/test/shared",
				Source:         "/dev/mapper/ssd-root",
				MountPoint:     "/var/lib/docker/devicemapper/test/shared",
				OptionalFields: []string{"master:1", "shared:44"},
				FsType:         "ext4",
				MountOptions:   []string{"rw", "relatime"},
				SuperOptions:   []string{"rw", "seclabel", "data=ordered"},
			},
		},
		{
			"cgroup-mountpoint",
			28,
			MountInfo{
				ID:             28,
				ParentID:       18,
				Major:          0,
				Minor:          24,
				Root:           "/",
				Source:         "tmpfs",
				MountPoint:     "/sys/fs/cgroup",
				OptionalFields: []string{"shared:9"},
				FsType:         "tmpfs",
				MountOptions:   []string{"ro", "nosuid", "nodev", "noexec"},
				SuperOptions:   []string{"ro", "mode=755"},
			},
		},
		{
			"cgroup-subsystem-systemd-mountpoint",
			29,
			MountInfo{
				ID:             29,
				ParentID:       28,
				Major:          0,
				Minor:          25,
				Root:           "/",
				Source:         "cgroup",
				MountPoint:     "/sys/fs/cgroup/systemd",
				OptionalFields: []string{"shared:10"},
				FsType:         "cgroup",
				MountOptions:   []string{"rw", "nosuid", "nodev", "noexec", "relatime"},
				SuperOptions:   []string{"rw", "xattr", "release_agent=/lib/systemd/systemd-cgroups-agent", "name=systemd"},
			},
		},
		{
			"cgroup-subsystem-cpuset-mountpoint",
			31,
			MountInfo{
				ID:             31,
				ParentID:       28,
				Major:          0,
				Minor:          27,
				Root:           "/",
				Source:         "cgroup",
				MountPoint:     "/sys/fs/cgroup/cpuset",
				OptionalFields: []string{"shared:13"},
				FsType:         "cgroup",
				MountOptions:   []string{"rw", "nosuid", "nodev", "noexec", "relatime"},
				SuperOptions:   []string{"rw", "cpuset"},
			},
		},
	}

	infos, err := ParseMountInfo(filename)
	if err != nil {
		t.Fatalf("Cannot parse %s: %s", filename, err)
	}

	for _, test := range tests {
		found := false
		for _, info := range infos {
			if info.ID == test.id {
				found = true
				if !reflect.DeepEqual(info, test.expectedInfo) {
					t.Errorf("Test case %q:\n expected: %+v\n got:      %+v", test.name, test.expectedInfo, info)
				}
				break
			}
		}
		if !found {
			t.Errorf("Test case %q: mountPoint %d not found", test.name, test.id)
		}
	}
}

func TestBadParseMountInfo(t *testing.T) {
	tests := []struct {
		info         string
		name         string
		id           int
		expectedInfo *MountInfo
		error        string
	}{
		{
			`224 62 253:0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered`,
			"good major:minor field",
			224,
			&MountInfo{
				ID:             224,
				ParentID:       62,
				Major:          253,
				Minor:          0,
				Root:           "/var/lib/docker/devicemapper/test/shared",
				Source:         "/dev/mapper/ssd-root",
				MountPoint:     "/var/lib/docker/devicemapper/test/shared",
				OptionalFields: []string{"master:1", "shared:44"},
				FsType:         "ext4",
				MountOptions:   []string{"rw", "relatime"},
				SuperOptions:   []string{"rw", "seclabel", "data=ordered"},
			},
			"",
		},
		{
			`224 62 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered`,
			"missing major:minor field",
			224,
			nil,
			`parsing '224 62 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered' failed: unexpected minor:major pair [/var/lib/docker/devicemapper/test/shared]`,
		},
		{
			`224 62 :0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered`,
			"empty major field",
			224,
			nil,
			`parsing '' failed: unable to parse major device id, err:strconv.Atoi: parsing "": invalid syntax`,
		},
		{
			`224 62 253: /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered`,
			"empty minor field",
			224,
			nil,
			`parsing '' failed: unable to parse minor device id, err:strconv.Atoi: parsing "": invalid syntax`,
		},
		{
			`224 62 foo:0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered`,
			"alphabet in major field",
			224,
			nil,
			`parsing 'foo' failed: unable to parse major device id, err:strconv.Atoi: parsing "foo": invalid syntax`,
		},
		{
			`224 62 253:bar /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered`,
			"alphabet in minor field",
			224,
			nil,
			`parsing 'bar' failed: unable to parse minor device id, err:strconv.Atoi: parsing "bar": invalid syntax`,
		},
	}

	for _, test := range tests {
		tempDir, filename, err := writeFile(test.info)
		if err != nil {
			t.Fatalf("cannot create temporary file: %v", err)
		}
		defer os.RemoveAll(tempDir)

		infos, err := ParseMountInfo(filename)
		if err != nil {
			if err.Error() != test.error {
				t.Errorf("Test case %q:\n expected error: %+v\n got:      %+v", test.name, test.error, err.Error())
			}
			continue
		}

		found := false
		for _, info := range infos {
			if info.ID == test.id {
				found = true
				if !reflect.DeepEqual(info, *test.expectedInfo) {
					t.Errorf("Test case %q:\n expected: %+v\n got:      %+v", test.name, test.expectedInfo, info)
				}
				break
			}
		}
		if !found {
			t.Errorf("Test case %q: mountPoint %d not found", test.name, test.id)
		}
	}
}
