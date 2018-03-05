// +build linux

/*
Copyright 2017 The Kubernetes Authors.

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
	"path"
	"strconv"
	"testing"
)

func TestParseFindMnt(t *testing.T) {
	tests := []struct {
		input       string
		target      string
		expectError bool
	}{
		{
			// standard mount name, e.g. for AWS
			"/var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/us-east-1d/vol-020f82b0759f72389 ext4\n",
			"/var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/us-east-1d/vol-020f82b0759f72389",
			false,
		},
		{
			// mount name with space, e.g. vSphere
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[datastore1] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk ext2\n",
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[datastore1] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk",
			false,
		},
		{
			// hypotetic mount with several spaces
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[ d a t a s t o r e 1 ] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk ext2\n",
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[ d a t a s t o r e 1 ] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk",
			false,
		},
		{
			// invalid output - no filesystem type
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/blabla",
			"",
			true,
		},
	}

	for i, test := range tests {
		target, err := parseFindMnt(test.input)
		if test.expectError && err == nil {
			t.Errorf("test %d expected error, got nil", i)
		}
		if !test.expectError && err != nil {
			t.Errorf("test %d returned error: %s", i, err)
		}
		if target != test.target {
			t.Errorf("test %d expected %q, got %q", i, test.target, target)
		}
	}
}

func TestGetPidOnHost(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "get_pid_on_host_tests")
	if err != nil {
		t.Fatalf(err.Error())
	}
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name        string
		procFile    string
		expectedPid int
		expectError bool
	}{
		{
			name: "valid status file",
			procFile: `Name:	cat
Umask:	0002
State:	R (running)
Tgid:	15041
Ngid:	0
Pid:	15041
PPid:	22699
TracerPid:	0
Uid:	1000	1000	1000	1000
Gid:	1000	1000	1000	1000
FDSize:	256
Groups:	10 135 156 157 158 973 984 1000 1001
NStgid:	15041
NSpid:	15041
NSpgid:	15041
NSsid:	22699
VmPeak:	  115016 kB
VmSize:	  115016 kB
VmLck:	       0 kB
VmPin:	       0 kB
VmHWM:	     816 kB
VmRSS:	     816 kB
RssAnon:	      64 kB
RssFile:	     752 kB
RssShmem:	       0 kB
VmData:	     312 kB
VmStk:	     136 kB
VmExe:	      32 kB
VmLib:	    2060 kB
VmPTE:	      44 kB
VmPMD:	      12 kB
VmSwap:	       0 kB
HugetlbPages:	       0 kB
Threads:	1
SigQ:	2/60752
SigPnd:	0000000000000000
ShdPnd:	0000000000000000
SigBlk:	0000000000000000
SigIgn:	0000000000000000
SigCgt:	0000000000000000
CapInh:	0000000000000000
CapPrm:	0000000000000000
CapEff:	0000000000000000
CapBnd:	0000003fffffffff
CapAmb:	0000000000000000
NoNewPrivs:	0
Seccomp:	0
Cpus_allowed:	ff
Cpus_allowed_list:	0-7
Mems_allowed:	00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000001
Mems_allowed_list:	0
voluntary_ctxt_switches:	0
nonvoluntary_ctxt_switches:	0
`,
			expectedPid: 15041,
		},
		{
			name: "no Pid:",
			procFile: `Name:	cat
Umask:	0002
State:	R (running)
Tgid:	15041
Ngid:	0
PPid:	22699
`,
			expectedPid: 0,
			expectError: true,
		},
		{
			name: "invalid Pid:",
			procFile: `Name:	cat
Umask:	0002
State:	R (running)
Tgid:	15041
Ngid:	0
Pid:	invalid
PPid:	22699
`,
			expectedPid: 0,
			expectError: true,
		},
	}

	for i, test := range tests {
		filename := path.Join(tempDir, strconv.Itoa(i))
		err := ioutil.WriteFile(filename, []byte(test.procFile), 0666)
		if err != nil {
			t.Fatalf(err.Error())
		}
		mounter := NsenterMounter{}
		pid, err := mounter.getPidOnHost(filename)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error: %s", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error, got none", test.name)
		}
		if pid != test.expectedPid {
			t.Errorf("Test %q: expected pid %d, got %d", test.name, test.expectedPid, pid)
		}
	}
}
