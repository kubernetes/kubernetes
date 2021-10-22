/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"strings"
	"testing"

	"k8s.io/utils/exec"
	testingexec "k8s.io/utils/exec/testing"
)

type ErrorMounter struct {
	*FakeMounter
	errIndex int
	err      []error
}

func (mounter *ErrorMounter) Mount(source string, target string, fstype string, options []string) error {
	return mounter.MountSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
}

func (mounter *ErrorMounter) MountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	i := mounter.errIndex
	mounter.errIndex++
	if mounter.err != nil && mounter.err[i] != nil {
		return mounter.err[i]
	}
	return mounter.FakeMounter.Mount(source, target, fstype, options)
}

type ExecArgs struct {
	command string
	args    []string
	output  string
	err     error
}

func TestSafeFormatAndMount(t *testing.T) {
	if runtime.GOOS == "darwin" || runtime.GOOS == "windows" {
		t.Skipf("not supported on GOOS=%s", runtime.GOOS)
	}
	mntDir, err := ioutil.TempDir(os.TempDir(), "mount")
	if err != nil {
		t.Fatalf("failed to create tmp dir: %v", err)
	}
	defer os.RemoveAll(mntDir)
	tests := []struct {
		description           string
		fstype                string
		mountOptions          []string
		sensitiveMountOptions []string
		execScripts           []ExecArgs
		mountErrs             []error
		expErrorType          MountErrorType
	}{
		{
			description:  "Test a read only mount of an already formatted device",
			fstype:       "ext4",
			mountOptions: []string{"ro"},
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "DEVNAME=/dev/foo\nTYPE=ext4\n", nil},
			},
		},
		{
			description: "Test a normal mount of an already formatted device",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "DEVNAME=/dev/foo\nTYPE=ext4\n", nil},
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
			},
		},
		{
			description:  "Test a read only mount of unformatted device",
			fstype:       "ext4",
			mountOptions: []string{"ro"},
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
			},
			expErrorType: UnformattedReadOnly,
		},
		{
			description: "Test a normal mount of unformatted device",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
				{"mkfs.ext4", []string{"-F", "-m0", "/dev/foo"}, "", nil},
			},
		},
		{
			description: "Test 'fsck' fails with exit status 4",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "DEVNAME=/dev/foo\nTYPE=ext4\n", nil},
				{"fsck", []string{"-a", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 4}},
			},
			expErrorType: HasFilesystemErrors,
		},
		{
			description: "Test 'fsck' fails with exit status 1 (errors found and corrected)",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "DEVNAME=/dev/foo\nTYPE=ext4\n", nil},
				{"fsck", []string{"-a", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 1}},
			},
		},
		{
			description: "Test 'fsck' fails with exit status other than 1 and 4 (likely unformatted device)",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "DEVNAME=/dev/foo\nTYPE=ext4\n", nil},
				{"fsck", []string{"-a", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 8}},
			},
		},
		{
			description: "Test that 'blkid' is called and fails",
			fstype:      "ext4",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "DEVNAME=/dev/foo\nPTTYPE=dos\n", nil},
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
			},
			expErrorType: FilesystemMismatch,
		},
		{
			description: "Test that 'blkid' is called and confirms unformatted disk, format fails",
			fstype:      "ext4",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
				{"mkfs.ext4", []string{"-F", "-m0", "/dev/foo"}, "", fmt.Errorf("formatting failed")},
			},
			expErrorType: FormatFailed,
		},
		{
			description: "Test that 'blkid' is called and confirms unformatted disk, format passes, second mount fails",
			fstype:      "ext4",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
				{"mkfs.ext4", []string{"-F", "-m0", "/dev/foo"}, "", nil},
			},
			expErrorType: UnknownMountError,
		},
		{
			description: "Test that 'blkid' is called and confirms unformatted disk, format passes, mount passes",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
				{"mkfs.ext4", []string{"-F", "-m0", "/dev/foo"}, "", nil},
			},
		},
		{
			description: "Test that 'blkid' is called and confirms unformatted disk, format passes, mount passes with ext3",
			fstype:      "ext3",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
				{"mkfs.ext3", []string{"-F", "-m0", "/dev/foo"}, "", nil},
			},
		},
		{
			description: "test that none ext4 fs does not get called with ext4 options.",
			fstype:      "xfs",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
				{"mkfs.xfs", []string{"/dev/foo"}, "", nil},
			},
		},
		{
			description: "Test that 'blkid' is called and reports ext4 partition",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "DEVNAME=/dev/foo\nTYPE=ext4\n", nil},
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
			},
		},
		{
			description: "Test that 'blkid' is called but has some usage or other errors (an exit code of 4 is returned)",
			fstype:      "xfs",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'"), nil},
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 4}},
				{"mkfs.xfs", []string{"/dev/foo"}, "", nil},
			},
			expErrorType: GetDiskFormatFailed,
		},
		{
			description:           "Test that 'blkid' is called and confirms unformatted disk, format fails with sensitive options",
			fstype:                "ext4",
			sensitiveMountOptions: []string{"mySecret"},
			mountErrs:             []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"blkid", []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", "/dev/foo"}, "", &testingexec.FakeExitError{Status: 2}},
				{"mkfs.ext4", []string{"-F", "-m0", "/dev/foo"}, "", fmt.Errorf("formatting failed")},
			},
			expErrorType: FormatFailed,
		},
	}

	for _, test := range tests {
		fakeMounter := ErrorMounter{NewFakeMounter(nil), 0, test.mountErrs}
		fakeExec := &testingexec.FakeExec{ExactOrder: true}
		for _, script := range test.execScripts {
			fakeCmd := &testingexec.FakeCmd{}
			cmdAction := makeFakeCmd(fakeCmd, script.command, script.args...)
			outputAction := makeFakeOutput(script.output, script.err)
			fakeCmd.CombinedOutputScript = append(fakeCmd.CombinedOutputScript, outputAction)
			fakeExec.CommandScript = append(fakeExec.CommandScript, cmdAction)
		}
		mounter := SafeFormatAndMount{
			Interface: &fakeMounter,
			Exec:      fakeExec,
		}

		device := "/dev/foo"
		dest := mntDir
		var err error
		if len(test.sensitiveMountOptions) == 0 {
			err = mounter.FormatAndMount(device, dest, test.fstype, test.mountOptions)
		} else {
			err = mounter.FormatAndMountSensitive(device, dest, test.fstype, test.mountOptions, test.sensitiveMountOptions)
		}
		if len(test.expErrorType) == 0 {
			if err != nil {
				t.Errorf("test \"%s\" unexpected non-error: %v", test.description, err)
			}

			// Check that something was mounted on the directory
			isNotMountPoint, err := fakeMounter.IsLikelyNotMountPoint(dest)
			if err != nil || isNotMountPoint {
				t.Errorf("test \"%s\" the directory was not mounted", test.description)
			}

			//check that the correct device was mounted
			mountedDevice, _, err := GetDeviceNameFromMount(fakeMounter.FakeMounter, dest)
			if err != nil || mountedDevice != device {
				t.Errorf("test \"%s\" the correct device was not mounted", test.description)
			}
		} else {
			mntErr, ok := err.(MountError)
			if !ok {
				t.Errorf("mount error not of mount error type: %v", err)
			}
			if mntErr.Type != test.expErrorType {
				t.Errorf("test \"%s\" unexpected error: \n          [%v]. \nExpecting err type[%v]", test.description, err, test.expErrorType)
			}
			if len(test.sensitiveMountOptions) == 0 {
				if strings.Contains(mntErr.Error(), sensitiveOptionsRemoved) {
					t.Errorf("test \"%s\" returned an error unexpectedly containing the string %q: %v", test.description, sensitiveOptionsRemoved, err)
				}
			} else {
				if !strings.Contains(err.Error(), sensitiveOptionsRemoved) {
					t.Errorf("test \"%s\" returned an error without the string %q: %v", test.description, sensitiveOptionsRemoved, err)
				}
				for _, sensitiveOption := range test.sensitiveMountOptions {
					if strings.Contains(err.Error(), sensitiveOption) {
						t.Errorf("test \"%s\" returned an error with a sensitive string (%q): %v", test.description, sensitiveOption, err)
					}
				}
			}
		}
	}
}

func makeFakeCmd(fakeCmd *testingexec.FakeCmd, cmd string, args ...string) testingexec.FakeCommandAction {
	c := cmd
	a := args
	return func(cmd string, args ...string) exec.Cmd {
		command := testingexec.InitFakeCmd(fakeCmd, c, a...)
		return command
	}
}

func makeFakeOutput(output string, err error) testingexec.FakeAction {
	o := output
	return func() ([]byte, []byte, error) {
		return []byte(o), nil, err
	}
}
