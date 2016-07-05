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
	"runtime"
	"testing"

	"k8s.io/kubernetes/pkg/util/exec"
)

type ErrorMounter struct {
	*FakeMounter
	errIndex int
	err      []error
}

func (mounter *ErrorMounter) Mount(source string, target string, fstype string, options []string) error {
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
	tests := []struct {
		description   string
		fstype        string
		mountOptions  []string
		execScripts   []ExecArgs
		mountErrs     []error
		expectedError error
	}{
		{
			description:  "Test a read only mount",
			fstype:       "ext4",
			mountOptions: []string{"ro"},
		},
		{
			description: "Test a normal mount",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
			},
		},
		{
			description: "Test 'fsck' fails with exit status 4",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", &exec.FakeExitError{Status: 4}},
			},
			expectedError: fmt.Errorf("'fsck' found errors on device /dev/foo but could not correct them: ."),
		},
		{
			description: "Test 'fsck' fails with exit status 1 (errors found and corrected)",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", &exec.FakeExitError{Status: 1}},
			},
		},
		{
			description: "Test 'fsck' fails with exit status other than 1 and 4 (likely unformatted device)",
			fstype:      "ext4",
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", &exec.FakeExitError{Status: 8}},
			},
		},
		{
			description: "Test that 'lsblk' is called and fails",
			fstype:      "ext4",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "ext4", nil},
			},
			expectedError: fmt.Errorf("unknown filesystem type '(null)'"),
		},
		{
			description: "Test that 'lsblk' is called and confirms unformatted disk, format fails",
			fstype:      "ext4",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext4", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", fmt.Errorf("formatting failed")},
			},
			expectedError: fmt.Errorf("formatting failed"),
		},
		{
			description: "Test that 'lsblk' is called and confirms unformatted disk, format passes, second mount fails",
			fstype:      "ext4",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'"), fmt.Errorf("Still cannot mount")},
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext4", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", nil},
			},
			expectedError: fmt.Errorf("Still cannot mount"),
		},
		{
			description: "Test that 'lsblk' is called and confirms unformatted disk, format passes, second mount passes",
			fstype:      "ext4",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'"), nil},
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext4", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", nil},
			},
			expectedError: nil,
		},
		{
			description: "Test that 'lsblk' is called and confirms unformatted disk, format passes, second mount passes with ext3",
			fstype:      "ext3",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'"), nil},
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext3", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", nil},
			},
			expectedError: nil,
		},
		{
			description: "test that none ext4 fs does not get called with ext4 options.",
			fstype:      "xfs",
			mountErrs:   []error{fmt.Errorf("unknown filesystem type '(null)'"), nil},
			execScripts: []ExecArgs{
				{"fsck", []string{"-a", "/dev/foo"}, "", nil},
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.xfs", []string{"/dev/foo"}, "", nil},
			},
			expectedError: nil,
		},
	}

	for _, test := range tests {
		commandScripts := []exec.FakeCommandAction{}
		for _, expected := range test.execScripts {
			ecmd := expected.command
			eargs := expected.args
			output := expected.output
			err := expected.err
			commandScript := func(cmd string, args ...string) exec.Cmd {
				if cmd != ecmd {
					t.Errorf("Unexpected command %s. Expecting %s", cmd, ecmd)
				}

				for j := range args {
					if args[j] != eargs[j] {
						t.Errorf("Unexpected args %v. Expecting %v", args, eargs)
					}
				}
				fake := exec.FakeCmd{
					CombinedOutputScript: []exec.FakeCombinedOutputAction{
						func() ([]byte, error) { return []byte(output), err },
					},
				}
				return exec.InitFakeCmd(&fake, cmd, args...)
			}
			commandScripts = append(commandScripts, commandScript)
		}

		fake := exec.FakeExec{
			CommandScript: commandScripts,
		}

		fakeMounter := ErrorMounter{&FakeMounter{}, 0, test.mountErrs}
		mounter := SafeFormatAndMount{
			Interface: &fakeMounter,
			Runner:    &fake,
		}

		device := "/dev/foo"
		dest := "/mnt/bar"
		err := mounter.FormatAndMount(device, dest, test.fstype, test.mountOptions)
		if test.expectedError == nil {
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
			if err == nil || test.expectedError.Error() != err.Error() {
				t.Errorf("test \"%s\" unexpected error: \n          [%v]. \nExpecting [%v]", test.description, err, test.expectedError)
			}
		}
	}
}
