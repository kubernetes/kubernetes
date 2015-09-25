/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	tests := []struct {
		fstype        string
		mountOptions  []string
		execScripts   []ExecArgs
		mountErrs     []error
		expectedError error
	}{
		{ // Test a read only mount
			fstype:       "ext4",
			mountOptions: []string{"ro"},
		},
		{ // Test a normal mount
			fstype: "ext4",
		},

		{ // Test that 'file' is called and fails
			fstype:    "ext4",
			mountErrs: []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "ext4", nil},
			},
			expectedError: fmt.Errorf("unknown filesystem type '(null)'"),
		},
		{ // Test that 'file' is called and confirms unformatted disk, format fails
			fstype:    "ext4",
			mountErrs: []error{fmt.Errorf("unknown filesystem type '(null)'")},
			execScripts: []ExecArgs{
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext4", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", fmt.Errorf("formatting failed")},
			},
			expectedError: fmt.Errorf("formatting failed"),
		},
		{ // Test that 'file' is called and confirms unformatted disk, format passes, second mount fails
			fstype:    "ext4",
			mountErrs: []error{fmt.Errorf("unknown filesystem type '(null)'"), fmt.Errorf("Still cannot mount")},
			execScripts: []ExecArgs{
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext4", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", nil},
			},
			expectedError: fmt.Errorf("Still cannot mount"),
		},
		{ // Test that 'file' is called and confirms unformatted disk, format passes, second mount passes
			fstype:    "ext4",
			mountErrs: []error{fmt.Errorf("unknown filesystem type '(null)'"), nil},
			execScripts: []ExecArgs{
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext4", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", nil},
			},
			expectedError: nil,
		},
		{ // Test that 'file' is called and confirms unformatted disk, format passes, second mount passes with ext3
			fstype:    "ext3",
			mountErrs: []error{fmt.Errorf("unknown filesystem type '(null)'"), nil},
			execScripts: []ExecArgs{
				{"lsblk", []string{"-nd", "-o", "FSTYPE", "/dev/foo"}, "", nil},
				{"mkfs.ext3", []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", "/dev/foo"}, "", nil},
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
		err := mounter.Mount(device, dest, test.fstype, test.mountOptions)
		if test.expectedError == nil {
			if err != nil {
				t.Errorf("unexpected non-error: %v", err)
			}

			// Check that something was mounted on the directory
			isNotMountPoint, err := fakeMounter.IsLikelyNotMountPoint(dest)
			if err != nil || isNotMountPoint {
				t.Errorf("the directory was not mounted")
			}

			//check that the correct device was mounted
			mountedDevice, _, err := GetDeviceNameFromMount(fakeMounter.FakeMounter, dest)
			if err != nil || mountedDevice != device {
				t.Errorf("the correct device was not mounted")
			}
		} else {
			if err == nil || test.expectedError.Error() != err.Error() {
				t.Errorf("unexpected error: %v. Expecting %v", err, test.expectedError)
			}
		}
	}
}
