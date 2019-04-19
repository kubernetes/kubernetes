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

package exec

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/util/mount"
)

var (
	sourcePath      = "/mnt/srv"
	destinationPath = "/mnt/dst"
	fsType          = "xfs"
	mountOptions    = []string{"vers=1", "foo=bar"}
)

func TestMount(t *testing.T) {
	exec := mount.NewFakeExec(func(cmd string, args ...string) ([]byte, error) {
		if cmd != "mount" {
			t.Errorf("expected mount command, got %q", cmd)
		}
		// mount -t fstype -o options source target
		expectedArgs := []string{"-t", fsType, "-o", strings.Join(mountOptions, ","), sourcePath, destinationPath}
		if !reflect.DeepEqual(expectedArgs, args) {
			t.Errorf("expected arguments %q, got %q", strings.Join(expectedArgs, " "), strings.Join(args, " "))
		}
		return nil, nil
	})

	wrappedMounter := &fakeMounter{FakeMounter: &mount.FakeMounter{}, t: t}
	mounter := NewExecMounter(exec, wrappedMounter)

	mounter.Mount(sourcePath, destinationPath, fsType, mountOptions)
}

func TestBindMount(t *testing.T) {
	cmdCount := 0
	exec := mount.NewFakeExec(func(cmd string, args ...string) ([]byte, error) {
		cmdCount++
		if cmd != "mount" {
			t.Errorf("expected mount command, got %q", cmd)
		}
		var expectedArgs []string
		switch cmdCount {
		case 1:
			// mount -t fstype -o "bind" source target
			expectedArgs = []string{"-t", fsType, "-o", "bind", sourcePath, destinationPath}
		case 2:
			// mount -t fstype -o "remount,opts" source target
			expectedArgs = []string{"-t", fsType, "-o", "bind,remount," + strings.Join(mountOptions, ","), sourcePath, destinationPath}
		}
		if !reflect.DeepEqual(expectedArgs, args) {
			t.Errorf("expected arguments %q, got %q", strings.Join(expectedArgs, " "), strings.Join(args, " "))
		}
		return nil, nil
	})

	wrappedMounter := &fakeMounter{FakeMounter: &mount.FakeMounter{}, t: t}
	mounter := NewExecMounter(exec, wrappedMounter)
	bindOptions := append(mountOptions, "bind")
	mounter.Mount(sourcePath, destinationPath, fsType, bindOptions)
}

func TestUnmount(t *testing.T) {
	exec := mount.NewFakeExec(func(cmd string, args ...string) ([]byte, error) {
		if cmd != "umount" {
			t.Errorf("expected unmount command, got %q", cmd)
		}
		// unmount $target
		expectedArgs := []string{destinationPath}
		if !reflect.DeepEqual(expectedArgs, args) {
			t.Errorf("expected arguments %q, got %q", strings.Join(expectedArgs, " "), strings.Join(args, " "))
		}
		return nil, nil
	})

	wrappedMounter := &fakeMounter{&mount.FakeMounter{}, t}
	mounter := NewExecMounter(exec, wrappedMounter)

	mounter.Unmount(destinationPath)
}

/* Fake wrapped mounter */
type fakeMounter struct {
	*mount.FakeMounter
	t *testing.T
}

func (fm *fakeMounter) Mount(source string, target string, fstype string, options []string) error {
	// Mount() of wrapped mounter should never be called. We call exec instead.
	fm.t.Errorf("Unexpected wrapped mount call")
	return fmt.Errorf("Unexpected wrapped mount call")
}

func (fm *fakeMounter) Unmount(target string) error {
	// umount() of wrapped mounter should never be called. We call exec instead.
	fm.t.Errorf("Unexpected wrapped mount call")
	return fmt.Errorf("Unexpected wrapped mount call")
}
