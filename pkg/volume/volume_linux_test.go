//go:build linux
// +build linux

/*
Copyright 2020 The Kubernetes Authors.

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

package volume

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	utiltesting "k8s.io/client-go/util/testing"
)

type localFakeMounter struct {
	path       string
	attributes Attributes
}

func (l *localFakeMounter) GetPath() string {
	return l.path
}

func (l *localFakeMounter) GetAttributes() Attributes {
	return l.attributes
}

func (l *localFakeMounter) SetUp(mounterArgs MounterArgs) error {
	return nil
}

func (l *localFakeMounter) SetUpAt(dir string, mounterArgs MounterArgs) error {
	return nil
}

func (l *localFakeMounter) GetMetrics() (*Metrics, error) {
	return nil, nil
}

func TestSkipPermissionChange(t *testing.T) {
	always := v1.FSGroupChangeAlways
	onrootMismatch := v1.FSGroupChangeOnRootMismatch
	tests := []struct {
		description         string
		fsGroupChangePolicy *v1.PodFSGroupChangePolicy
		gidOwnerMatch       bool
		permissionMatch     bool
		sgidMatch           bool
		skipPermssion       bool
	}{
		{
			description:   "skippermission=false, policy=nil",
			skipPermssion: false,
		},
		{
			description:         "skippermission=false, policy=always",
			fsGroupChangePolicy: &always,
			skipPermssion:       false,
		},
		{
			description:         "skippermission=false, policy=always, gidmatch=true",
			fsGroupChangePolicy: &always,
			skipPermssion:       false,
			gidOwnerMatch:       true,
		},
		{
			description:         "skippermission=false, policy=nil, gidmatch=true",
			fsGroupChangePolicy: nil,
			skipPermssion:       false,
			gidOwnerMatch:       true,
		},
		{
			description:         "skippermission=false, policy=onrootmismatch, gidmatch=false",
			fsGroupChangePolicy: &onrootMismatch,
			gidOwnerMatch:       false,
			skipPermssion:       false,
		},
		{
			description:         "skippermission=false, policy=onrootmismatch, gidmatch=true, permmatch=false",
			fsGroupChangePolicy: &onrootMismatch,
			gidOwnerMatch:       true,
			permissionMatch:     false,
			skipPermssion:       false,
		},
		{
			description:         "skippermission=false, policy=onrootmismatch, gidmatch=true, permmatch=true",
			fsGroupChangePolicy: &onrootMismatch,
			gidOwnerMatch:       true,
			permissionMatch:     true,
			skipPermssion:       false,
		},
		{
			description:         "skippermission=false, policy=onrootmismatch, gidmatch=true, permmatch=true, sgidmatch=true",
			fsGroupChangePolicy: &onrootMismatch,
			gidOwnerMatch:       true,
			permissionMatch:     true,
			sgidMatch:           true,
			skipPermssion:       true,
		},
	}

	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("volume_linux_test")
			if err != nil {
				t.Fatalf("error creating temp dir: %v", err)
			}
			defer func() {
				err := os.RemoveAll(tmpDir)
				if err != nil {
					t.Fatalf("error removing tmpDir %s: %v", tmpDir, err)
				}
			}()

			info, err := os.Lstat(tmpDir)
			if err != nil {
				t.Fatalf("error reading permission of tmpdir: %v", err)
			}

			stat, ok := info.Sys().(*syscall.Stat_t)
			if !ok || stat == nil {
				t.Fatalf("error reading permission stats for tmpdir: %s", tmpDir)
			}

			gid := stat.Gid

			var expectedGID int64

			if test.gidOwnerMatch {
				expectedGID = int64(gid)
			} else {
				expectedGID = int64(gid + 3000)
			}

			mask := rwMask

			if test.permissionMatch {
				mask |= execMask

			}
			if test.sgidMatch {
				mask |= os.ModeSetgid
				mask = info.Mode() | mask
			} else {
				nosgidPerm := info.Mode() &^ os.ModeSetgid
				mask = nosgidPerm | mask
			}

			err = os.Chmod(tmpDir, mask)
			if err != nil {
				t.Errorf("Chmod failed on %v: %v", tmpDir, err)
			}

			mounter := &localFakeMounter{path: tmpDir}
			ok = skipPermissionChange(mounter, tmpDir, &expectedGID, test.fsGroupChangePolicy)
			if ok != test.skipPermssion {
				t.Errorf("for %s expected skipPermission to be %v got %v", test.description, test.skipPermssion, ok)
			}

		})
	}
}

func TestSetVolumeOwnershipMode(t *testing.T) {
	always := v1.FSGroupChangeAlways
	onrootMismatch := v1.FSGroupChangeOnRootMismatch
	expectedMask := rwMask | os.ModeSetgid | execMask

	tests := []struct {
		description         string
		fsGroupChangePolicy *v1.PodFSGroupChangePolicy
		setupFunc           func(path string) error
		assertFunc          func(path string) error
	}{
		{
			description:         "featuregate=on, fsgroupchangepolicy=always",
			fsGroupChangePolicy: &always,
			setupFunc: func(path string) error {
				info, err := os.Lstat(path)
				if err != nil {
					return err
				}
				// change mode of root folder to be right
				err = os.Chmod(path, info.Mode()|expectedMask)
				if err != nil {
					return err
				}

				// create a subdirectory with invalid permissions
				rogueDir := filepath.Join(path, "roguedir")
				nosgidPerm := info.Mode() &^ os.ModeSetgid
				err = os.Mkdir(rogueDir, nosgidPerm)
				if err != nil {
					return err
				}
				return nil
			},
			assertFunc: func(path string) error {
				rogueDir := filepath.Join(path, "roguedir")
				hasCorrectPermissions := verifyDirectoryPermission(rogueDir, false /*readOnly*/)
				if !hasCorrectPermissions {
					return fmt.Errorf("invalid permissions on %s", rogueDir)
				}
				return nil
			},
		},
		{
			description:         "featuregate=on, fsgroupchangepolicy=onrootmismatch,rootdir=validperm",
			fsGroupChangePolicy: &onrootMismatch,
			setupFunc: func(path string) error {
				info, err := os.Lstat(path)
				if err != nil {
					return err
				}
				// change mode of root folder to be right
				err = os.Chmod(path, info.Mode()|expectedMask)
				if err != nil {
					return err
				}

				// create a subdirectory with invalid permissions
				rogueDir := filepath.Join(path, "roguedir")
				err = os.Mkdir(rogueDir, rwMask)
				if err != nil {
					return err
				}
				return nil
			},
			assertFunc: func(path string) error {
				rogueDir := filepath.Join(path, "roguedir")
				hasCorrectPermissions := verifyDirectoryPermission(rogueDir, false /*readOnly*/)
				if hasCorrectPermissions {
					return fmt.Errorf("invalid permissions on %s", rogueDir)
				}
				return nil
			},
		},
		{
			description:         "featuregate=on, fsgroupchangepolicy=onrootmismatch,rootdir=invalidperm",
			fsGroupChangePolicy: &onrootMismatch,
			setupFunc: func(path string) error {
				// change mode of root folder to be right
				err := os.Chmod(path, 0770)
				if err != nil {
					return err
				}

				// create a subdirectory with invalid permissions
				rogueDir := filepath.Join(path, "roguedir")
				err = os.Mkdir(rogueDir, rwMask)
				if err != nil {
					return err
				}
				return nil
			},
			assertFunc: func(path string) error {
				rogueDir := filepath.Join(path, "roguedir")
				hasCorrectPermissions := verifyDirectoryPermission(rogueDir, false /*readOnly*/)
				if !hasCorrectPermissions {
					return fmt.Errorf("invalid permissions on %s", rogueDir)
				}
				return nil
			},
		},
	}

	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("volume_linux_ownership")
			if err != nil {
				t.Fatalf("error creating temp dir: %v", err)
			}

			defer func() {
				err := os.RemoveAll(tmpDir)
				if err != nil {
					t.Fatalf("error removing tmpDir %s: %v", tmpDir, err)
				}
			}()

			info, err := os.Lstat(tmpDir)
			if err != nil {
				t.Fatalf("error reading permission of tmpdir: %v", err)
			}

			stat, ok := info.Sys().(*syscall.Stat_t)
			if !ok || stat == nil {
				t.Fatalf("error reading permission stats for tmpdir: %s", tmpDir)
			}

			var expectedGID = int64(stat.Gid)
			err = test.setupFunc(tmpDir)
			if err != nil {
				t.Errorf("for %s error running setup with: %v", test.description, err)
			}

			mounter := &localFakeMounter{path: "FAKE_DIR_DOESNT_EXIST"} // SetVolumeOwnership() must rely on tmpDir
			ownershipChanger := NewVolumeOwnership(mounter, tmpDir, &expectedGID, test.fsGroupChangePolicy, nil)
			err = ownershipChanger.ChangePermissions()
			if err != nil {
				t.Errorf("for %s error changing ownership with: %v", test.description, err)
			}
			err = test.assertFunc(tmpDir)
			if err != nil {
				t.Errorf("for %s error verifying permissions with: %v", test.description, err)
			}
		})
	}
}

func TestProgressTracking(t *testing.T) {
	alwaysApplyPolicy := v1.FSGroupChangeAlways
	var expectedGID int64 = 9999

	//  capture original variable
	originalfilePermissionChangeFunc := filePermissionChangeFunc
	originalProgressReportDuration := progressReportDuration
	originalfirstEventReportDuration := firstEventReportDuration

	var permissionSleepDuration = 5 * time.Millisecond

	// Override how often progress is reported
	progressReportDuration = 200 * time.Millisecond
	// Override when first event about progress is reported
	firstEventReportDuration = 50 * time.Millisecond

	// Override how permission change is applied, so as to artificially slow
	// permission change
	filePermissionChangeFunc = func(filename string, fsGroup *int64, readonly bool, info os.FileInfo) error {
		time.Sleep(permissionSleepDuration)
		originalfilePermissionChangeFunc(filename, fsGroup, readonly, info)
		return nil
	}
	t.Cleanup(func() {
		filePermissionChangeFunc = originalfilePermissionChangeFunc
		progressReportDuration = originalProgressReportDuration
		firstEventReportDuration = originalfirstEventReportDuration
	})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name                             string
		filePermissionChangeTimeDuration time.Duration
		totalWaitTime                    time.Duration
		currentPod                       *v1.Pod
		expectedEvents                   []string
	}{
		{
			name:                             "When permission change finishes quickly, no events should be logged",
			filePermissionChangeTimeDuration: 30 * time.Millisecond,
			totalWaitTime:                    1 * time.Second,
			currentPod:                       pod,
			expectedEvents:                   []string{},
		},
		{
			name:                             "When no pod is specified, no events should be logged",
			filePermissionChangeTimeDuration: 300 * time.Millisecond,
			totalWaitTime:                    1 * time.Second,
			currentPod:                       nil,
			expectedEvents:                   []string{},
		},
		{
			name:                             "When permission change takes loo long and pod is specified",
			filePermissionChangeTimeDuration: 300 * time.Millisecond,
			totalWaitTime:                    1 * time.Second,
			currentPod:                       pod,
			expectedEvents: []string{
				"Warning VolumePermissionChangeInProgress Setting volume ownership for pod1uid/volumes/faketype is taking longer than expected, consider using OnRootMismatch - https://kubernetes.io/docs/tasks/configure-pod-container/security-context/#configure-volume-permission-and-ownership-change-policy-for-pods",
				"Warning VolumePermissionChangeInProgress Setting volume ownership for pod1uid/volumes/faketype, processed 1 files.",
			},
		},
	}

	for i := range tests {
		tc := tests[i]
		t.Run(tc.name, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("volume_linux_ownership")
			if err != nil {
				t.Fatalf("error creating temp dir: %v", err)
			}
			podUID := "placeholder"
			if tc.currentPod != nil {
				podUID = string(tc.currentPod.UID)
			}
			volumePath := filepath.Join(tmpDir, podUID, "volumes", "faketype")
			err = os.MkdirAll(volumePath, 0770)
			if err != nil {
				t.Fatalf("error creating volumePath %s: %v", volumePath, err)
			}
			defer func() {
				err := os.RemoveAll(tmpDir)
				if err != nil {
					t.Fatalf("error removing tmpDir %s: %v", tmpDir, err)
				}
			}()

			mounter := &localFakeMounter{path: "FAKE_DIR_DOESNT_EXIST"} // SetVolumeOwnership() must rely on tmpDir

			fakeRecorder := record.NewFakeRecorder(100)
			recordedEvents := []string{}

			// Set how long file permission change takes
			permissionSleepDuration = tc.filePermissionChangeTimeDuration

			ownershipChanger := NewVolumeOwnership(mounter, volumePath, &expectedGID, &alwaysApplyPolicy, nil)
			if tc.currentPod != nil {
				ownershipChanger.AddProgressNotifier(tc.currentPod, fakeRecorder)
			}
			err = ownershipChanger.ChangePermissions()
			if err != nil {
				t.Errorf("unexpected error: %+v", err)
			}
			time.Sleep(tc.totalWaitTime)
			actualEventCount := len(fakeRecorder.Events)
			if len(tc.expectedEvents) == 0 && actualEventCount != len(tc.expectedEvents) {
				t.Errorf("expected 0 events got %d", actualEventCount)
			}

			for range actualEventCount {
				event := <-fakeRecorder.Events
				recordedEvents = append(recordedEvents, event)
			}

			for i, event := range tc.expectedEvents {
				if event != recordedEvents[i] {
					t.Errorf("expected event %d to be %s, got: %s", i, event, recordedEvents[i])
				}
			}
		})
	}
}

// verifyDirectoryPermission checks if given path has directory permissions
// that is expected by k8s. If returns true if it does otherwise false
func verifyDirectoryPermission(path string, readonly bool) bool {
	info, err := os.Lstat(path)
	if err != nil {
		return false
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat == nil {
		return false
	}
	unixPerms := rwMask

	if readonly {
		unixPerms = roMask
	}

	unixPerms |= execMask
	filePerm := info.Mode().Perm()
	if (unixPerms&filePerm == unixPerms) && (info.Mode()&os.ModeSetgid != 0) {
		return true
	}
	return false
}

func TestSetVolumeOwnershipOwner(t *testing.T) {
	fsGroup := int64(3000)
	currentUid := os.Geteuid()
	if currentUid != 0 {
		t.Skip("running as non-root")
	}
	currentGID := os.Getgid()

	tests := []struct {
		description string
		fsGroup     *int64
		setupFunc   func(path string) error
		assertFunc  func(path string) error
	}{
		{
			description: "fsGroup=nil",
			fsGroup:     nil,
			setupFunc: func(path string) error {
				filename := filepath.Join(path, "file.txt")
				file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0755)
				if err != nil {
					return err
				}
				file.Close()
				return nil
			},
			assertFunc: func(path string) error {
				filename := filepath.Join(path, "file.txt")
				if !verifyFileOwner(filename, currentUid, currentGID) {
					return fmt.Errorf("invalid owner on %s", filename)
				}
				return nil
			},
		},
		{
			description: "*fsGroup=3000",
			fsGroup:     &fsGroup,
			setupFunc: func(path string) error {
				filename := filepath.Join(path, "file.txt")
				file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0755)
				if err != nil {
					return err
				}
				file.Close()
				return nil
			},
			assertFunc: func(path string) error {
				filename := filepath.Join(path, "file.txt")
				if !verifyFileOwner(filename, currentUid, int(fsGroup)) {
					return fmt.Errorf("invalid owner on %s", filename)
				}
				return nil
			},
		},
		{
			description: "symlink",
			fsGroup:     &fsGroup,
			setupFunc: func(path string) error {
				filename := filepath.Join(path, "file.txt")
				file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0755)
				if err != nil {
					return err
				}
				file.Close()

				symname := filepath.Join(path, "file_link.txt")
				err = os.Symlink(filename, symname)
				if err != nil {
					return err
				}

				return nil
			},
			assertFunc: func(path string) error {
				symname := filepath.Join(path, "file_link.txt")
				if !verifyFileOwner(symname, currentUid, int(fsGroup)) {
					return fmt.Errorf("invalid owner on %s", symname)
				}
				return nil
			},
		},
	}

	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("volume_linux_ownership")
			if err != nil {
				t.Fatalf("error creating temp dir: %v", err)
			}

			defer func() {
				err := os.RemoveAll(tmpDir)
				if err != nil {
					t.Fatalf("error removing tmpDir %s: %v", tmpDir, err)
				}
			}()

			err = test.setupFunc(tmpDir)
			if err != nil {
				t.Errorf("for %s error running setup with: %v", test.description, err)
			}

			mounter := &localFakeMounter{path: tmpDir}
			always := v1.FSGroupChangeAlways
			ownershipChanger := NewVolumeOwnership(mounter, tmpDir, test.fsGroup, &always, nil)
			err = ownershipChanger.ChangePermissions()
			if err != nil {
				t.Errorf("for %s error changing ownership with: %v", test.description, err)
			}
			err = test.assertFunc(tmpDir)
			if err != nil {
				t.Errorf("for %s error verifying permissions with: %v", test.description, err)
			}
		})
	}
}

// verifyFileOwner checks if given path is owned by uid and gid.
// It returns true if it is otherwise false.
func verifyFileOwner(path string, uid, gid int) bool {
	info, err := os.Lstat(path)
	if err != nil {
		return false
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat == nil {
		return false
	}

	if int(stat.Uid) != uid || int(stat.Gid) != gid {
		return false
	}

	return true
}
