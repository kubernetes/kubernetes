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

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
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

func (l *localFakeMounter) CanMount() error {
	return nil
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

			defer os.RemoveAll(tmpDir)

			info, err := os.Lstat(tmpDir)
			if err != nil {
				t.Fatalf("error reading permission of tmpdir: %v", err)
			}

			stat, ok := info.Sys().(*syscall.Stat_t)
			if !ok || stat == nil {
				t.Fatalf("error reading permission stats for tmpdir: %s", tmpDir)
			}

			gid := stat.Gid

			var expectedGid int64

			if test.gidOwnerMatch {
				expectedGid = int64(gid)
			} else {
				expectedGid = int64(gid + 3000)
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
			ok = skipPermissionChange(mounter, &expectedGid, test.fsGroupChangePolicy)
			if ok != test.skipPermssion {
				t.Errorf("for %s expected skipPermission to be %v got %v", test.description, test.skipPermssion, ok)
			}

		})
	}
}

func TestSetVolumeOwnership(t *testing.T) {
	always := v1.FSGroupChangeAlways
	onrootMismatch := v1.FSGroupChangeOnRootMismatch
	expectedMask := rwMask | os.ModeSetgid | execMask

	tests := []struct {
		description         string
		fsGroupChangePolicy *v1.PodFSGroupChangePolicy
		setupFunc           func(path string) error
		assertFunc          func(path string) error
		featureGate         bool
	}{
		{
			description:         "featuregate=on, fsgroupchangepolicy=always",
			fsGroupChangePolicy: &always,
			featureGate:         true,
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
			featureGate:         true,
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
			featureGate:         true,
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
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConfigurableFSGroupPolicy, test.featureGate)()
			tmpDir, err := utiltesting.MkTmpdir("volume_linux_ownership")
			if err != nil {
				t.Fatalf("error creating temp dir: %v", err)
			}

			defer os.RemoveAll(tmpDir)
			info, err := os.Lstat(tmpDir)
			if err != nil {
				t.Fatalf("error reading permission of tmpdir: %v", err)
			}

			stat, ok := info.Sys().(*syscall.Stat_t)
			if !ok || stat == nil {
				t.Fatalf("error reading permission stats for tmpdir: %s", tmpDir)
			}

			var expectedGid int64 = int64(stat.Gid)
			err = test.setupFunc(tmpDir)
			if err != nil {
				t.Errorf("for %s error running setup with: %v", test.description, err)
			}

			mounter := &localFakeMounter{path: tmpDir}
			err = SetVolumeOwnership(mounter, &expectedGid, test.fsGroupChangePolicy, nil)
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
