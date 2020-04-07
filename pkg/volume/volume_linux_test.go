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
			description: "skippermission=false, policy=nil",
		},
		{
			description:         "skippermission=false, policy=always",
			fsGroupChangePolicy: &always,
		},
		{
			description:         "skippermission=false, policy=always, gidmatch=true",
			fsGroupChangePolicy: &always,
			gidOwnerMatch:       true,
		},
		{
			description:         "skippermission=false, policy=nil, gidmatch=true",
			fsGroupChangePolicy: nil,
			gidOwnerMatch:       true,
		},
		{
			description:         "skippermission=false, policy=onrootmismatch, gidmatch=false",
			fsGroupChangePolicy: &onrootMismatch,
		},
		{
			description:         "skippermission=false, policy=onrootmismatch, gidmatch=true, permmatch=false",
			fsGroupChangePolicy: &onrootMismatch,
			gidOwnerMatch:       true,
		},
		{
			description:         "skippermission=false, policy=onrootmismatch, gidmatch=true, permmatch=true",
			fsGroupChangePolicy: &onrootMismatch,
			gidOwnerMatch:       true,
			permissionMatch:     true,
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

			mask := os.FileMode(0770)

			if test.sgidMatch {
				mask |= os.ModeSetgid
			}

			if !test.permissionMatch {
				mask &= ^os.FileMode(0110)
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
	tests := []struct {
		desc            string
		fsGroup         bool
		readOnly        bool
		rootMode        os.FileMode
		dirMode         os.FileMode
		fileMode        os.FileMode
		wantRootMode    os.FileMode
		wantDirMode     os.FileMode
		wantFileMode    os.FileMode
		wantSymlinkMode os.FileMode
	}{
		{
			desc:            "no fsGroup",
			fsGroup:         false,
			readOnly:        false,
			rootMode:        0777 | os.ModeSticky | os.ModeDir,
			dirMode:         0777 | os.ModeDir,
			fileMode:        0666,
			wantRootMode:    0777 | os.ModeSticky | os.ModeDir,
			wantDirMode:     0777 | os.ModeDir,
			wantFileMode:    0666,
			wantSymlinkMode: 0777 | os.ModeSymlink,
		},
		{
			desc:            "fsGroup",
			fsGroup:         true,
			readOnly:        false,
			rootMode:        0700 | os.ModeSticky | os.ModeDir,
			dirMode:         0700 | os.ModeDir,
			fileMode:        0600,
			wantRootMode:    0770 | os.ModeSticky | os.ModeSetgid | os.ModeDir,
			wantDirMode:     0770 | os.ModeDir | os.ModeSetgid,
			wantFileMode:    0660,
			wantSymlinkMode: 0777 | os.ModeSymlink,
		},
		{
			desc:            "fsGroup, read only",
			fsGroup:         true,
			readOnly:        true,
			rootMode:        0700 | os.ModeSticky | os.ModeDir,
			dirMode:         0700 | os.ModeDir,
			fileMode:        0600,
			wantRootMode:    0550 | os.ModeSticky | os.ModeSetgid | os.ModeDir,
			wantDirMode:     0550 | os.ModeDir | os.ModeSetgid,
			wantFileMode:    0440,
			wantSymlinkMode: 0777 | os.ModeSymlink,
		},
	}

	// when i == 0, FsGroupPolicy is enabled to test new path.
	//      i == 1, FsGroupPolicy is disabled to test legacy path.
	for i := 0; i < 2; i++ {
		for _, test := range tests {
			t.Run(test.desc, func(t *testing.T) {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConfigurableFSGroupPolicy, i == 0)()

				root := utiltesting.MkTmpdirOrDie("root")
				defer os.RemoveAll(root)
				chmod(root, test.rootMode)

				dir := filepath.Join(root, "dir")
				err := os.Mkdir(dir, test.dirMode)
				if err != nil {
					t.Fatalf("error creating a dir in root dir: %v", err)
				}
				chmod(dir, test.dirMode)

				file := filepath.Join(dir, "file")
				_, err = os.Create(file)
				if err != nil {
					t.Fatalf("error creating a file in dir: %v", err)
				}
				chmod(file, test.fileMode)

				symlink := filepath.Join(root, "symlink")
				err = os.Symlink(file, symlink)
				if err != nil {
					t.Fatalf("error creating a symlink in root dir: %v", err)
				}
				// should NOT change mode of symlink because it follows the link.

				if !(uid(root) == uid(dir) && uid(dir) == uid(file) && uid(file) == uid(symlink)) {
					panic("uid not the same")
				}
				if !(gid(root) == gid(dir) && gid(dir) == gid(file) && gid(file) == gid(symlink)) {
					panic("gid not the same")
				}

				// until now, a volume is created:
				// - root
				//   - dir
				//     - file <-
				//             |
				//   - symlink->

				var wantGID *int64
				if test.fsGroup {
					// ideally should set this to a different group.
					wantGID = new(int64)
					*wantGID = gid(root)
				}
				mounter := &localFakeMounter{path: root, attributes: Attributes{ReadOnly: test.readOnly}}

				err = SetVolumeOwnership(mounter, wantGID, &always)
				if err != nil {
					t.Errorf("error setting ownership: %v", err)
				}

				if test.fsGroup && (gid(root) != *wantGID || gid(dir) != *wantGID ||
					gid(file) != *wantGID || gid(symlink) != *wantGID) {
					t.Errorf(`
gid of root want %v got %v
gid of dir want %v got %v
gid of file want %v got %v
gid of symlink want %v got %v
                    `,
						*wantGID, gid(root),
						*wantGID, gid(dir),
						*wantGID, gid(file),
						*wantGID, gid(symlink))
				}

				if mode(root) != test.wantRootMode || mode(dir) != test.wantDirMode ||
					mode(file) != test.wantFileMode || mode(symlink) != test.wantSymlinkMode {
					t.Errorf(`
mode of root want %v got %v
mode of dir want %v got %v
mode of file want %v got %v
mode of symlink want %v got %v
                    `,
						test.wantRootMode, mode(root),
						test.wantDirMode, mode(dir),
						test.wantFileMode, mode(file),
						test.wantSymlinkMode, mode(symlink))
				}
			})
		}

	}
}

func uid(path string) int64 {
	info, err := os.Lstat(path)
	if err != nil {
		panic(err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat == nil {
		panic(err)
	}
	return int64(stat.Uid)
}

func gid(path string) int64 {
	info, err := os.Lstat(path)
	if err != nil {
		panic(err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat == nil {
		panic(err)
	}
	return int64(stat.Gid)
}

func mode(path string) os.FileMode {
	info, err := os.Lstat(path)
	if err != nil {
		panic(err)
	}
	return info.Mode()
}

func chmod(path string, mode os.FileMode) {
	err := os.Chmod(path, mode)
	if err != nil {
		panic(err)
	}
}
