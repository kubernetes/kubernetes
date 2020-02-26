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
	"syscall"
	"testing"

	v1 "k8s.io/api/core/v1"
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
	always := v1.AlwaysChangeVolumePermission
	onrootMismatch := v1.OnRootMismatch
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

			if test.sgidMatch {
				mask |= os.ModeSetgid
			}

			if test.permissionMatch {
				mask |= execMask

			}
			err = os.Chmod(tmpDir, info.Mode()|mask)
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
