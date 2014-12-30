// +build linux

/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strings"
	"testing"
)

func TestReadProcMountsFrom(t *testing.T) {
	successCase :=
		`/dev/0 /path/to/0 type0 flags 0 0
		/dev/1    /path/to/1   type1	flags 1 1
		/dev/2 /path/to/2 type2 flags,1,2=3 2 2
		`
	hash, err := readProcMountsFrom(strings.NewReader(successCase), nil)
	if err != nil {
		t.Errorf("expected success")
	}
	if hash != 0xa3522051 {
		t.Errorf("expected 0xa3522051, got %#x", hash)
	}
	mounts := []MountPoint{}
	hash, err = readProcMountsFrom(strings.NewReader(successCase), &mounts)
	if err != nil {
		t.Errorf("expected success")
	}
	if hash != 0xa3522051 {
		t.Errorf("expected 0xa3522051, got %#x", hash)
	}
	if len(mounts) != 3 {
		t.Fatalf("expected 3 mounts, got %d", len(mounts))
	}
	mp := MountPoint{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0}
	if !mountPointsEqual(&mounts[0], &mp) {
		t.Errorf("got unexpected MountPoint[0]: %#v", mounts[0])
	}
	mp = MountPoint{"/dev/1", "/path/to/1", "type1", []string{"flags"}, 1, 1}
	if !mountPointsEqual(&mounts[1], &mp) {
		t.Errorf("got unexpected MountPoint[1]: %#v", mounts[1])
	}
	mp = MountPoint{"/dev/2", "/path/to/2", "type2", []string{"flags", "1", "2=3"}, 2, 2}
	if !mountPointsEqual(&mounts[2], &mp) {
		t.Errorf("got unexpected MountPoint[2]: %#v", mounts[2])
	}

	errorCases := []string{
		"/dev/0 /path/to/mount\n",
		"/dev/1 /path/to/mount type flags a 0\n",
		"/dev/2 /path/to/mount type flags 0 b\n",
	}
	for _, ec := range errorCases {
		_, err := readProcMountsFrom(strings.NewReader(ec), &mounts)
		if err == nil {
			t.Errorf("expected error")
		}
	}
}

func mountPointsEqual(a, b *MountPoint) bool {
	if a.Device != b.Device || a.Path != b.Path || a.Type != b.Type || !slicesEqual(a.Opts, b.Opts) || a.Pass != b.Pass || a.Freq != b.Freq {
		return false
	}
	return true
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
