// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build linux freebsd netbsd openbsd

package tarheader

import (
	"archive/tar"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// mknod requires privilege ...
func TestHeaderUnixDev(t *testing.T) {
	hExpect := tar.Header{
		Name:     "./dev/test0",
		Size:     0,
		Typeflag: tar.TypeBlock,
		Devminor: 5,
		Devmajor: 233,
	}
	// make our test block device
	var path string
	{
		var err error
		path, err = ioutil.TempDir("", "tarheader-test-")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(path)
		if err := os.Mkdir(filepath.Join(path, "dev"), os.FileMode(0755)); err != nil {
			t.Fatal(err)
		}
		mode := uint32(hExpect.Mode&07777) | syscall.S_IFBLK
		dev := uint32(((hExpect.Devminor & 0xfff00) << 12) | ((hExpect.Devmajor & 0xfff) << 8) | (hExpect.Devminor & 0xff))
		if err := syscall.Mknod(filepath.Join(path, hExpect.Name), mode, int(dev)); err != nil {
			if err == syscall.EPERM {
				t.Skip("no permission to CAP_MKNOD")
			}
			t.Fatal(err)
		}
	}
	fi, err := os.Stat(filepath.Join(path, hExpect.Name))
	if err != nil {
		t.Fatal(err)
	}

	hGot := tar.Header{
		Name:     "./dev/test0",
		Size:     0,
		Typeflag: tar.TypeBlock,
	}

	seen := map[uint64]string{}
	populateHeaderUnix(&hGot, fi, seen)
	if hGot.Devminor != hExpect.Devminor {
		t.Errorf("dev minor: got %d, expected %d", hGot.Devminor, hExpect.Devminor)
	}
	if hGot.Devmajor != hExpect.Devmajor {
		t.Errorf("dev major: got %d, expected %d", hGot.Devmajor, hExpect.Devmajor)
	}
}
