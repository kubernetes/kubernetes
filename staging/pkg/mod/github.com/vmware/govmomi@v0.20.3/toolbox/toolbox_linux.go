/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package toolbox

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/vmware/govmomi/toolbox/vix"
)

func fileExtendedInfoFormat(dir string, info os.FileInfo) string {
	const format = "<fxi>" +
		"<Name>%s</Name>" +
		"<ft>%d</ft>" +
		"<fs>%d</fs>" +
		"<mt>%d</mt>" +
		"<at>%d</at>" +
		"<uid>%d</uid>" +
		"<gid>%d</gid>" +
		"<perm>%d</perm>" +
		"<slt>%s</slt>" +
		"</fxi>"

	props := 0
	targ := ""

	if info.IsDir() {
		props |= vix.FileAttributesDirectory
	}

	if info.Mode()&os.ModeSymlink == os.ModeSymlink {
		props |= vix.FileAttributesSymlink
		targ, _ = os.Readlink(filepath.Join(dir, info.Name()))
	}

	size := info.Size()
	mtime := info.ModTime().Unix()
	perm := info.Mode().Perm()

	atime := mtime
	uid := os.Getuid()
	gid := os.Getgid()

	if sys, ok := info.Sys().(*syscall.Stat_t); ok {
		atime = time.Unix(sys.Atim.Unix()).Unix()
		uid = int(sys.Uid)
		gid = int(sys.Gid)
	}

	return fmt.Sprintf(format, info.Name(), props, size, mtime, atime, uid, gid, perm, targ)
}
