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

// +build linux

package tarheader

import (
	"archive/tar"
	"os"
	"syscall"
	"time"
)

func init() {
	populateHeaderStat = append(populateHeaderStat, populateHeaderCtime)
}

func populateHeaderCtime(h *tar.Header, fi os.FileInfo, _ map[uint64]string) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return
	}

	sec, nsec := st.Ctim.Unix()
	ctime := time.Unix(sec, nsec)
	h.ChangeTime = ctime
}
