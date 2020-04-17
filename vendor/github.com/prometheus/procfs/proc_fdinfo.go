// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strings"
)

// Regexp variables
var (
	rPos     = regexp.MustCompile(`^pos:\s+(\d+)$`)
	rFlags   = regexp.MustCompile(`^flags:\s+(\d+)$`)
	rMntID   = regexp.MustCompile(`^mnt_id:\s+(\d+)$`)
	rInotify = regexp.MustCompile(`^inotify`)
)

// ProcFDInfo contains represents file descriptor information.
type ProcFDInfo struct {
	// File descriptor
	FD string
	// File offset
	Pos string
	// File access mode and status flags
	Flags string
	// Mount point ID
	MntID string
	// List of inotify lines (structed) in the fdinfo file (kernel 3.8+ only)
	InotifyInfos []InotifyInfo
}

// FDInfo constructor. On kernels older than 3.8, InotifyInfos will always be empty.
func (p Proc) FDInfo(fd string) (*ProcFDInfo, error) {
	f, err := os.Open(p.path("fdinfo", fd))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fdinfo, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("could not read %s: %s", f.Name(), err)
	}

	var text, pos, flags, mntid string
	var inotify []InotifyInfo

	scanner := bufio.NewScanner(strings.NewReader(string(fdinfo)))
	for scanner.Scan() {
		text = scanner.Text()
		if rPos.MatchString(text) {
			pos = rPos.FindStringSubmatch(text)[1]
		} else if rFlags.MatchString(text) {
			flags = rFlags.FindStringSubmatch(text)[1]
		} else if rMntID.MatchString(text) {
			mntid = rMntID.FindStringSubmatch(text)[1]
		} else if rInotify.MatchString(text) {
			newInotify, err := parseInotifyInfo(text)
			if err != nil {
				return nil, err
			}
			inotify = append(inotify, *newInotify)
		}
	}

	i := &ProcFDInfo{
		FD:           fd,
		Pos:          pos,
		Flags:        flags,
		MntID:        mntid,
		InotifyInfos: inotify,
	}

	return i, nil
}

// InotifyInfo represents a single inotify line in the fdinfo file.
type InotifyInfo struct {
	// Watch descriptor number
	WD string
	// Inode number
	Ino string
	// Device ID
	Sdev string
	// Mask of events being monitored
	Mask string
}

// InotifyInfo constructor. Only available on kernel 3.8+.
func parseInotifyInfo(line string) (*InotifyInfo, error) {
	r := regexp.MustCompile(`^inotify\s+wd:([0-9a-f]+)\s+ino:([0-9a-f]+)\s+sdev:([0-9a-f]+)\s+mask:([0-9a-f]+)`)
	m := r.FindStringSubmatch(line)
	i := &InotifyInfo{
		WD:   m[1],
		Ino:  m[2],
		Sdev: m[3],
		Mask: m[4],
	}
	return i, nil
}

// ProcFDInfos represents a list of ProcFDInfo structs.
type ProcFDInfos []ProcFDInfo

func (p ProcFDInfos) Len() int           { return len(p) }
func (p ProcFDInfos) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p ProcFDInfos) Less(i, j int) bool { return p[i].FD < p[j].FD }

// InotifyWatchLen returns the total number of inotify watches
func (p ProcFDInfos) InotifyWatchLen() (int, error) {
	length := 0
	for _, f := range p {
		length += len(f.InotifyInfos)
	}

	return length, nil
}
