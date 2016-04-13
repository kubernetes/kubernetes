// Copyright 2015 The rkt Authors
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

//+build linux

package stage0

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"

	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
)

// GC enters the pod by fork/exec()ing the stage1's /gc similar to /init.
// /gc can expect to have its CWD set to the pod root.
// stage1Path is the path of the stage1 rootfs
func GC(pdir string, uuid *types.UUID, stage1Path string) error {
	err := unregisterPod(pdir, uuid)
	if err != nil {
		// Probably not worth abandoning the rest
		log.PrintE("warning: could not unregister pod with metadata service", err)
	}

	ep, err := getStage1Entrypoint(pdir, gcEntrypoint)
	if err != nil {
		return errwrap.Wrap(errors.New("error determining 'gc' entrypoint"), err)
	}

	args := []string{filepath.Join(stage1Path, ep)}
	if debugEnabled {
		args = append(args, "--debug")
	}
	args = append(args, uuid.String())

	c := exec.Cmd{
		Path:   args[0],
		Args:   args,
		Stderr: os.Stderr,
		Dir:    pdir,
	}
	return c.Run()
}

type mount struct {
	id         int
	parentID   int
	mountPoint string
	opt        map[string]struct{}
}

type mounts []*mount

// getMountDepth determines and returns the number of ancestors of the mount at index i
func (m mounts) getMountDepth(i int) int {
	ancestorCount := 0
	current := m[i]
	for found := true; found; {
		found = false
		for _, mnt := range m {
			if mnt.id == current.parentID {
				ancestorCount += 1
				current = mnt
				found = true
				break
			}
		}
	}
	return ancestorCount
}

// Less ensures that mounts are sorted in an order we can unmount; descendant before ancestor.
// The requirement of transitivity for Less has to be fulfilled otherwise the sort algorithm will fail.
func (m mounts) Less(i, j int) (result bool) { return m.getMountDepth(i) >= m.getMountDepth(j) }
func (m mounts) Len() int                    { return len(m) }
func (m mounts) Swap(i, j int)               { m[i], m[j] = m[j], m[i] }

// getMountsForPrefix parses mi (/proc/PID/mountinfo) and returns mounts for path prefix
func getMountsForPrefix(path string, mi io.Reader) (mounts, error) {
	var podMounts mounts
	sc := bufio.NewScanner(mi)
	var (
		mountID    int
		parentID   int
		mountPoint string
		opt        map[string]struct{}
	)

	for sc.Scan() {
		line := sc.Text()
		lineResult := strings.Split(line, " ")
		if len(lineResult) < 7 {
			return nil, fmt.Errorf("Not enough fields from line %q: %+v", line, lineResult)
		}

		opt = map[string]struct{}{}
		for i, s := range lineResult {
			if s == "-" {
				break
			}
			var err error
			switch i {
			case 0:
				mountID, err = strconv.Atoi(s)
			case 1:
				parentID, err = strconv.Atoi(s)
			case 2, 3:
			case 4:
				mountPoint = s
			default:
				split := strings.Split(s, ":")
				switch len(split) {
				case 1:
					// we ignore modes like rw, relatime, etc.
				case 2:
					opt[split[0]] = struct{}{}
				default:
					err = fmt.Errorf("found unexpected key:value field with more than two colons: %s", s)
				}
			}
			if err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("could not parse mountinfo line %q", line), err)
			}
		}

		if strings.Contains(mountPoint, path) {
			mnt := &mount{
				id:         mountID,
				parentID:   parentID,
				mountPoint: mountPoint,
				opt:        opt,
			}
			podMounts = append(podMounts, mnt)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, errwrap.Wrap(errors.New("problem parsing mountinfo"), err)
	}
	sort.Sort(podMounts)
	return podMounts, nil
}

func needsRemountPrivate(mnt *mount) bool {
	for _, key := range []string{
		"shared",
		"master",
	} {
		if _, needsRemount := mnt.opt[key]; needsRemount {
			return true
		}
	}
	return false
}

// MountGC removes mounts from pods that couldn't be GCed cleanly.
func MountGC(path, uuid string) error {
	mi, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return err
	}
	defer mi.Close()

	mnts, err := getMountsForPrefix(path, mi)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("error getting mounts for pod %s from mountinfo", uuid), err)
	}

	for i := len(mnts) - 1; i >= 0; i -= 1 {
		mnt := mnts[i]
		if needsRemountPrivate(mnt) {
			if err := syscall.Mount("", mnt.mountPoint, "", syscall.MS_PRIVATE, ""); err != nil {
				return errwrap.Wrap(fmt.Errorf("could not remount at %v", mnt.mountPoint), err)
			}
		}
	}

	for _, mnt := range mnts {
		if err := syscall.Unmount(mnt.mountPoint, 0); err != nil {
			if err != syscall.ENOENT && err != syscall.EINVAL {
				return errwrap.Wrap(fmt.Errorf("could not unmount %v", mnt.mountPoint), err)
			}
		}
	}
	return nil
}
