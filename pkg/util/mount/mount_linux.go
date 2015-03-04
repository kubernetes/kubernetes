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
	"bufio"
	"fmt"
	"hash/adler32"
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/golang/glog"
)

const FlagBind = syscall.MS_BIND
const FlagReadOnly = syscall.MS_RDONLY

type Mounter struct{}

// Wraps syscall.Mount()
func (mounter *Mounter) Mount(source string, target string, fstype string, flags uintptr, data string) error {
	glog.V(5).Infof("Mounting %s %s %s %d %s", source, target, fstype, flags, data)
	return syscall.Mount(source, target, fstype, flags, data)
}

// Wraps syscall.Unmount()
func (mounter *Mounter) Unmount(target string, flags int) error {
	return syscall.Unmount(target, flags)
}

// How many times to retry for a consistent read of /proc/mounts.
const maxListTries = 3

func (*Mounter) List() ([]MountPoint, error) {
	hash1, err := readProcMounts(nil)
	if err != nil {
		return nil, err
	}

	for i := 0; i < maxListTries; i++ {
		mps := []MountPoint{}
		hash2, err := readProcMounts(&mps)
		if err != nil {
			return nil, err
		}
		if hash1 == hash2 {
			// Success
			return mps, nil
		}
		hash1 = hash2
	}
	return nil, fmt.Errorf("failed to get a consistent snapshot of /proc/mounts after %d tries", maxListTries)
}

// As per the fstab man page.
const expectedNumFieldsPerLine = 6

// Read /proc/mounts and produces a hash of the contents.  If the out argument
// is not nil, this fills it with MountPoint structs.
func readProcMounts(out *[]MountPoint) (uint32, error) {
	file, err := os.Open("/proc/mounts")
	if err != nil {
		return 0, err
	}
	defer file.Close()
	return readProcMountsFrom(file, out)
}

func readProcMountsFrom(file io.Reader, out *[]MountPoint) (uint32, error) {
	hash := adler32.New()
	scanner := bufio.NewReader(file)
	for {
		line, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		}
		fields := strings.Fields(line)
		if len(fields) != expectedNumFieldsPerLine {
			return 0, fmt.Errorf("wrong number of fields (expected %d, got %d): %s", expectedNumFieldsPerLine, len(fields), line)
		}

		fmt.Fprintf(hash, "%s", line)

		if out != nil {
			mp := MountPoint{
				Device: fields[0],
				Path:   fields[1],
				Type:   fields[2],
				Opts:   strings.Split(fields[3], ","),
			}

			freq, err := strconv.Atoi(fields[4])
			if err != nil {
				return 0, err
			}
			mp.Freq = freq

			pass, err := strconv.Atoi(fields[5])
			if err != nil {
				return 0, err
			}
			mp.Pass = pass

			*out = append(*out, mp)
		}
	}
	return hash.Sum32(), nil
}
