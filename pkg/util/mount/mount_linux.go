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
	"os/exec"
	"strconv"
	"strings"
	"syscall"

	"github.com/golang/glog"
)

const (
	// How many times to retry for a consistent read of /proc/mounts.
	maxListTries = 3
	// Number of fields per line in "/proc/mounts", as per the fstab man page.
	expectedNumFieldsPerLine = 6
)

// Mounter implements mount.Interface for linux platform.
type Mounter struct{}

// Mount mounts source to target as fstype with given options. 'source' and 'fstype' must
// be an emtpy string in case it's not required, e.g. for remount, or for auto filesystem
// type, where kernel handles fs type for you. The mount 'options' is a list of options,
// currently come from mount(8), e.g. "ro", "remount", "bind", etc. If no more option is
// required, call Mount with an empty string list or nil.
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	// The remount options to use in case of bind mount, due to the fact that bind mount doesn't
	// respect mount options.  The list equals:
	//   options - 'bind' + 'remount' (no duplicate)
	bindRemountOpts := []string{"remount"}
	bind := false

	if len(options) != 0 {
		for _, option := range options {
			switch option {
			case "bind":
				bind = true
				break
			case "remount":
				break
			default:
				bindRemountOpts = append(bindRemountOpts, option)
			}
		}
	}

	if bind {
		err := doMount(source, target, fstype, []string{"bind"})
		if err != nil {
			return err
		}
		return doMount(source, target, fstype, bindRemountOpts)
	} else {
		return doMount(source, target, fstype, options)
	}
}

func doMount(source string, target string, fstype string, options []string) error {
	glog.V(5).Infof("Mounting %s %s %s %v", source, target, fstype, options)
	// Build mount command as follows:
	//   mount [-t $fstype] [-o $options] [$source] $target
	mountArgs := []string{}
	if len(fstype) > 0 {
		mountArgs = append(mountArgs, "-t", fstype)
	}
	if len(options) > 0 {
		mountArgs = append(mountArgs, "-o", strings.Join(options, ","))
	}
	if len(source) > 0 {
		mountArgs = append(mountArgs, source)
	}
	mountArgs = append(mountArgs, target)
	command := exec.Command("mount", mountArgs...)
	output, err := command.CombinedOutput()
	if err != nil {
		glog.Errorf("Mount failed: %v\nMounting arguments: %s %s %s %v\nOutput: %s\n",
			err, source, target, fstype, options, string(output))
	}
	return err
}

// Unmount unmounts target with given options.
func (mounter *Mounter) Unmount(target string) error {
	glog.V(5).Infof("Unmounting %s %v")
	command := exec.Command("umount", target)
	output, err := command.CombinedOutput()
	if err != nil {
		glog.Errorf("Unmount failed: %v\nUnmounting arguments: %s\nOutput: %s\n", err, target, string(output))
		return err
	}
	return nil
}

// List returns a list of all mounted filesystems.
func (mounter *Mounter) List() ([]MountPoint, error) {
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

// IsMountPoint determines if a directory is a mountpoint, by comparing the device for the
// directory with the device for it's parent.  If they are the same, it's not a mountpoint,
// if they're different, it is.
func (mounter *Mounter) IsMountPoint(file string) (bool, error) {
	stat, err := os.Stat(file)
	if err != nil {
		return false, err
	}
	rootStat, err := os.Lstat(file + "/..")
	if err != nil {
		return false, err
	}
	// If the directory has the same device as parent, then it's not a mountpoint.
	return stat.Sys().(*syscall.Stat_t).Dev != rootStat.Sys().(*syscall.Stat_t).Dev, nil
}

// readProcMounts reads /proc/mounts and produces a hash of the contents.  If the out
// argument is not nil, this fills it with MountPoint structs.
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
