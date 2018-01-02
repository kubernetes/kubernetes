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

package main

import (
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"syscall"

	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/cgroup"
	"github.com/coreos/rkt/common/cgroup/v1"
	"github.com/coreos/rkt/networking"
	rktlog "github.com/coreos/rkt/pkg/log"
)

const (
	cgroupFsPath = "/sys/fs/cgroup"
)

var (
	debug       bool
	log         *rktlog.Logger
	diag        *rktlog.Logger
	localConfig string
)

func init() {
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")
	flag.StringVar(&localConfig, "local-config", common.DefaultLocalConfigDir, "Local config path")

	// this ensures that main runs only on main thread (thread group leader).
	// since namespace ops (unshare, setns) are done for a single thread, we
	// must ensure that the goroutine does not jump from OS thread to thread
	runtime.LockOSThread()
}

func main() {
	flag.Parse()

	log, diag, _ = rktlog.NewLogSet("stage1 gc", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	podID, err := types.NewUUID(flag.Arg(0))
	if err != nil {
		log.Fatal("UUID is missing or malformed")
	}

	diag.Printf("Removing journal link.")
	if err := removeJournalLink(podID); err != nil {
		log.PrintE("error removing journal link", err)
	}

	diag.Printf("Cleaning up cgroups.")
	if err := cleanupV1Cgroups(); err != nil {
		log.PrintE("error cleaning up cgroups", err)
	}

	diag.Printf("Tearing down networks.")
	if err := gcNetworking(podID); err != nil {
		log.FatalE("", err)
	}
}

func gcNetworking(podID *types.UUID) error {
	var flavor string
	// we first try to read the flavor from stage1 for backwards compatibility
	flavor, err := os.Readlink(filepath.Join(common.Stage1RootfsPath("."), "flavor"))
	if err != nil {
		// if we couldn't read the flavor from stage1 it could mean the overlay
		// filesystem is already unmounted (e.g. the system has been rebooted).
		// In that case we try to read it from the pod's root directory
		flavor, err = os.Readlink("flavor")
		if err != nil {
			return errwrap.Wrap(errors.New("failed to get stage1 flavor"), err)
		}
	}

	n, err := networking.Load(".", podID, localConfig)
	switch {
	case err == nil:
		n.Teardown(flavor, debug)
	case os.IsNotExist(err):
		// either ran with --net=host, or failed during setup
		if err := networking.CleanUpGarbage(".", podID); err != nil {
			diag.PrintE("failed cleaning up nework NS", err)
		}
	default:
		return errwrap.Wrap(errors.New("failed loading networking state"), err)
	}

	return nil
}

func removeJournalLink(uuid *types.UUID) error {
	// if the host runs systemd, we link the journal and set pod's machine-id
	// as pod's UUID without the dashes in init.go:
	// https://github.com/coreos/rkt/blob/95e6bc/stage1/init/init.go#L382
	machineID := strings.Replace(uuid.String(), "-", "", -1)
	journalLink := filepath.Join("/var/log/journal", machineID)
	err := os.Remove(journalLink)
	if err != nil && err.(*os.PathError).Err == syscall.ENOENT {
		err = nil
	}
	return err
}

type dirsByLength []string

func (c dirsByLength) Len() int           { return len(c) }
func (c dirsByLength) Less(i, j int) bool { return len(c[i]) < len(c[j]) }
func (c dirsByLength) Swap(i, j int)      { c[i], c[j] = c[j], c[i] }

func cleanupV1Cgroups() error {
	isUnified, err := cgroup.IsCgroupUnified("/")
	if err != nil {
		return errwrap.Wrap(errors.New("failed to determine the cgroup version"), err)
	}
	if isUnified {
		return nil
	}

	b, err := ioutil.ReadFile("subcgroup")
	if err != nil {
		if os.IsNotExist(err) {
			diag.Printf("subcgroup file missing, probably a failed pod. Skipping cgroup cleanup.")
			return nil
		}
		return errwrap.Wrap(errors.New("error reading subcgroup file"), err)
	}
	subcgroup := string(b)

	// if we're trying to clean up our own cgroup it means we're running in the
	// same unit file as the rkt pod. We don't have to do anything, systemd
	// will do the cleanup for us
	ourCgroupPath, err := v1.GetOwnCgroupPath("name=systemd")
	if err == nil {
		if strings.HasPrefix(ourCgroupPath, "/"+subcgroup) {
			return nil
		}
	}

	f, err := os.Open(cgroupFsPath)
	if err != nil {
		return err
	}
	defer f.Close()
	ns, err := f.Readdirnames(0)
	if err != nil {
		return err
	}
	var cgroupDirs []string
	for _, c := range ns {
		scPath := filepath.Join(cgroupFsPath, c, subcgroup)
		walkCgroupDirs := func(path string, info os.FileInfo, err error) error {
			// if the subcgroup is already removed, we're fine
			if os.IsNotExist(err) {
				return nil
			}
			if err != nil {
				return err
			}
			mode := info.Mode()
			if mode.IsDir() {
				cgroupDirs = append(cgroupDirs, path)
			}
			return nil
		}
		if err := filepath.Walk(scPath, walkCgroupDirs); err != nil {
			log.PrintE(fmt.Sprintf("error walking subcgroup %q", scPath), err)
			continue
		}
	}

	// remove descendants before ancestors
	sort.Sort(sort.Reverse(dirsByLength(cgroupDirs)))
	for _, d := range cgroupDirs {
		if err := os.RemoveAll(d); err != nil {
			log.PrintE(fmt.Sprintf("error removing subcgroup %q", d), err)
		}
	}

	return nil
}
