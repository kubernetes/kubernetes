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
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/networking"
	rktlog "github.com/coreos/rkt/pkg/log"
)

var (
	debug bool
)

func init() {
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")

	// this ensures that main runs only on main thread (thread group leader).
	// since namespace ops (unshare, setns) are done for a single thread, we
	// must ensure that the goroutine does not jump from OS thread to thread
	runtime.LockOSThread()
}

func main() {
	flag.Parse()

	log := rktlog.New(os.Stderr, "stage1 gc", debug)

	podID, err := types.NewUUID(flag.Arg(0))
	if err != nil {
		log.Fatal("UUID is missing or malformed")
	}

	if err := removeJournalLink(podID); err != nil {
		log.PrintE("error removing journal link", err)
	}

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

	n, err := networking.Load(".", podID)
	switch {
	case err == nil:
		n.Teardown(flavor, debug)
	case os.IsNotExist(err):
		// probably ran with --net=host
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
