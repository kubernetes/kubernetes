// Copyright 2016 The rkt Authors
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
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"

	"github.com/coreos/rkt/common"
	rktlog "github.com/coreos/rkt/pkg/log"
	stage1common "github.com/coreos/rkt/stage1/common"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/pkg/mountinfo"
)

var (
	debug     bool
	flagApp   string
	flagStage int

	log  *rktlog.Logger
	diag *rktlog.Logger
)

func init() {
	flag.StringVar(&flagApp, "app", "", "Application name")
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")

	// `--stage` is an internal implementation detail, not part of stage1 contract
	flag.IntVar(&flagStage, "stage", 0, "Removal step, defaults to 0 when called from the outside")
}

// This is a multi-step entrypoint. It starts in stage0 context then invokes
// itself again in stage1 context to perform further cleanup at pod level.
func main() {
	flag.Parse()

	stage1initcommon.InitDebug(debug)

	log, diag, _ = rktlog.NewLogSet("app-rm", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	appName, err := types.NewACName(flagApp)
	if err != nil {
		log.FatalE("invalid app name", err)
	}

	enterCmd := stage1common.PrepareEnterCmd(false)
	switch flagStage {
	case 0:
		// clean resources in stage0
		err = cleanupStage0(appName, enterCmd)
	case 1:
		// clean resources in stage1
		err = cleanupStage1(appName, enterCmd)
	default:
		// unknown step
		err = fmt.Errorf("unsupported cleaning step %d", flagStage)
	}
	if err != nil {
		log.FatalE("cleanup error", err)
	}

	os.Exit(0)
}

// cleanupStage0 is the default initial step for rm entrypoint, which takes
// care of cleaning up resources in stage0 and calling into stage1 by:
//  1. ensuring that the service has been stopped
//  2. removing unit files
//  3. calling itself in stage1 for further cleanups
//  4. calling `systemctl daemon-reload` in stage1
func cleanupStage0(appName *types.ACName, enterCmd []string) error {
	args := enterCmd
	args = append(args, "/usr/bin/systemctl")
	args = append(args, "is-active")
	args = append(args, appName.String())

	cmd := exec.Cmd{
		Path: args[0],
		Args: args,
	}

	// rely only on the output, since is-active returns non-zero for inactive units
	out, _ := cmd.Output()

	switch string(out) {
	case "failed\n":
	case "inactive\n":
	default:
		return fmt.Errorf("app %q is still running", appName.String())
	}

	s1rootfs := common.Stage1RootfsPath(".")
	serviceDir := filepath.Join(s1rootfs, "usr", "lib", "systemd", "system")
	appServicePaths := []string{
		filepath.Join(serviceDir, appName.String()+".service"),
		filepath.Join(serviceDir, "reaper-"+appName.String()+".service"),
	}

	for _, p := range appServicePaths {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("error removing app service file: %s", err)
		}
	}

	// TODO(sur): find all RW cgroups exposed for this app and clean them up in stage0 context

	// last cleaning steps are performed after entering pod context
	tasks := [][]string{
		// inception: call itself to clean stage1 before proceeding
		{"/app_rm", "--stage=1", fmt.Sprintf("--app=%s", appName), fmt.Sprintf("--debug=%t", debug)},
		// all cleaned-up, let systemd reload and forget about this app
		{"/usr/bin/systemctl", "daemon-reload"},
	}
	for _, cmdLine := range tasks {
		args := append(enterCmd, cmdLine...)
		cmd = exec.Cmd{
			Path: args[0],
			Args: args,
		}

		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("%q removal failed:\n%s", appName, out)
		}
	}
	return nil
}

// cleanupStage1 is meant to be executed in stage1 context. It inspects pod systemd-pid1 mountinfo to
// find all remaining mountpoints for appName and proceed to clean them up.
func cleanupStage1(appName *types.ACName, enterCmd []string) error {
	// TODO(lucab): re-evaluate once/if we support systemd as non-pid1 (eg. host pid-ns inheriting)
	mnts, err := mountinfo.ParseMounts(1)
	if err != nil {
		return err
	}
	appRootFs := filepath.Join("/opt/stage2", appName.String(), "rootfs")
	mnts = mnts.Filter(mountinfo.HasPrefix(appRootFs))

	// soft-errors here, stage0 may still be able to continue with the removal anyway
	for _, m := range mnts {
		// unlink first to avoid back-propagation
		_ = syscall.Mount("", m.MountPoint, "", syscall.MS_PRIVATE|syscall.MS_REC, "")
		// simple unmount, it may fail if the target is busy (eg. overlapping children)
		if e := syscall.Unmount(m.MountPoint, 0); e != nil {
			// if busy, just detach here and let the kernel clean it once free
			_ = syscall.Unmount(m.MountPoint, syscall.MNT_DETACH)
		}
	}

	return nil
}
