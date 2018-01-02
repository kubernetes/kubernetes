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
	"reflect"
	"runtime"
	"strconv"
	"syscall"

	"github.com/coreos/go-systemd/unit"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/fs"
	rktlog "github.com/coreos/rkt/pkg/log"
	stage1common "github.com/coreos/rkt/stage1/common"
	stage1types "github.com/coreos/rkt/stage1/common/types"
	stage1init "github.com/coreos/rkt/stage1/init/common"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

var (
	flagApp    string
	flagUUID   string
	flagStage  int
	flagTarget string

	debug bool
	log   *rktlog.Logger
	diag  *rktlog.Logger
)

func init() {
	flag.StringVar(&flagApp, "app", "", "Application name")
	flag.StringVar(&flagUUID, "uuid", "", "Pod UUID")
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")

	// `--stage` is an internal implementation detail, not part of stage1 contract
	flag.IntVar(&flagStage, "stage", 0, "app add stage, defaults to 0 when called from the outside")
	flag.StringVar(&flagTarget, "target", "", "the target mount point. this is only relevant for -stage=1")
}

func main() {
	flag.Parse()

	stage1init.InitDebug(debug)

	log, diag, _ = rktlog.NewLogSet("stage"+strconv.Itoa(flagStage)+": app-add", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	uuid, err := types.NewUUID(flagUUID)
	if err != nil {
		log.FatalE("UUID is missing or malformed", err)
	}

	appName, err := types.NewACName(flagApp)
	if err != nil {
		log.FatalE("invalid app name", err)
	}

	switch flagStage {
	case 0:
		err = appAddStage0(appName, uuid)
	case 1:
		err = appAddStage1(appName, uuid, flagTarget)
	default:
		log.Fatalf("unknown stage %d", flagStage)
	}

	if err != nil {
		log.FatalE("app add failed", err)
	}

	os.Exit(0)
}

func appAddStage0(appName *types.ACName, uuid *types.UUID) error {
	root, err := os.Getwd()
	if err != nil {
		return errwrap.Wrapf("failed to determine current directory", err)
	}
	p, err := stage1types.LoadPod(root, uuid, nil)
	if err != nil {
		return errwrap.Wrapf("failed to load pod", err)
	}

	ra := p.Manifest.Apps.Get(*appName)
	if ra == nil {
		return fmt.Errorf("failed to find app %q", *appName)
	}

	binPath, err := stage1init.FindBinPath(p, ra)
	if err != nil {
		return errwrap.Wrapf("failed to find bin path", err)
	}

	if ra.App.WorkingDirectory == "" {
		ra.App.WorkingDirectory = "/"
	}

	enterCmd := stage1common.PrepareEnterCmd(false)
	err = addMountsStage0(p, ra, enterCmd)
	if err != nil {
		return errwrap.Wrapf("adding mounts failed", err)
	}

	// write service files
	w := stage1init.NewUnitWriter(p)
	w.AppUnit(ra, binPath,
		unit.NewUnitOption("Unit", "Before", "halt.target"),
		unit.NewUnitOption("Unit", "Conflicts", "halt.target"),
	)
	w.AppReaperUnit(ra.Name, binPath)
	if err := w.Error(); err != nil {
		return errwrap.Wrapf("error generating app units", err)
	}

	// stage2 environment is ready at this point, but systemd does not know
	// about the new application yet
	args := enterCmd
	args = append(args, "/usr/bin/systemctl")
	args = append(args, "daemon-reload")

	cmd := exec.Cmd{
		Path: args[0],
		Args: args,
	}

	out, err := cmd.CombinedOutput()
	diag.Printf("systemctl daemon-reload output:\n%s\n", out)
	if err != nil {
		return errwrap.Wrapf(fmt.Sprintf("daemon reload of %s failed", appName), err)
	}

	return nil
}

// addMountStage0 iterates over all requested mounts and bind mounts them to the stage2 rootfs.
func addMountsStage0(p *stage1types.Pod, ra *schema.RuntimeApp, enterCmd []string) error {
	sharedVolPath, err := common.CreateSharedVolumesPath(p.Root)
	if err != nil {
		return err
	}

	vols := make(map[types.ACName]types.Volume)
	for _, v := range p.Manifest.Volumes {
		vols[v.Name] = v
	}

	imageManifest := p.Images[ra.Name.String()]

	mounts, err := stage1init.GenerateMounts(ra, p.Manifest.Volumes, stage1init.ConvertedFromDocker(imageManifest))
	if err != nil {
		return errwrap.Wrapf("could not generate mounts", err)
	}

	absRoot, err := filepath.Abs(p.Root)
	if err != nil {
		return errwrap.Wrapf("could not determine pod's absolute path", err)
	}

	appRootfs := common.AppRootfsPath(absRoot, ra.Name)

	// This logic is mostly copied from appToNspawnArgs
	// TODO(cdc): deduplicate
	for _, m := range mounts {
		shPath := filepath.Join(sharedVolPath, m.Volume.Name.String())

		// Evaluate symlinks within the app's rootfs - otherwise absolute
		// symlinks will be wrong.
		mntPath, err := stage1init.EvaluateSymlinksInsideApp(appRootfs, m.Mount.Path)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("could not evaluate path %v", m.Mount.Path), err)
		}

		// Create the stage1 destination
		if err := stage1init.PrepareMountpoints(shPath, filepath.Join(appRootfs, mntPath), &m.Volume, m.DockerImplicit); err != nil {
			return errwrap.Wrapf("could not prepare mountpoint", err)
		}

		err = addMountStage0(p, ra, &m, mntPath, enterCmd)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("error adding mount volume %v path %v", m.Mount.Volume, m.Mount.Path), err)
		}
	}
	return nil
}

// addMountStage1 bind-mounts (moves) the given mount from the host in the container.
//
// We use the propagation mechanism of systemd-nspawn. In all systemd-nspawn
// containers, the directory "/run/systemd/nspawn/propagate/$MACHINE_ID" on
// the host is propagating mounts to the directory
// "/run/systemd/nspawn/incoming/" in the container mount namespace. Once a
// bind mount is propagated, we simply move to its correct location.
//
// The algorithm is the same as in "machinectl bind":
// https://github.com/systemd/systemd/blob/v231/src/machine/machine-dbus.c#L865
// except that we don't use setns() to enter the mount namespace of the pod
// because Linux does not allow multithreaded applications (such as Go
// programs) to change mount namespaces with setns. Instead, we fork another
// process written in C (single-threaded) to enter the mount namespace. The
// command used is specified by the "enterCmd" parameter.
//
// Users might request a bind mount to be set up read-only. This complicates
// things a bit because on Linux, setting up a read-only bind mount involves
// two mount() calls, so it is not atomic. We don't want the container to see
// the mount in read-write mode, even for a short time, so we don't create the
// bind mount directly in "/run/systemd/nspawn/propagate/$MACHINE_ID" to avoid
// an immediate propagation to the container. Instead, we create a temporary
// playground in "/tmp/rkt.propagate.XXXX" and create the bind mount in
// "/tmp/rkt.propagate.XXXX/mount" with the correct read-only attribute before
// moving it.
//
// Another complication is that the playground cannot be on a shared mount
// because Linux does not allow MS_MOVE to be applied to mounts with MS_SHARED
// parent mounts. But by default, systemd mounts everything as shared, see:
// https://github.com/systemd/systemd/blob/v231/src/core/mount-setup.c#L392
// We set up the temporary playground as a slave bind mount to avoid this
// limitation.
func addMountStage0(
	p *stage1types.Pod, ra *schema.RuntimeApp,
	m *stage1init.Mount, target string,
	enterCmd []string,
) error {
	mnt := fs.NewLoggingMounter(
		fs.MounterFunc(syscall.Mount),
		fs.UnmounterFunc(syscall.Unmount),
		log.Printf,
	)

	hostPodRoot, err := filepath.Abs(p.Root)
	if err != nil {
		return errwrap.Wrapf("error reading absolute pod root path", err)
	}

	src := m.Source(hostPodRoot)
	warn := warner(diag.Printf)

	pg, err := newPlayground(src, "", "rkt.propagate.stage1", mnt)
	if err != nil {
		return errwrap.Wrapf("error creating stage1 playground", err)
	}
	defer warn(pg.Cleanup)

	err = mnt.Mount(src, pg.Playground(), "bind", syscall.MS_BIND, "")
	if err != nil {
		return errwrap.Wrapf("bind mount src to rkt.propagate.stage1/mount failed", err)
	}

	if m.ReadOnly {
		err = mnt.Mount("", pg.Playground(), "bind", syscall.MS_REMOUNT|syscall.MS_RDONLY|syscall.MS_BIND, "")
		if err != nil {
			return errwrap.Wrapf("remount of rkt.propagate.stage1/mount failed", err)
		}
	}

	// move mount it to the propagation directory prepared by systemd-nspawn
	propagate := filepath.Join("/run/systemd/nspawn/propagate/", "rkt-"+p.UUID.String(), "rkt.mount")
	if err := stage1init.EnsureTargetExists(src, propagate); err != nil {
		return errwrap.Wrapf("error creating propagate mountpoint", err)
	}
	defer warn(func() error { return os.Remove(propagate) })

	err = mnt.Mount(pg.Playground(), propagate, "", syscall.MS_MOVE, "")
	if err != nil {
		return errwrap.Wrapf("error moving temporary mountpoint to propagate directory", err)
	}
	defer warn(func() error { return mnt.Unmount(propagate, 0) })

	// enter stage1
	args := append(enterCmd,
		"/app_add", "-stage=1",
		"-debug="+strconv.FormatBool(debug),
		"-uuid="+p.UUID.String(),
		"-app="+ra.Name.String(),
		"-target="+target,
	)

	diag.Printf("entering stage1 %q", args)

	cmd := exec.Cmd{
		Path: args[0],
		Args: args,
	}

	out, err := cmd.CombinedOutput()
	diag.Printf("stage1 output:\n%s\n", out)
	if err != nil {
		return errwrap.Wrapf("app-add in stage1 failed", err)
	}

	return nil
}

// appAddStage1 bind mounts the incoming mount /run/systemd/nspawn/incoming/rkt.mount
// to the final target inside the stage2 rootfs.
//
// It is assumed that it is executed in the mount namespace of stage1.
// We chroot into the stage2 rootfs and the incoming mount is bind mounted to the actual target.
func appAddStage1(appName *types.ACName, uuid *types.UUID, target string) error {
	mnt := fs.NewLoggingMounter(
		fs.MounterFunc(syscall.Mount),
		fs.UnmounterFunc(syscall.Unmount),
		log.Printf,
	)
	appRoot := filepath.Join("/opt/stage2", appName.String(), "rootfs")
	target = filepath.Join(appRoot, target)
	incoming := "/run/systemd/nspawn/incoming/rkt.mount"

	if err := stage1init.EnsureTargetExists(incoming, target); err != nil {
		return errwrap.Wrapf("ensuring "+target+" failed", err)
	}

	if err := mnt.Mount(incoming, target, "", syscall.MS_MOVE, ""); err != nil {
		return errwrap.Wrapf("mount move "+incoming+" to "+target+" failed", err)
	}

	return nil
}

// warner enables printing a warning message in case a function call fails.
// Given the warnf function (compatible with fmt.Printf) for printing warning messages it returns the actual warner.
// The returned warner takes a possibly failing function "func() error" and prints a warning in case it failed.
// The actual error is ignored.
//
// Example usage:
//
//  warn := warner(log.Printf)
//  file, err := os.Open("file")
//  if err != nil { return err }
//  // do stuff
//  defer warn(file.Close)
func warner(warnf func(string, ...interface{})) func(func() error) {
	return func(f func() error) {
		if err := f(); err != nil {
			caller := runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
			warnf("%s failed: %v\n", caller, err)
		}
	}
}

// playground is the struct that represents a directory (or file)
// which can be moved to new locations using syscall.MS_MOVE
// even in environments marked as MS_SHARED (i.e. systemd based systems).
//
// This is ensured by creating a directory which is bind-mounted over itself
// and marked as a slave mount.
//
// The caller is responsible for calling Cleanup() after move operations.
type playground struct {
	tmpDir     string // the directory the playground resides in
	playground string // the full path to the playground

	mnt fs.MountUnmounter
}

// newPlayground creates a playground directory (or file) in the given directory dir with a name beginning with prefix.
// If dir is empty string, newPlayground uses the default directory for temporary files (see os.TempDir).
// The playground type (file or directory) is equal to the type of the type of src.
// The playground can be moved to a new location using syscall.MS_MOVE.
// Use the Playground() method to get the full playground path.
func newPlayground(src, dir, prefix string, mnt fs.MountUnmounter) (*playground, error) {
	p := playground{
		mnt: mnt,
	}

	var err error
	p.tmpDir, err = ioutil.TempDir(dir, prefix+".")
	if err != nil {
		return nil, errwrap.Wrapf("creating temporary "+prefix+" failed", err)
	}
	// removed in Cleanup

	err = p.mnt.Mount(p.tmpDir, p.tmpDir, "bind", syscall.MS_BIND, "")
	if err != nil {
		return nil, errwrap.Wrapf("bind mount of "+prefix+" failed", err)
	}
	// unmounted in Cleanup

	err = mnt.Mount("", p.tmpDir, "none", syscall.MS_SLAVE, "")
	if err != nil {
		return nil, errwrap.Wrapf("slave mount of "+prefix+" failed", err)
	}

	p.playground = filepath.Join(p.tmpDir, "playground")
	if err := stage1init.EnsureTargetExists(src, p.playground); err != nil {
		return nil, errwrap.Wrapf("creating rkt.propagate.stage2/mount failed", err)
	}
	// removed in Cleanup

	return &p, nil
}

// Playground returns its full path.
func (p *playground) Playground() string {
	return p.playground
}

// Cleanup cleans up the playground.
// It removes it, unmounts the parent directory, and removes the unmounted directory.
func (p *playground) Cleanup() error {
	if err := os.Remove(p.playground); err != nil {
		return err
	}

	if err := p.mnt.Unmount(p.tmpDir, 0); err != nil {
		return err
	}

	return os.Remove(p.tmpDir)
}
