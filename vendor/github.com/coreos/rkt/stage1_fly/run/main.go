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

package main

import (
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/fileutil"
	pkgflag "github.com/coreos/rkt/pkg/flag"
	"github.com/coreos/rkt/pkg/fs"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/pkg/mountinfo"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/pkg/user"
	stage1common "github.com/coreos/rkt/stage1/common"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"
)

const (
	flavor = "fly"
)

type flyMount struct {
	HostPath         string
	TargetPrefixPath string
	RelTargetPath    string
	Fs               string
	Flags            uintptr
}

type volumeMountTuple struct {
	V types.Volume
	M schema.Mount
}

var (
	debug bool

	discardNetlist common.NetList
	discardBool    bool
	discardString  string

	log  *rktlog.Logger
	diag *rktlog.Logger
)

func parseFlags() *stage1commontypes.RuntimePod {
	rp := stage1commontypes.RuntimePod{}

	flag.BoolVar(&debug, "debug", false, "Run in debug mode")

	// The following flags need to be supported by stage1 according to
	// https://github.com/coreos/rkt/blob/master/Documentation/devel/stage1-implementors-guide.md
	// Most of them are ignored
	// These are ignored, but stage0 always passes them
	flag.Var(&discardNetlist, "net", "Setup networking")
	flag.StringVar(&discardString, "local-config", common.DefaultLocalConfigDir, "Local config path")

	// These are discarded with a warning
	// TODO either implement these, or stop passing them
	flag.Bool("interactive", true, "The pod is interactive (ignored, always true)")
	flag.Var(pkgflag.NewDiscardFlag("mds-token"), "mds-token", "MDS auth token (not implemented)")

	flag.Var(pkgflag.NewDiscardFlag("hostname"), "hostname", "Set hostname (not implemented)")
	flag.Bool("disable-capabilities-restriction", true, "ignored")
	flag.Bool("disable-paths", true, "ignored")
	flag.Bool("disable-seccomp", true, "ignored")

	dnsConfMode := pkgflag.MustNewPairList(map[string][]string{
		"resolv": {"host", "stage0", "none", "default"},
		"hosts":  {"host", "stage0", "default"},
	}, map[string]string{
		"resolv": "default",
		"hosts":  "default",
	})
	flag.Var(dnsConfMode, "dns-conf-mode", "DNS config file modes")

	flag.Parse()

	rp.Debug = debug
	rp.ResolvConfMode = dnsConfMode.Pairs["resolv"]
	rp.EtcHostsMode = dnsConfMode.Pairs["hosts"]

	return &rp
}

func addMountPoints(namedVolumeMounts map[types.ACName]volumeMountTuple, mountpoints []types.MountPoint) error {
	for _, mp := range mountpoints {
		tuple, exists := namedVolumeMounts[mp.Name]
		switch {
		case exists && tuple.M.Path != mp.Path:
			return fmt.Errorf("conflicting path information from mount and mountpoint %q", mp.Name)
		case !exists:
			namedVolumeMounts[mp.Name] = volumeMountTuple{M: schema.Mount{Volume: mp.Name, Path: mp.Path}}
			diag.Printf("adding %+v", namedVolumeMounts[mp.Name])
		}
	}
	return nil
}

func evaluateMounts(rfs string, app string, p *stage1commontypes.Pod) ([]flyMount, error) {
	namedVolumeMounts := map[types.ACName]volumeMountTuple{}

	// Insert the PodManifest's first RuntimeApp's Mounts
	for _, m := range p.Manifest.Apps[0].Mounts {
		_, exists := namedVolumeMounts[m.Volume]
		if exists {
			return nil, fmt.Errorf("duplicate mount given: %q", m.Volume)
		}
		namedVolumeMounts[m.Volume] = volumeMountTuple{M: m}
		diag.Printf("adding %+v", namedVolumeMounts[m.Volume])
	}

	// Merge command-line Mounts with ImageManifest's MountPoints
	var imAppManifestMPs []types.MountPoint
	if imApp := p.Images[app].App; imApp != nil {
		imAppManifestMPs = imApp.MountPoints
		if err := addMountPoints(namedVolumeMounts, imAppManifestMPs); err != nil {
			return nil, err
		}
	}

	// Merge command-line Mounts with PodManifest's RuntimeApp's App's MountPoints
	raApp := p.Manifest.Apps[0]
	if err := addMountPoints(namedVolumeMounts, raApp.App.MountPoints); err != nil {
		return nil, err
	}

	// Insert the command-line Volumes
	for _, v := range p.Manifest.Volumes {
		// Check if we have a mount for this volume
		tuple, exists := namedVolumeMounts[v.Name]
		if !exists {
			return nil, fmt.Errorf("missing mount for volume %q", v.Name)
		} else if tuple.M.Volume != v.Name {
			// assertion regarding the implementation, should never happen
			return nil, fmt.Errorf("mismatched volume:mount pair: %q != %q", v.Name, tuple.M.Volume)
		}
		namedVolumeMounts[v.Name] = volumeMountTuple{V: v, M: tuple.M}
		diag.Printf("adding %+v", namedVolumeMounts[v.Name])
	}

	// Merge command-line Volumes with ImageManifest's MountPoints
	for _, mp := range imAppManifestMPs {
		// Check if we have a volume for this mountpoint
		tuple, exists := namedVolumeMounts[mp.Name]
		if !exists || tuple.V.Name == "" {
			return nil, fmt.Errorf("missing volume for mountpoint %q", mp.Name)
		}

		// If empty, fill in ReadOnly bit
		if tuple.V.ReadOnly == nil {
			v := tuple.V
			v.ReadOnly = &mp.ReadOnly
			namedVolumeMounts[mp.Name] = volumeMountTuple{M: tuple.M, V: v}
			diag.Printf("adding %+v", namedVolumeMounts[mp.Name])
		}
	}

	// Gather host mounts which we make MS_SHARED if passed as a volume source
	hostMounts := map[string]struct{}{}
	mnts, err := mountinfo.ParseMounts(0)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("can't gather host mounts"), err)
	}
	for _, m := range mnts {
		hostMounts[m.MountPoint] = struct{}{}
	}

	argFlyMounts := []flyMount{}
	for _, tuple := range namedVolumeMounts {
		if _, isHostMount := hostMounts[tuple.V.Source]; isHostMount {
			// Mark the host mount as SHARED so the container's changes to the mount are propagated to the host
			argFlyMounts = append(argFlyMounts,
				flyMount{"", "", tuple.V.Source, "none", syscall.MS_REC | syscall.MS_SHARED},
			)
		}

		var (
			flags     uintptr = syscall.MS_BIND
			recursive         = tuple.V.Recursive != nil && *tuple.V.Recursive
			ro                = tuple.V.ReadOnly != nil && *tuple.V.ReadOnly
		)

		// If Recursive is not set, default to non-recursive.
		if recursive {
			flags |= syscall.MS_REC
		}

		argFlyMounts = append(argFlyMounts,
			flyMount{tuple.V.Source, rfs, tuple.M.Path, "none", flags},
		)

		if ro {
			argFlyMounts = append(argFlyMounts,
				flyMount{"", rfs, tuple.M.Path, "none", flags | syscall.MS_REMOUNT | syscall.MS_RDONLY},
			)

			if recursive {
				// Every sub-mount needs to be remounted read-only separately
				mnts, err := mountinfo.ParseMounts(0)
				if err != nil {
					return nil, errwrap.Wrap(fmt.Errorf("error getting mounts under %q from mountinfo", tuple.V.Source), err)
				}
				mnts = mnts.Filter(mountinfo.HasPrefix(tuple.V.Source + "/"))

				for _, mnt := range mnts {
					innerRelPath := tuple.M.Path + strings.Replace(mnt.MountPoint, tuple.V.Source, "", -1)
					argFlyMounts = append(argFlyMounts,
						flyMount{"", rfs, innerRelPath, "none", flags | syscall.MS_REMOUNT | syscall.MS_RDONLY},
					)
				}
			}
		}
	}
	return argFlyMounts, nil
}

func stage1(rp *stage1commontypes.RuntimePod) int {
	uuid, err := types.NewUUID(flag.Arg(0))
	if err != nil {
		log.Print("UUID is missing or malformed\n")
		return 254
	}

	root := "."
	p, err := stage1commontypes.LoadPod(root, uuid, rp)
	if err != nil {
		log.PrintE("can't load pod", err)
		return 254
	}

	if err := p.SaveRuntime(); err != nil {
		log.FatalE("failed to save runtime parameters", err)
	}

	// Sanity checks
	if len(p.Manifest.Apps) != 1 {
		log.Printf("flavor %q only supports 1 application per Pod for now", flavor)
		return 254
	}

	ra := p.Manifest.Apps[0]

	imgName := p.AppNameToImageName(ra.Name)
	args := ra.App.Exec
	if len(args) == 0 {
		log.Printf(`image %q has an empty "exec" (try --exec=BINARY)`, imgName)
		return 254
	}

	lfd, err := common.GetRktLockFD()
	if err != nil {
		log.PrintE("can't get rkt lock fd", err)
		return 254
	}

	workDir := "/"
	if ra.App.WorkingDirectory != "" {
		workDir = ra.App.WorkingDirectory
	}

	env := []string{"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
	for _, e := range ra.App.Environment {
		env = append(env, e.Name+"="+e.Value)
	}

	rfs := filepath.Join(common.AppPath(p.Root, ra.Name), "rootfs")

	argFlyMounts, err := evaluateMounts(rfs, string(ra.Name), p)
	if err != nil {
		log.PrintE("can't evaluate mounts", err)
		return 254
	}

	effectiveMounts := append(
		[]flyMount{
			{"", "", "/dev", "none", syscall.MS_REC | syscall.MS_SHARED},
			{"/dev", rfs, "/dev", "none", syscall.MS_BIND | syscall.MS_REC},

			{"", "", "/proc", "none", syscall.MS_REC | syscall.MS_SHARED},
			{"/proc", rfs, "/proc", "none", syscall.MS_BIND | syscall.MS_REC},

			{"", "", "/sys", "none", syscall.MS_REC | syscall.MS_SHARED},
			{"/sys", rfs, "/sys", "none", syscall.MS_BIND | syscall.MS_REC},

			{"tmpfs", rfs, "/tmp", "tmpfs", 0},
		},
		argFlyMounts...,
	)

	/* Process DNS config files
	 *
	 * /etc/resolv.conf: four modes
	 * 'host' - bind-mount host's file
	 * 'stage0' - bind-mount the file created by stage0
	 * 'default' - do nothing (we would respect CNI if fly had networking)
	 * 'none' - do nothing
	 */
	switch p.ResolvConfMode {
	case "host":
		effectiveMounts = append(effectiveMounts,
			flyMount{"/etc/resolv.conf", rfs, "/etc/resolv.conf", "none", syscall.MS_BIND | syscall.MS_RDONLY})
	case "stage0":
		if err := copyResolv(p); err != nil {
			log.PrintE("can't copy /etc/resolv.conf", err)
			return 254
		}
	}

	/*
	 * /etc/hosts: three modes:
	 * 'host' - bind-mount hosts's file
	 * 'stage0' - bind mount the file created by stage1
	 * 'default' - create a stub /etc/hosts if needed
	 */

	switch p.EtcHostsMode {
	case "host":
		effectiveMounts = append(effectiveMounts,
			flyMount{"/etc/hosts", rfs, "/etc/hosts", "none", syscall.MS_BIND | syscall.MS_RDONLY})
	case "stage0":
		effectiveMounts = append(effectiveMounts, flyMount{
			filepath.Join(common.Stage1RootfsPath(p.Root), "etc", "rkt-hosts"),
			rfs,
			"/etc/hosts",
			"none",
			syscall.MS_BIND | syscall.MS_RDONLY})
	case "default":
		stage2HostsPath := filepath.Join(common.AppRootfsPath(p.Root, ra.Name), "etc", "hosts")
		if _, err := os.Stat(stage2HostsPath); err != nil && os.IsNotExist(err) {
			fallbackHosts := []byte("127.0.0.1 localhost localdomain\n")
			ioutil.WriteFile(stage2HostsPath, fallbackHosts, 0644)
		}
	}

	mounter := fs.NewLoggingMounter(
		fs.MounterFunc(syscall.Mount),
		fs.UnmounterFunc(syscall.Unmount),
		diag.Printf,
	)
	for _, mount := range effectiveMounts {
		var (
			err            error
			hostPathInfo   os.FileInfo
			targetPathInfo os.FileInfo
		)

		if strings.HasPrefix(mount.HostPath, "/") {
			if hostPathInfo, err = os.Stat(mount.HostPath); err != nil {
				log.PrintE(fmt.Sprintf("stat of host path %s", mount.HostPath), err)
				return 254
			}
		} else {
			hostPathInfo = nil
		}

		absTargetPath := mount.RelTargetPath
		if mount.TargetPrefixPath != "" {
			absStage2RootFS := common.AppRootfsPath(p.Root, ra.Name)
			targetPath, err := stage1initcommon.EvaluateSymlinksInsideApp(absStage2RootFS, mount.RelTargetPath)
			if err != nil {
				log.PrintE(fmt.Sprintf("evaluate target path %q in %q", mount.RelTargetPath, absStage2RootFS), err)
				return 254
			}
			absTargetPath = filepath.Join(absStage2RootFS, targetPath)
		}
		if targetPathInfo, err = os.Stat(absTargetPath); err != nil && !os.IsNotExist(err) {
			log.PrintE(fmt.Sprintf("stat of target path %s", absTargetPath), err)
			return 254
		}

		switch {
		case (mount.Flags & syscall.MS_REMOUNT) != 0:
			{
				diag.Printf("don't attempt to create files for remount of %q", absTargetPath)
			}
		case targetPathInfo == nil:
			absTargetPathParent, _ := filepath.Split(absTargetPath)
			if err := os.MkdirAll(absTargetPathParent, 0755); err != nil {
				log.PrintE(fmt.Sprintf("can't create directory %q", absTargetPath), err)
				return 254
			}
			switch {
			case hostPathInfo == nil || hostPathInfo.IsDir():
				if err := os.Mkdir(absTargetPath, 0755); err != nil {
					log.PrintE(fmt.Sprintf("can't create directory %q", absTargetPath), err)
					return 254
				}
			case !hostPathInfo.IsDir():
				file, err := os.OpenFile(absTargetPath, os.O_CREATE, 0700)
				if err != nil {
					log.PrintE(fmt.Sprintf("can't create file %q", absTargetPath), err)
					return 254
				}
				file.Close()
			}
		case hostPathInfo != nil:
			switch {
			case hostPathInfo.IsDir() && !targetPathInfo.IsDir():
				log.Printf("can't mount because %q is a directory while %q is not", mount.HostPath, absTargetPath)
				return 254
			case !hostPathInfo.IsDir() && targetPathInfo.IsDir():
				log.Printf("can't mount because %q is not a directory while %q is", mount.HostPath, absTargetPath)
				return 254
			}
		}

		if err := mounter.Mount(mount.HostPath, absTargetPath, mount.Fs, mount.Flags, ""); err != nil {
			log.PrintE(fmt.Sprintf("can't mount %q on %q with flags %v", mount.HostPath, absTargetPath, mount.Flags), err)
			return 254
		}
	}

	// stage1 interface: pod-leader pid
	if err = stage1common.WritePid(os.Getpid(), "pid"); err != nil {
		log.Error(err)
		return 254
	}

	// stage1-fly internal: pod pgid
	// (used by stop, as fly has weak pod-grouping of processes)
	if err = stage1common.WritePid(syscall.Getpgrp(), "pgid"); err != nil {
		log.Error(err)
		return 254
	}

	var uidResolver, gidResolver user.Resolver
	var uid, gid int

	uidResolver, err = user.NumericIDs(ra.App.User)
	if err != nil {
		uidResolver, err = user.IDsFromStat(rfs, ra.App.User, nil)
	}

	if err != nil { // give up
		log.PrintE(fmt.Sprintf("invalid user %q", ra.App.User), err)
		return 254
	}

	if uid, _, err = uidResolver.IDs(); err != nil {
		log.PrintE(fmt.Sprintf("failed to configure user %q", ra.App.User), err)
		return 254
	}

	gidResolver, err = user.NumericIDs(ra.App.Group)
	if err != nil {
		gidResolver, err = user.IDsFromStat(rfs, ra.App.Group, nil)
	}

	if err != nil { // give up
		log.PrintE(fmt.Sprintf("invalid group %q", ra.App.Group), err)
		return 254
	}

	if _, gid, err = gidResolver.IDs(); err != nil {
		log.PrintE(fmt.Sprintf("failed to configure group %q", ra.App.Group), err)
		return 254
	}

	diag.Printf("chroot to %q", rfs)
	if err := syscall.Chroot(rfs); err != nil {
		log.PrintE("can't chroot", err)
		return 254
	}

	if err := os.Chdir(workDir); err != nil {
		log.PrintE(fmt.Sprintf("can't change to working directory %q", workDir), err)
		return 254
	}

	// lock the current goroutine to its current OS thread.
	// This will force the subsequent syscalls to be executed in the same OS thread as Setresuid, and Setresgid,
	// see https://github.com/golang/go/issues/1435#issuecomment-66054163.
	runtime.LockOSThread()

	// set process credentials
	diag.Printf("setting credentials: uid=%d, gid=%d", uid, gid)
	if err := syscall.Setresgid(gid, gid, gid); err != nil {
		log.PrintE(fmt.Sprintf("can't set gid %d", gid), err)
		return 254
	}
	if err := syscall.Setresuid(uid, uid, uid); err != nil {
		log.PrintE(fmt.Sprintf("can't set uid %d", uid), err)
		return 254
	}

	// clear close-on-exec flag on RKT_LOCK_FD, to keep pod status as running after exec().
	if err := sys.CloseOnExec(lfd, false); err != nil {
		log.PrintE("unable to clear FD_CLOEXEC on pod lock", err)
		return 254
	}

	diag.Printf("execing %q in %q", args, rfs)
	if err = syscall.Exec(args[0], args, env); err != nil {
		log.PrintE(fmt.Sprintf("can't execute %q", args[0]), err)
		return 254
	}

	// unreachable, as successful exec() never returns.
	return 0
}

func copyResolv(p *stage1commontypes.Pod) error {
	ra := p.Manifest.Apps[0]

	stage1Rootfs := common.Stage1RootfsPath(p.Root)
	resolvPath := filepath.Join(stage1Rootfs, "etc", "rkt-resolv.conf")

	appRootfs := common.AppRootfsPath(p.Root, ra.Name)
	targetEtc := filepath.Join(appRootfs, "etc")
	targetResolvPath := filepath.Join(targetEtc, "resolv.conf")

	_, err := os.Stat(resolvPath)
	switch {
	case os.IsNotExist(err):
		return nil
	case err != nil:
		return err
	}

	_, err = os.Stat(targetResolvPath)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	return fileutil.CopyRegularFile(resolvPath, targetResolvPath)
}

func main() {
	rp := parseFlags()

	log, diag, _ = rktlog.NewLogSet("run", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	// move code into stage1() helper so defered fns get run
	os.Exit(stage1(rp))
}
