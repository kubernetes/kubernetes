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
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"

	"github.com/coreos/rkt/common"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/pkg/sys"
	stage1common "github.com/coreos/rkt/stage1/common"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"
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

func getHostMounts() (map[string]struct{}, error) {
	hostMounts := map[string]struct{}{}

	mi, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	defer mi.Close()

	sc := bufio.NewScanner(mi)
	for sc.Scan() {
		var (
			discard    string
			mountPoint string
		)

		_, err := fmt.Sscanf(sc.Text(),
			"%s %s %s %s %s",
			&discard, &discard, &discard, &discard, &mountPoint,
		)
		if err != nil {
			return nil, err
		}

		hostMounts[mountPoint] = struct{}{}
	}
	if sc.Err() != nil {
		return nil, errwrap.Wrap(errors.New("problem parsing mountinfo"), sc.Err())
	}
	return hostMounts, nil
}

func init() {
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")

	// The following flags need to be supported by stage1 according to
	// https://github.com/coreos/rkt/blob/master/Documentation/devel/stage1-implementors-guide.md
	// TODO: either implement functionality or give not implemented warnings
	flag.Var(&discardNetlist, "net", "Setup networking")
	flag.BoolVar(&discardBool, "interactive", true, "The pod is interactive")
	flag.StringVar(&discardString, "mds-token", "", "MDS auth token")
	flag.StringVar(&discardString, "local-config", common.DefaultLocalConfigDir, "Local config path")
}

func evaluateMounts(rfs string, app string, p *stage1commontypes.Pod) ([]flyMount, error) {
	imApp := p.Images[app].App
	namedVolumeMounts := map[types.ACName]volumeMountTuple{}

	var manifestMPs []types.MountPoint
	if imApp != nil {
		manifestMPs = imApp.MountPoints
	}

	for _, m := range p.Manifest.Apps[0].Mounts {
		_, exists := namedVolumeMounts[m.Volume]
		if exists {
			return nil, fmt.Errorf("duplicate mount given: %q", m.Volume)
		}
		namedVolumeMounts[m.Volume] = volumeMountTuple{M: m}
		diag.Printf("adding %+v", namedVolumeMounts[m.Volume])
	}

	// Merge command-line Mounts with ImageManifest's MountPoints
	for _, mp := range manifestMPs {
		tuple, exists := namedVolumeMounts[mp.Name]
		switch {
		case exists && tuple.M.Path != mp.Path:
			return nil, fmt.Errorf("conflicting path information from mount and mountpoint %q", mp.Name)
		case !exists:
			namedVolumeMounts[mp.Name] = volumeMountTuple{M: schema.Mount{Volume: mp.Name, Path: mp.Path}}
			diag.Printf("adding %+v", namedVolumeMounts[mp.Name])
		}
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
	for _, mp := range manifestMPs {
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
	hostMounts, err := getHostMounts()
	if err != nil {
		return nil, errwrap.Wrap(errors.New("can't gather host mounts"), err)
	}

	argFlyMounts := []flyMount{}
	var flags uintptr = syscall.MS_BIND // TODO: allow optional | syscall.MS_REC
	for _, tuple := range namedVolumeMounts {
		if _, isHostMount := hostMounts[tuple.V.Source]; isHostMount {
			// Mark the host mount as SHARED so the container's changes to the mount are propagated to the host
			argFlyMounts = append(argFlyMounts,
				flyMount{"", "", tuple.V.Source, "none", syscall.MS_REC | syscall.MS_SHARED},
			)
		}
		argFlyMounts = append(argFlyMounts,
			flyMount{tuple.V.Source, rfs, tuple.M.Path, "none", flags},
		)

		if tuple.V.ReadOnly != nil && *tuple.V.ReadOnly {
			argFlyMounts = append(argFlyMounts,
				flyMount{"", rfs, tuple.M.Path, "none", flags | syscall.MS_REMOUNT | syscall.MS_RDONLY},
			)
		}
	}
	return argFlyMounts, nil
}

func stage1() int {
	uuid, err := types.NewUUID(flag.Arg(0))
	if err != nil {
		log.Print("UUID is missing or malformed\n")
		return 1
	}

	root := "."
	p, err := stage1commontypes.LoadPod(root, uuid)
	if err != nil {
		log.PrintE("can't load pod", err)
		return 1
	}

	// Sanity checks
	if len(p.Manifest.Apps) != 1 {
		log.Printf("flavor %q only supports 1 application per Pod for now", flavor)
		return 1
	}

	ra := p.Manifest.Apps[0]

	imgName := p.AppNameToImageName(ra.Name)
	args := ra.App.Exec
	if len(args) == 0 {
		log.Printf(`image %q has an empty "exec" (try --exec=BINARY)`, imgName)
		return 1
	}

	lfd, err := common.GetRktLockFD()
	if err != nil {
		log.PrintE("can't get rkt lock fd", err)
		return 1
	}

	// set close-on-exec flag on RKT_LOCK_FD so it gets correctly closed after execution is finished
	if err := sys.CloseOnExec(lfd, true); err != nil {
		log.PrintE("can't set FD_CLOEXEC on rkt lock", err)
		return 1
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
		return 1
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

	for _, mount := range effectiveMounts {
		var (
			err            error
			hostPathInfo   os.FileInfo
			targetPathInfo os.FileInfo
		)

		if strings.HasPrefix(mount.HostPath, "/") {
			if hostPathInfo, err = os.Stat(mount.HostPath); err != nil {
				log.PrintE(fmt.Sprintf("stat of host path %s", mount.HostPath), err)
				return 1
			}
		} else {
			hostPathInfo = nil
		}

		absTargetPath := filepath.Join(mount.TargetPrefixPath, mount.RelTargetPath)
		if targetPathInfo, err = os.Stat(absTargetPath); err != nil && !os.IsNotExist(err) {
			log.PrintE(fmt.Sprintf("stat of target path %s", absTargetPath), err)
			return 1
		}

		switch {
		case targetPathInfo == nil:
			absTargetPathParent, _ := filepath.Split(absTargetPath)
			if err := os.MkdirAll(absTargetPathParent, 0700); err != nil {
				log.PrintE(fmt.Sprintf("can't create directory %q", absTargetPath), err)
				return 1
			}
			switch {
			case hostPathInfo == nil || hostPathInfo.IsDir():
				if err := os.Mkdir(absTargetPath, 0700); err != nil {
					log.PrintE(fmt.Sprintf("can't create directory %q", absTargetPath), err)
					return 1
				}
			case !hostPathInfo.IsDir():
				file, err := os.OpenFile(absTargetPath, os.O_CREATE, 0700)
				if err != nil {
					log.PrintE(fmt.Sprintf("can't create file %q", absTargetPath), err)
					return 1
				}
				file.Close()
			}
		case hostPathInfo != nil:
			switch {
			case hostPathInfo.IsDir() && !targetPathInfo.IsDir():
				log.Printf("can't mount because %q is a directory while %q is not", mount.HostPath, absTargetPath)
				return 1
			case !hostPathInfo.IsDir() && targetPathInfo.IsDir():
				log.Printf("can't mount because %q is not a directory while %q is", mount.HostPath, absTargetPath)
				return 1
			}
		}

		if err := syscall.Mount(mount.HostPath, absTargetPath, mount.Fs, mount.Flags, ""); err != nil {
			log.PrintE(fmt.Sprintf("can't mount %q on %q with flags %v", mount.HostPath, absTargetPath, mount.Flags), err)
			return 1
		}
	}

	if err = stage1common.WritePpid(os.Getpid()); err != nil {
		log.Error(err)
		return 4
	}

	diag.Printf("chroot to %q", rfs)
	if err := syscall.Chroot(rfs); err != nil {
		log.PrintE("can't chroot", err)
		return 1
	}

	if err := os.Chdir(workDir); err != nil {
		log.PrintE(fmt.Sprintf("can't change to working directory %q", workDir), err)
		return 1
	}

	diag.Printf("execing %q in %q", args, rfs)
	err = stage1common.WithClearedCloExec(lfd, func() error {
		return syscall.Exec(args[0], args, env)
	})
	if err != nil {
		log.PrintE(fmt.Sprintf("can't execute %q", args[0]), err)
		return 7
	}

	return 0
}

func main() {
	flag.Parse()

	log, diag, _ = rktlog.NewLogSet("run", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	// move code into stage1() helper so defered fns get run
	os.Exit(stage1())
}
