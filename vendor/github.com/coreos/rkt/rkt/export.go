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

package main

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"

	"github.com/appc/spec/aci"
	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/overlay"
	"github.com/coreos/rkt/pkg/mountinfo"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"
	"github.com/hashicorp/errwrap"
	"github.com/spf13/cobra"
)

var (
	cmdExport = &cobra.Command{
		Use:   "export [--app=APPNAME] UUID OUTPUT_ACI_FILE",
		Short: "Export an app from an exited pod to an ACI file",
		Long:  `UUID should be the uuid of an exited pod.`,
		Run:   runWrapper(runExport),
	}
	flagExportAppName string
)

func init() {
	cmdRkt.AddCommand(cmdExport)
	cmdExport.Flags().StringVar(&flagExportAppName, "app", "", "name of the app to export within the specified pod")
	cmdExport.Flags().BoolVar(&flagOverwriteACI, "overwrite", false, "overwrite output ACI")
}

func runExport(cmd *cobra.Command, args []string) (exit int) {
	if len(args) != 2 {
		cmd.Usage()
		return 254
	}

	outACI := args[1]
	ext := filepath.Ext(outACI)
	if ext != schema.ACIExtension {
		stderr.Printf("extension must be %s (given %s)", schema.ACIExtension, outACI)
		return 254
	}

	p, err := pkgPod.PodFromUUIDString(getDataDir(), args[0])
	if err != nil {
		stderr.PrintE("problem retrieving pod", err)
		return 254
	}
	defer p.Close()

	state := p.State()
	if state != pkgPod.Exited && state != pkgPod.ExitedGarbage {
		stderr.Print("pod is not exited. Only exited pods can be exported")
		return 254
	}

	app, err := getApp(p)
	if err != nil {
		stderr.PrintE("unable to find app", err)
		return 254
	}

	root := common.AppPath(p.Path(), app.Name)
	manifestPath := filepath.Join(common.AppInfoPath(p.Path(), app.Name), aci.ManifestFile)
	if p.UsesOverlay() {
		tmpDir := filepath.Join(getDataDir(), "tmp")
		if err := os.MkdirAll(tmpDir, common.DefaultRegularDirPerm); err != nil {
			stderr.PrintE("unable to create temp directory", err)
			return 254
		}
		podDir, err := ioutil.TempDir(tmpDir, fmt.Sprintf("rkt-export-%s", p.UUID))
		if err != nil {
			stderr.PrintE("unable to create export temp directory", err)
			return 254
		}
		defer func() {
			if err := os.RemoveAll(podDir); err != nil {
				stderr.PrintE("problem removing temp directory", err)
				exit = 1
			}
		}()
		mntDir := filepath.Join(podDir, "rootfs")
		if err := os.Mkdir(mntDir, common.DefaultRegularDirPerm); err != nil {
			stderr.PrintE("unable to create rootfs directory inside temp directory", err)
			return 254
		}

		if err := mountOverlay(p, app, mntDir); err != nil {
			stderr.PrintE(fmt.Sprintf("couldn't mount directory at %s", mntDir), err)
			return 254
		}
		defer func() {
			if err := syscall.Unmount(mntDir, 0); err != nil {
				stderr.PrintE(fmt.Sprintf("error unmounting directory %s", mntDir), err)
				exit = 1
			}
		}()
		root = podDir
	} else {
		// trailing filepath separator so we don't match the appRootfs path
		appRootfs := common.AppRootfsPath(p.Path(), app.Name) + string(filepath.Separator)
		mnts, err := mountinfo.ParseMounts(0)
		if err != nil {
			stderr.PrintE("error parsing mountpoints", err)
			return 254
		}
		mnts = mnts.Filter(mountinfo.HasPrefix(appRootfs))
		if len(mnts) > 0 {
			stderr.Printf("pod has remaining mountpoints. Only pods using overlayfs or with no mountpoints can be exported")
			return 254
		}
	}

	// Check for user namespace (--private-user), if in use get uidRange
	var uidRange *user.UidRange
	privUserFile := filepath.Join(p.Path(), common.PrivateUsersPreparedFilename)
	privUserContent, err := ioutil.ReadFile(privUserFile)
	if err == nil {
		uidRange = user.NewBlankUidRange()
		// The file was found, save uid & gid shift and count
		if err := uidRange.Deserialize(privUserContent); err != nil {
			stderr.PrintE(fmt.Sprintf("problem deserializing the content of %s", common.PrivateUsersPreparedFilename), err)
			return 254
		}
	}

	if err = buildAci(root, manifestPath, outACI, uidRange); err != nil {
		stderr.PrintE("error building aci", err)
		return 254
	}
	return 0
}

// getApp returns the app to export
// If one was supplied in the flags then it's returned if present
// If the PM contains a single app, that app is returned
// If the PM has multiple apps, the names are printed and an error is returned
func getApp(p *pkgPod.Pod) (*schema.RuntimeApp, error) {
	_, manifest, err := p.PodManifest()
	if err != nil {
		return nil, errwrap.Wrap(errors.New("problem getting the pod's manifest"), err)
	}

	apps := manifest.Apps

	if flagExportAppName != "" {
		exportAppName, err := types.NewACName(flagExportAppName)
		if err != nil {
			return nil, err
		}
		for _, ra := range apps {
			if *exportAppName == ra.Name {
				return &ra, nil
			}
		}
		return nil, fmt.Errorf("app %s is not present in pod", flagExportAppName)
	}

	switch len(apps) {
	case 0:
		return nil, fmt.Errorf("pod contains zero apps")
	case 1:
		return &apps[0], nil
	default:
	}

	stderr.Print("pod contains multiple apps:")
	for _, ra := range apps {
		stderr.Printf("\t%v", ra.Name)
	}

	return nil, fmt.Errorf("specify app using \"rkt export --app= ...\"")
}

// mountOverlay mounts the app from the overlay-rendered pod to the destination directory.
func mountOverlay(pod *pkgPod.Pod, app *schema.RuntimeApp, dest string) error {
	if _, err := os.Stat(dest); err != nil {
		return err
	}

	s, err := imagestore.NewStore(getDataDir())
	if err != nil {
		return errwrap.Wrap(errors.New("cannot open store"), err)
	}

	ts, err := treestore.NewStore(treeStoreDir(), s)
	if err != nil {
		return errwrap.Wrap(errors.New("cannot open treestore"), err)
	}

	treeStoreID, err := pod.GetAppTreeStoreID(app.Name)
	if err != nil {
		return err
	}
	lower := ts.GetRootFS(treeStoreID)
	imgDir := filepath.Join(filepath.Join(pod.Path(), "overlay"), treeStoreID)
	if _, err := os.Stat(imgDir); err != nil {
		return err
	}
	upper := filepath.Join(imgDir, "upper", app.Name.String())
	if _, err := os.Stat(upper); err != nil {
		return err
	}
	work := filepath.Join(imgDir, "work", app.Name.String())
	if _, err := os.Stat(work); err != nil {
		return err
	}

	if err := overlay.Mount(&overlay.MountCfg{lower, upper, work, dest, ""}); err != nil {
		return errwrap.Wrap(errors.New("problem mounting overlayfs directory"), err)
	}

	return nil
}

// buildAci builds a target aci from the root directory using any uid shift
// information from uidRange.
func buildAci(root, manifestPath, target string, uidRange *user.UidRange) (e error) {
	mode := os.O_CREATE | os.O_WRONLY
	if flagOverwriteACI {
		mode |= os.O_TRUNC
	} else {
		mode |= os.O_EXCL
	}
	aciFile, err := os.OpenFile(target, mode, 0644)
	if err != nil {
		if os.IsExist(err) {
			return errors.New("target file exists (try --overwrite)")
		} else {
			return errwrap.Wrap(fmt.Errorf("unable to open target %s", target), err)
		}
	}

	gw := gzip.NewWriter(aciFile)
	tr := tar.NewWriter(gw)

	defer func() {
		tr.Close()
		gw.Close()
		aciFile.Close()
		// e is implicitly assigned by the return statement. As defer runs
		// after return, but before actually returning, this works.
		if e != nil {
			os.Remove(target)
		}
	}()

	b, err := ioutil.ReadFile(manifestPath)
	if err != nil {
		return errwrap.Wrap(errors.New("unable to read Image Manifest"), err)
	}
	var im schema.ImageManifest
	if err := im.UnmarshalJSON(b); err != nil {
		return errwrap.Wrap(errors.New("unable to load Image Manifest"), err)
	}
	iw := aci.NewImageWriter(im, tr)

	// Unshift uid and gid when pod was started with --private-user (user namespace)
	var walkerCb aci.TarHeaderWalkFunc = func(hdr *tar.Header) bool {
		if uidRange != nil {
			uid, gid, err := uidRange.UnshiftRange(uint32(hdr.Uid), uint32(hdr.Gid))
			if err != nil {
				stderr.PrintE("error unshifting gid and uid", err)
				return false
			}
			hdr.Uid, hdr.Gid = int(uid), int(gid)
		}
		return true
	}

	if err := filepath.Walk(root, aci.BuildWalker(root, iw, walkerCb)); err != nil {
		return errwrap.Wrap(errors.New("error walking rootfs"), err)
	}

	if err = iw.Close(); err != nil {
		return errwrap.Wrap(fmt.Errorf("unable to close image %s", target), err)
	}

	return
}
