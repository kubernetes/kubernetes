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

package stage0

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/hashicorp/errwrap"
)

type RmConfig struct {
	*CommonConfig
	PodPath     string
	UsesOverlay bool
	AppName     *types.ACName
	PodPID      int
}

func RmApp(cfg RmConfig) error {
	pod, err := pkgPod.PodFromUUIDString(cfg.DataDir, cfg.UUID.String())
	if err != nil {
		return errwrap.Wrap(errors.New("error loading pod"), err)
	}
	defer pod.Close()

	debug("locking sandbox manifest")
	if err := pod.ExclusiveLockManifest(); err != nil {
		return errwrap.Wrap(errors.New("failed to lock sandbox manifest"), err)
	}
	defer pod.UnlockManifest()

	pm, err := pod.SandboxManifest()
	if err != nil {
		return errwrap.Wrap(errors.New("cannot remove application, sandbox validation failed"), err)
	}

	app := pm.Apps.Get(*cfg.AppName)
	if app == nil {
		return fmt.Errorf("error: nonexistent app %q", *cfg.AppName)
	}

	if cfg.PodPID > 0 {
		// Call app-stop and app-rm entrypoint only if the pod is still running.
		// Otherwise, there's not much we can do about it except unmounting/removing
		// the file system.
		args := []string{
			fmt.Sprintf("--debug=%t", cfg.Debug),
			fmt.Sprintf("--app=%s", cfg.AppName),
		}

		ce := CrossingEntrypoint{
			PodPath:        cfg.PodPath,
			PodPID:         cfg.PodPID,
			AppName:        cfg.AppName.String(),
			EntrypointName: appStopEntrypoint,
			EntrypointArgs: args,
			Interactive:    false,
		}
		if err := ce.Run(); err != nil {
			status, err := common.GetExitStatus(err)
			// ignore nonexistent units failing to stop. Exit status 5
			// comes from systemctl and means the unit doesn't exist
			if err != nil {
				return err
			} else if status != 5 {
				return fmt.Errorf("exit status %d", status)
			}
		}

		ce.EntrypointName = appRmEntrypoint
		if err := ce.Run(); err != nil {
			return err
		}
	}

	if cfg.UsesOverlay {
		treeStoreID, err := ioutil.ReadFile(common.AppTreeStoreIDPath(cfg.PodPath, *cfg.AppName))
		if err != nil {
			return err
		}

		appRootfs := common.AppRootfsPath(cfg.PodPath, *cfg.AppName)
		if err := syscall.Unmount(appRootfs, 0); err != nil {
			return err
		}

		ts := filepath.Join(cfg.PodPath, "overlay", string(treeStoreID))
		if err := os.RemoveAll(ts); err != nil {
			return errwrap.Wrap(errors.New("error removing app info directory"), err)
		}
	}

	appInfoDir := common.AppInfoPath(cfg.PodPath, *cfg.AppName)
	if err := os.RemoveAll(appInfoDir); err != nil {
		return errwrap.Wrap(errors.New("error removing app info directory"), err)
	}

	if err := os.RemoveAll(common.AppPath(cfg.PodPath, *cfg.AppName)); err != nil {
		return err
	}

	appStatusPath := filepath.Join(common.Stage1RootfsPath(cfg.PodPath), "rkt", "status", cfg.AppName.String())
	if err := os.Remove(appStatusPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	envPath := filepath.Join(common.Stage1RootfsPath(cfg.PodPath), "rkt", "env", cfg.AppName.String())
	if err := os.Remove(envPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	for i, app := range pm.Apps {
		if app.Name == *cfg.AppName {
			pm.Apps = append(pm.Apps[:i], pm.Apps[i+1:]...)
			break
		}
	}

	return pod.UpdateManifest(pm, cfg.PodPath)
}
