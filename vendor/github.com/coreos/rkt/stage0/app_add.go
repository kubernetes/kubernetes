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
	"os"
	"path/filepath"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/apps"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/pkg/user"
	"github.com/hashicorp/errwrap"
)

type AddConfig struct {
	*CommonConfig
	Image                types.Hash
	Apps                 *apps.Apps
	RktGid               int
	UsesOverlay          bool
	PodPath              string
	PodPID               int
	InsecureCapabilities bool
	InsecurePaths        bool
	InsecureSeccomp      bool
}

func AddApp(cfg AddConfig) error {
	// there should be only one app in the config
	app := cfg.Apps.Last()
	if app == nil {
		return errors.New("no image specified")
	}

	am, err := cfg.Store.GetImageManifest(cfg.Image.String())
	if err != nil {
		return err
	}

	var appName *types.ACName
	if app.Name != "" {
		appName, err = types.NewACName(app.Name)
		if err != nil {
			return err
		}
	} else {
		appName, err = imageNameToAppName(am.Name)
		if err != nil {
			return err
		}
	}

	pod, err := pkgPod.PodFromUUIDString(cfg.DataDir, cfg.UUID.String())
	if err != nil {
		return errwrap.Wrap(errors.New("error loading pod"), err)
	}
	defer pod.Close()

	debug("locking pod manifest")
	if err := pod.ExclusiveLockManifest(); err != nil {
		return errwrap.Wrap(errors.New("failed to lock pod manifest"), err)
	}
	defer pod.UnlockManifest()

	pm, err := pod.SandboxManifest()
	if err != nil {
		return errwrap.Wrap(errors.New("cannot add application"), err)
	}

	if pm.Apps.Get(*appName) != nil {
		return fmt.Errorf("error: multiple apps with name %s", *appName)
	}

	if am.App == nil && app.Exec == "" {
		return fmt.Errorf("error: image %s has no app section and --exec argument is not provided", cfg.Image)
	}

	appInfoDir := common.AppInfoPath(cfg.PodPath, *appName)
	if err := os.MkdirAll(appInfoDir, common.DefaultRegularDirPerm); err != nil {
		return errwrap.Wrap(errors.New("error creating apps info directory"), err)
	}

	pcfg := PrepareConfig{
		CommonConfig: cfg.CommonConfig,
		PrivateUsers: user.NewBlankUidRange(),
	}

	if cfg.UsesOverlay {
		privateUsers, err := preparedWithPrivateUsers(cfg.PodPath)
		if err != nil {
			log.FatalE("error reading user namespace information", err)
		}

		if err := pcfg.PrivateUsers.Deserialize([]byte(privateUsers)); err != nil {
			return err
		}
	}

	treeStoreID, err := prepareAppImage(pcfg, *appName, cfg.Image, cfg.PodPath, cfg.UsesOverlay)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("error preparing image %s", cfg.Image), err)
	}

	rcfg := RunConfig{
		CommonConfig: cfg.CommonConfig,
		UseOverlay:   cfg.UsesOverlay,
		RktGid:       cfg.RktGid,
	}

	if err := setupAppImage(rcfg, *appName, cfg.Image, cfg.PodPath, cfg.UsesOverlay); err != nil {
		return fmt.Errorf("error setting up app image: %v", err)
	}

	if cfg.UsesOverlay {
		imgDir := filepath.Join(cfg.PodPath, "overlay", treeStoreID)
		if err := os.Chown(imgDir, -1, cfg.RktGid); err != nil {
			return err
		}
	}

	ra, err := generateRuntimeApp(app, am, cfg.Apps.Mounts)
	if err != nil {
		return err
	}
	if app.Args != nil {
		ra.App.Exec = append(ra.App.Exec[:1], app.Args...)
	}
	pm.Apps = append(pm.Apps, ra)

	env := ra.App.Environment

	env.Set("AC_APP_NAME", appName.String())
	envFilePath := filepath.Join(common.Stage1RootfsPath(cfg.PodPath), "rkt", "env", appName.String())

	if err := common.WriteEnvFile(env, pcfg.PrivateUsers, envFilePath); err != nil {
		return err
	}

	debug("adding app to sandbox")
	if err := pod.UpdateManifest(pm, cfg.PodPath); err != nil {
		return err
	}

	args := []string{
		fmt.Sprintf("--debug=%t", cfg.Debug),
		fmt.Sprintf("--uuid=%s", cfg.UUID),
		fmt.Sprintf("--app=%s", appName),
	}

	if _, err := os.Create(common.AppCreatedPath(pod.Path(), appName.String())); err != nil {
		return err
	}

	ce := CrossingEntrypoint{
		PodPath:        cfg.PodPath,
		PodPID:         cfg.PodPID,
		AppName:        appName.String(),
		EntrypointName: appAddEntrypoint,
		EntrypointArgs: args,
		Interactive:    false,
	}

	if err := ce.Run(); err != nil {
		return err
	}

	return nil
}
