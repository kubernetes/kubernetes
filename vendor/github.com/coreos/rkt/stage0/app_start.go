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

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/hashicorp/errwrap"
)

type StartConfig struct {
	*CommonConfig
	PodPath     string
	UsesOverlay bool
	AppName     *types.ACName
	PodPID      int
}

func StartApp(cfg StartConfig) error {
	pod, err := pkgPod.PodFromUUIDString(cfg.DataDir, cfg.UUID.String())
	if err != nil {
		return errwrap.Wrap(errors.New("error loading pod"), err)
	}
	defer pod.Close()

	pm, err := pod.SandboxManifest()
	if err != nil {
		return errwrap.Wrap(errors.New("cannot start application"), err)
	}

	app := pm.Apps.Get(*cfg.AppName)
	if app == nil {
		return fmt.Errorf("error: nonexistent app %q", *cfg.AppName)
	}

	args := []string{
		fmt.Sprintf("--debug=%t", cfg.Debug),
		fmt.Sprintf("--app=%s", cfg.AppName),
	}

	appStatusPath := common.AppStatusPath(cfg.PodPath, cfg.AppName.String())
	appStartedPath := common.AppStartedPath(cfg.PodPath, cfg.AppName.String())

	// The app may be restarted. In this case the /rkt/status/app and
	// rkt/status/app-started files already exist.
	// We could touch app-started file and compare its mtime with the app status file
	// to determine the actual status but this won't work if the server time was modified.
	//
	// Instead we:
	// 1. remove the app-started file
	//    In this window rkt app list may report that the application exited.
	// 2. remove the app status file
	//    In this window rkt app list may report that the app was created.
	// 3. re-create the the app-started file

	_ = os.Remove(appStartedPath)
	_ = os.Remove(appStatusPath)

	if _, err := os.Create(appStartedPath); err != nil {
		log.FatalE(fmt.Sprintf("error creating %s-started file", cfg.AppName.String()), err)
	}

	ce := CrossingEntrypoint{
		PodPath:        cfg.PodPath,
		PodPID:         cfg.PodPID,
		AppName:        cfg.AppName.String(),
		EntrypointName: appStartEntrypoint,
		EntrypointArgs: args,
		Interactive:    false,
	}
	if err := ce.Run(); err != nil {
		return err
	}

	return nil
}
